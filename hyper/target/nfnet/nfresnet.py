""" Adapted from https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/resnet.py

Adapted by David Smerkous for usage with our hypernetwork library

STATUS: buggy! do not use

Note: biggest changes are no BN, using weight scaled convolutions, and variance preserving activations

Also, it seems like downsample/shortcut in original pytorch version is wrong
"""
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import math
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param

from hyper.layers.conv import Conv2d, ScaledWConv2d, FinalConv2d, FinalScaledWConv2
from hyper.layers.linear import Linear, FinalLinear
from hyper.generators.base import LayerCodeModelGenerator
from hyper.layers.generators.conv import MLPSampledConvLayerGenerator
from hyper.layers.generators.base import MLPLayerGenerator
from hyper.layers.pool import AvgPool2d
from hyper.net.activation import activation_gamma
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module
from hyper.net.activation import Crater, crater
# from hyper.target.nfnet import StochDepth
from hyper.util.collections import flatten_keys


# replace original WSConv2d with our version
WSConv2D = ScaledWConv2d
WEIGHT_DEBUG = True


__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, act=None, gamma=1.0) -> ScaledWConv2d:
    """3x3 convolution with padding"""
    return WSConv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        act=act,
        gamma=gamma
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, act=None, gamma=None) -> ScaledWConv2d:
    """1x1 convolution"""
    return WSConv2D(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False, act=act, gamma=gamma)


class SharedStochDepth(nn.Module):
  """ This module does not need to be generated. It's fully isolated """
  def __init__(self, stochdepth_rate: float):
    super(SharedStochDepth, self).__init__()

    self.drop_rate = stochdepth_rate

  def forward(self, x):
    if not self.training:
      return x

    batch_size = x.shape[0]
    model_bs = x.shape[1]
    rand_tensor = torch.rand(batch_size, 1, 1, 1, 1).type_as(x).to(x.device)
    keep_prob = 1 - self.drop_rate
    binary_tensor = torch.floor(rand_tensor + keep_prob)

    return x * binary_tensor


class BasicBlock(GenModule):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[GenModule] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        act: str = 'crater',
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        alpha: float = 0.1,
        use_smod: bool = False,
        track: bool=True,
        sdrate: float = 0.0
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.use_stochdepth = sdrate is not None and sdrate > 0. and sdrate < 1.
        if self.use_stochdepth:
            self.stoch_depth = SharedStochDepth(sdrate)
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        conv_layer = conv1x1 if use_smod else conv3x3
        self.conv1 = conv_layer(inplanes, planes, stride, act=act, gamma=act) # REPL act=act
        # self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.act = Activation(act)
        self.dropout = nn.Dropout(p=0.15)
        self.track = track
        self.conv2 = conv_layer(planes, planes, act=None, gamma=act)
        self.skip_gain = nn.Parameter(torch.zeros(())) # alpha*torch.ones(()))  # learnable gain for residuals
        # self.norm = nn.GroupNorm(
        #     num_groups=4,
        #     num_channels=planes,
        #     affine=True
        # )
        
        # self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def is_generated(self):
        """ This module itself does not have any generated parameters but uses generated modules """
        return False

    def define_generated_modules(self):
        mod = super().define_generated_modules()

        # order here matters! for anyone else looking at this
        mod['conv1'] = self.conv1.define_generated_modules()
        mod['conv2'] = self.conv2.define_generated_modules()
        
        if self.downsample is not None:
          mod['downsample'] = self.downsample.define_generated_modules()
        return mod

    def forward(self, params, x: Tensor) -> Tensor:
        identity = x

        features = OrderedDict()
        # features['prekvarin'] = x
        _, out = self.act(None, x)
        # out = self.dropout(out)
        
        # features['prekvarpost'] = out
        _, out = self.conv1(params['conv1'], out)
        # features['conv1']
        
        # features['conv21_prek'] = x
        # mbs, bs, c, h, w = out.shape
        # print('pre shape', out.shape)
        # out = out.reshape(mbs*bs, c, h, w)
        # print('post shape', out.shape)
        # out = self.bn1(out.reshape(mbs*bs, c, h, w)).reshape(mbs, bs, c, h, w)
        # out = self.relu(out)
        # _, out = self.act(None, out)
        
        _, out = self.conv2(params['conv2'], out)
        # features['conv2']
        
        # out = self.dropout(out)
        # features['identity_prek'] = identity
        # mbs, bs, c, h, w = out.shape
        # out = self.bn2(out.reshape(mbs*bs, c, h, w)).reshape(mbs, bs, c, h, w)

        # apply stochastic depth
        if self.use_stochdepth:
            out = self.stoch_depth(out)

        if self.downsample is not None:
            _, identity = self.downsample(params['downsample'], identity)
            # features['downsample']

        # print(self.skip_gain.item())
        out = (self.skip_gain * out) + identity
        out = self.dropout(out)
        # mbs, bs, c, h, w = out.shape
        # out = self.norm((out + identity).reshape(mbs*bs, c, h, w)).reshape(mbs, bs, c, h, w)
        # features['blockout'] = out
        
        # variance preserving residual
        # out = (0.75 * out) + (0.75 * identity)
        
        # features['out_prek'] = identity
        # out = self.relu(out)
        # _, out = self.act(None, out)
        if self.track:
            features['block'] = out
        return features, out


class Bottleneck(GenModule):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[GenModule] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer = None,
        dilation: int = 1,
        act: str = 'crater',
        alpha: float = 0.1,
        use_smod: bool = False,
        track: bool=True
        # norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, act=act, gamma=act)
        # self.bn1 = norm_layer(width)
        if use_smod:
            self.conv2 = conv1x1(width, width, stride, groups, act=act, gamma=act)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation, act=act, gamma=act)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, act=act, gamma=act)
        # self.bn3 = norm_layer(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.skip_gain = nn.Parameter(torch.zeros(()))
        self.act = Activation(act)
        self.downsample = downsample
        self.stride = stride
        self.track = track

    def is_generated(self):
        """ This module itself does not have any generated parameters but uses generated modules """
        return False

    def define_generated_modules(self):
        mod = super().define_generated_modules()

        # order here matters! for anyone else looking at this
        mod['conv1'] = self.conv1.define_generated_modules()
        mod['conv2'] = self.conv2.define_generated_modules()
        mod['conv3'] = self.conv3.define_generated_modules()
        
        if self.downsample is not None:
          mod['downsample'] = self.downsample.define_generated_modules()
        return mod

    def forward(self, params, x: Tensor) -> Tensor:
        identity = x

        features = OrderedDict()
        _, out = self.conv1(params['conv1'], x)
        # features['conv1']
        # out = self.bn1(out)
        # out = self.bn1(out.reshape(mbs*bs, c, h, w)).reshape(mbs, bs, c, h, w)
        # out = self.relu(out)

        _, out = self.conv2(params['conv2'], out)
        # features['conv2']
        # out = self.bn2(out)
        # out = self.relu(out)

        _, out = self.conv3(params['conv3'], out)
        # features['conv3']
        # out = self.bn3(out)

        if self.downsample is not None:
            _, identity = self.downsample(params['downsample'], identity)
            # features['downsample']
            
        out = (self.skip_gain * out) + identity
        # out = self.relu(out)
        # _, out = self.act(None, out)

        if self.track:
            features['block'] = out

        return features, out


class ResNet(GenModule):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,  # unused but keep for easy transfer
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,  # unused but keep for easy transfer
        activation: str = 'crater',
        gamma: float=1.0,
        in_channels: int=3,
        in_planes: int=64,
        feature_step=[1, 1, 1, 1],  # include every nth feature 
        use_smod=False,
        stochdepth_rate=0.25
        # alpha: float = 0.2
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        act = activation
        self.inplanes = in_planes
        self.dilation = 1
        # self.stem_dropout = nn.Dropout(p=0.05)
        self.dropout = nn.Dropout(p=0.2)
        self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0. and stochdepth_rate < 1.
        if self.use_stochdepth:
            self.stoch_depth = SharedStochDepth(stochdepth_rate)
            print('Building resnet with stochdepth of', stochdepth_rate)
        
        self.num_classes = num_classes
        self.feature_step = feature_step
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self._act_name = act
        # self._expected_std = 1.0
        # self._alpha = alpha
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = WSConv2D(
          in_channels,
          self.inplanes,
          kernel_size=7,
          stride=2,
          padding=3,
          bias=False,
          act=None,  # activation
          gamma=activation_gamma(gamma),  # adj for max + pad
          pooling='max',  # apply max pooling (more efficient here than as a separate module)
          pooling_kwargs=dict(
            kernel_size=3,
            stride=2,
            padding=1
          )
        )
        # self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # total number of blocks
        self._num_blocks = sum(layers)
        self._cur_index = 0  # keep track of block index internally
        self._sdrate = stochdepth_rate
        self.layer1 = self._make_layer(block, 64, layers[0], fstep=feature_step[0], sdrate=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], fstep=feature_step[1], sdrate=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], fstep=feature_step[2], sdrate=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1 if use_smod else 2), dilate=replace_stride_with_dilation[2], use_smod=use_smod, fstep=feature_step[3], sdrate=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FinalLinear(512 * block.expansion, num_classes, act=None, gamma=None)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def is_generated(self):
        """ This module itself does not have any generated parameters but uses generated modules """
        return False

    def define_generated_modules(self):
        mod = super().define_generated_modules()

        # order here matters! for anyone else looking at this
        mod['conv1'] = self.conv1.define_generated_modules()
        mod['layer1'] = self.layer1.define_generated_modules()
        mod['layer2'] = self.layer2.define_generated_modules()
        mod['layer3'] = self.layer3.define_generated_modules()
        mod['layer4'] = self.layer4.define_generated_modules()
        mod['fc'] = self.fc.define_generated_modules()
        return mod

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_smod: bool = False,  # modification to 4th layer for small image size datasets like cifar
        fstep: int = 1,
        sdrate: bool = True
    ) -> SequentialModule:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = SequentialModule(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     # norm_layer(planes * block.expansion),
            # )
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, act=None, gamma=None)

        drop_rate = self._cur_index * self._sdrate / self._num_blocks

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, act=self._act_name, use_smod=use_smod, sdrate=drop_rate  # norm_layer
            )
        )
        self._cur_index += 1
        
        self.inplanes = planes * block.expansion
        # print('LAYER', blocks)
        for ind in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    act=self._act_name,
                    norm_layer=norm_layer,
                    use_smod=use_smod,
                    track=((ind) % fstep == 0),
                    sdrate=drop_rate
                )
            )
            self._cur_index += 1


        return SequentialModule(*layers)

    def _forward_impl(self, params, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        feat = OrderedDict()
        feat['conv1'] = x = (self.conv1(params['conv1'], x)[1] - (math.sqrt(2) - 0.05)) * math.sqrt(3)  # adjust dist for max and padding (2 + 1) 
        # x = self.stem_dropout(x)
        # feat['_conv_prek'] = x
        # mbs, bs, c, h, w = x.shape
        # x = self.bn1(x.reshape(mbs*bs, c, h, w)).reshape(mbs, bs, c, h, w)
        # x = torch.relu(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        feat['layer1'], x = self.layer1(params['layer1'], x)
        feat['layer2'], x = self.layer2(params['layer2'], x)
        feat['layer3'], x = self.layer3(params['layer3'], x)
        # print([v.sum() for v in flatten_keys(params['layer4']).values()])
        # print(x.sum(), x.shape)
        feat['layer4'], x = self.layer4(params['layer4'], x)

        # x = self.avgpool(x)
        # out is [B, N, C, H, W] we want avg to (1, 1) over H, W
        h, w = x.shape[-2:]  # know img size before for variance scaling
        if x.shape[-1] > 1:  # mean if feature size greater than 1
            x = torch.mean(x, dim=(3, 4)) * math.sqrt(math.sqrt(h * w) + 2.0)  # 2 for padding
        
        # x = torch.flatten(x, 1)
        b, n = x.shape[:2]
        x = x.reshape(b, n, -1)  # flatten along channels
        x = self.dropout(x)
        feat['fc'], x = self.fc(params['fc'], x)

        return feat, x

    def forward(self, params, x: Tensor) -> Tensor:
        return self._forward_impl(params, x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    DEFAULT = None


class ResNet34_Weights(WeightsEnum):
    DEFAULT = None


class ResNet50_Weights(WeightsEnum):
    DEFAULT = None


class ResNet101_Weights(WeightsEnum):
    DEFAULT = None


class ResNet152_Weights(WeightsEnum):
    DEFAULT = None


class ResNeXt50_32X4D_Weights(WeightsEnum):
    DEFAULT = None


class ResNeXt101_32X8D_Weights(WeightsEnum):
    DEFAULT = None


class ResNeXt101_64X4D_Weights(WeightsEnum):
    DEFAULT = None


class Wide_ResNet50_2_Weights(WeightsEnum):
    DEFAULT = None


class Wide_ResNet101_2_Weights(WeightsEnum):
    DEFAULT = None


def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, feature_step=[2, 1, 1, 1], **kwargs)


def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet34_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet34_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet34_Weights
        :members:
    """
    weights = ResNet34_Weights.verify(weights)

    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, feature_step=[2, 2, 3, 1], **kwargs)

def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, feature_step=[2, 2, 3, 1], **kwargs)


def resnet101(*, weights: Optional[ResNet101_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet101_Weights
        :members:
    """
    weights = ResNet101_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def resnet152(*, weights: Optional[ResNet152_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet152_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet152_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet152_Weights
        :members:
    """
    weights = ResNet152_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)



def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """
    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)



def resnext101_32x8d(
    *, weights: Optional[ResNeXt101_32X8D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 32x8d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_32X8D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_32X8D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_32X8D_Weights
        :members:
    """
    weights = ResNeXt101_32X8D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)



def resnext101_64x4d(
    *, weights: Optional[ResNeXt101_64X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """ResNeXt-101 64x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt101_64X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNeXt101_64X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt101_64X4D_Weights
        :members:
    """
    weights = ResNeXt101_64X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def wide_resnet28_10(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-28-10 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.
    """
    weights = None  # Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    _ovewrite_named_param("in_planes", 16)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)



def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-50-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet50_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet50_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet50_2_Weights
        :members:
    """
    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def wide_resnet101_2(
    *, weights: Optional[Wide_ResNet101_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Wide ResNet-101-2 model from
    `Wide Residual Networks <https://arxiv.org/abs/1605.07146>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-101 has 2048-512-2048
    channels, and in Wide ResNet-101-2 has 2048-1024-2048.

    Args:
        weights (:class:`~torchvision.models.Wide_ResNet101_2_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Wide_ResNet101_2_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.Wide_ResNet101_2_Weights
        :members:
    """
    weights = Wide_ResNet101_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


""" ----  HYPERNETWORK SECTION --- """
""" defintions for layer code generators """
def build_resnet_layer_generator(target: ResNet, code_size: int):

  # basic builder
  def conv_build(mlp_dims):
    return partial(MLPSampledConvLayerGenerator,
                    input_size=code_size,
                    mlp_dims=mlp_dims,
                    bias=True
                  )

  # the individual layer generator for target
  layer_generator = LayerCodeModelGenerator(
    target=target,
    code_size=code_size,
    default_generators={
      ScaledWConv2d: partial(MLPSampledConvLayerGenerator,  # any intermediate layers
        input_size=code_size,
        mlp_dims=[412, 412],
        bias=True
      ),
      FinalLinear: partial(  # final class determinant
        MLPLayerGenerator,
          input_size=code_size,
          mlp_dims=[2 * target.num_classes, 2 * target.num_classes],
          norm_last=True,
          bias=True
      )
    },
    specific_generators={  # scale initial filter samples
      'conv1': conv_build([256, 256])
    }
  )

  return layer_generator
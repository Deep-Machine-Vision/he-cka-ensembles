""" ResNets adapted for our framework and supporting D'Angelo's modifications for a ResNet32 model

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from typing import List, Union
from hyper.layers.linear import Linear, FinalLinear
from hyper.layers.conv import Conv2d
from hyper.layers.norm import BatchNorm2d
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module, register_gen_module, build_gen_module
from hyper.target.convnext import build_norm_layer
from collections import OrderedDict


@register_gen_module('resnet_basic_block')
class BasicBlock(GenModule):
    """ Basic block in a ResNet only supported by FixedEnsemble hyper models """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track: bool=True, act='relu', block_norm=BatchNorm2d, gamma=1.0):
        super(BasicBlock, self).__init__(track=track)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, gamma=gamma)
        
        self.act = Activation(act)
        self.norm1 = block_norm(planes, track=False)
        
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, gamma=act)
        self.norm2 = block_norm(planes, track=False)
        
        self.shortcut = SequentialModule()
        self.use_short = False
        self.stride = stride
        if stride != 1 or in_planes != self.expansion*planes:
            self.use_short = True

            self.fs_in = in_planes
            self.fs_next = self.expansion*planes
            self.shortcut = SequentialModule(
                Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False, gamma=gamma),
                block_norm(self.expansion*planes)
            )
    
    def is_generated(self):
        return False
    
    def define_generated_modules(self):
        mod = super().define_generated_modules()
        mod['conv1'] = self.conv1.define_generated_modules()
        mod['norm1'] = self.norm1.define_generated_modules()
        
        mod['conv2'] = self.conv2.define_generated_modules()
        mod['norm2'] = self.norm2.define_generated_modules()
        
        if self.use_short:
          mod['shortcut'] = self.shortcut.define_generated_modules()
        return mod

    def forward(self, params, x):
        feat = OrderedDict()
        out = self.act(None, self.norm1(params['norm1'], self.conv1(params['conv1'], x)[1])[1])[1]
        out = self.norm2(params['norm2'], self.conv2(params['conv2'], out)[1])[1]

        # run through shortcude module if needed
        if self.use_short:
            out += self.shortcut(params['shortcut'], x)[1]
        else:
            out += x  # simple shortcut

        out = self.act(None, out)[1]

        # add whole block to tracking
        if self._track:
          feat['block'] = out
        return feat, out

    @staticmethod
    def from_config(config):
        return config


class BasicSharedBlock(GenModule):
    """ Same as above but using parameters shared across ensemble members """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track: bool=True, use_norm: bool=True):
        raise NotImplementedError
        if not use_norm:
            raise NotImplementedError()
        super(BasicSharedBlock, self).__init__(track=track)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = build_norm_layer(planes, groups='auto')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm_layer(planes, groups='auto')

        self.use_short = False
        if stride != 1 or in_planes != self.expansion*planes:
            self.use_short = True
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.short_bn = build_norm_layer(self.expansion*planes, groups='auto')

    def forward(self, params, x):
        out = F.relu(self.bn1(None, self.conv1(x).unsqueeze(0))[1].squeeze(0))
        out = self.bn2(None, self.conv2(out).unsqueeze(0))[1].squeeze(0)
        
        # use conv for shortcut or normal shortcut
        if self.use_short:
            out += self.short_bn(None, self.shortcut(x).unsqueeze(0))[1].squeeze(0)
        else:
            out += x
        
        out = F.relu(out)
        return None, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        raise NotImplementedError('not impl yet')
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



@register_gen_module('resnet')
class ResNet(GenModule):
    def __init__(self, block, block_shared=BasicSharedBlock, block_norm=BatchNorm2d, num_blocks=None, num_classes=10, shared_blocks: int=0, in_chan: int=3, planes: int=[64, 64, 128, 256, 512], act: str='relu', track_every_n=[2, 2, 1, 1], gamma: float=1.0):
        super(ResNet, self).__init__()
        first_out = planes[0]
        self.in_planes = first_out
        self._act = act
        self.track_every = track_every_n
        self.block_norm = block_norm

        self.shared = shared_blocks > 0
        if shared_blocks > 0:
            self.conv1 = nn.Conv2d(in_chan, first_out, kernel_size=3, stride=1, padding=1, bias=False)
            shared_blocks -= 1
        else:
            self.conv1 = Conv2d(in_chan, first_out, kernel_size=3, stride=1, padding=1, bias=False, gamma=gamma)
        self.act = Activation(self._act)
        self.norm1 = block_norm(first_out, track=False)
        
        self.num_blocks = num_blocks
        self.layer1 = self._make_layer(block, block_shared, planes[1], num_blocks[0], stride=1, every=self.track_every[0], shared=shared_blocks, act=self._act, gamma=self._act)
        shared_blocks -= num_blocks[0]
        self.layer2 = self._make_layer(block, block_shared, planes[2], num_blocks[1], stride=2, every=self.track_every[1], shared=shared_blocks, act=self._act, gamma=self._act)
        shared_blocks -= num_blocks[1]
        self.layer3 = self._make_layer(block, block_shared, planes[3], num_blocks[2], stride=2, every=self.track_every[2], shared=shared_blocks, act=self._act, gamma=self._act)
        shared_blocks -= num_blocks[2]
        self.layer4 = self._make_layer(block, block_shared, planes[4], num_blocks[3], stride=2, every=self.track_every[3], shared=shared_blocks, act=self._act, gamma=self._act)
        last_planes = planes[4]
        self.linear = Linear(last_planes*block.expansion, num_classes, act=None, gamma=self._act, bias=True)


    def _make_layer(self, block, block_shared, planes, num_blocks, stride, every, shared, act='relu', gamma=1.0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for ind, stride in enumerate(strides):
            use_block = block_shared if ind < shared else block
            layers.append(use_block(self.in_planes, planes, stride, track=(ind % every == 0), act=act, block_norm=self.block_norm, gamma=gamma))
            self.in_planes = planes * block.expansion
        return SequentialModule(*layers)

    def is_generated(self):
        return False
    
    def define_generated_modules(self):
        mod = super().define_generated_modules()
        if not self.shared:
            mod['conv1'] = self.conv1.define_generated_modules()
            mod['norm1'] = self.norm1.define_generated_modules()
            
        mod['layer1'] = self.layer1.define_generated_modules()
        mod['layer2'] = self.layer2.define_generated_modules()
        mod['layer3'] = self.layer3.define_generated_modules()
        mod['layer4'] = self.layer4.define_generated_modules()
        mod['linear'] = self.linear.define_generated_modules()
        return mod

    def forward(self, params, x):
        feat = OrderedDict()
        
        if self.shared:  # use a shared convolution layer
            out = self.act(None, self.bn1(None, self.conv1(x).unsqueeze(0))[1].squeeze(0))[1]
        else:
            feat['conv1'] = out = self.act(None, self.norm1(params['norm1'], self.conv1(params['conv1'], x)[1])[1])[1]
        feat['layer1'], out = self.layer1(params['layer1'], out)
        feat['layer2'], out = self.layer2(params['layer2'], out)
        feat['layer3'], out = self.layer3(params['layer3'], out)
        feat['layer4'], out = self.layer4(params['layer4'], out)
        
        # global average pooling
        # out = torch.mean(out, dim=[-1, -2])
        # out = torch.stack([F.adaptive_avg_pool2d(out[i], (1, 1)) for i in range(5)])
        
        # # for some reason mean/avg_pool2d/adaptive_avg_pool2d produced slightly different results?
        out = torch.vmap(partial(F.avg_pool2d, kernel_size=out.shape[-2:]))(out)
        
        # flatten
        out = out.view(out.shape[0], out.shape[1], -1)
        
        # linear
        feat['linear'], out = self.linear(params['linear'], out)
        return feat, out

    @staticmethod
    def from_config(configs):
        configs['block'] = build_gen_module(configs['block'])
        return configs


@register_gen_module('resnet18')
def resnet18(num_classes=10, **kwargs):
    """ Builds a typical ResNet18 model"""
    return ResNet(
        block=BasicBlock,
        block_shared=BasicSharedBlock,
        block_norm=BatchNorm2d,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        track_every_n=[2, 2, 1, 1],  # default which features to track
        **kwargs
    )


@register_gen_module('resnet32')
def resnet32(num_classes=10, **kwargs):
    """ 6n + 2 blocks Def of ResNet32 following https://github.com/ratschlab/repulsive_ensembles/blob/master/models/mnets_resnet.py """
    raise NotImplementedError('This has not been finished in the rewrite of the code. Come back later')
    return ResNet(BasicBlock, BasicSharedBlock, num_blocks=[5, 5, 5], planes=[16, 16, 32, 64], num_classes=num_classes, dang_mod=True, track_every_n=[3, 3, 2], block_norm=None, **kwargs)


# @TODO finish these
# def resnet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def resnet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def resnet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def resnet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


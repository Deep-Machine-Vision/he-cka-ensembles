""" NFNET models using generated parameters.

File adapted from: https://github.com/benjs/nfnets_pytorch/blob/master/nfnets/model.py
"""
from collections import OrderedDict
from hyper.layers.conv import Conv2d, ScaledWConv2d, FinalConv2d, FinalScaledWConv2
from hyper.layers.linear import Linear, FinalLinear
from hyper.generators.base import LayerCodeModelGenerator
from hyper.layers.generators.conv import MLPSampledConvLayerGenerator
from hyper.layers.generators.base import MLPLayerGenerator
from hyper.layers.pool import AvgPool2d
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module
from hyper.net.activation import Crater, crater
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# replace original WSConv2d with our version
WSConv2D = ScaledWConv2d
WEIGHT_DEBUG = False

nfnet_params = {
  'small': {
    # 'width': [64, 128, 256, 256], 'depth': [1, 2, 3, 2],
    'stem_layers': [16, 32, 48, 64],
    'width': [128, 256, 512], 'depth': [1, 2, 3],
    'group_sizes': [16, 32, 32],
    'train_imsize': 224, 'test_imsize': 256,
    'RA_level': '405', 'drop_rate': 0.2,
    'last_std': 1.0
  },
  'F0': {
    'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
    'train_imsize': 192, 'test_imsize': 256,
    'RA_level': '405', 'drop_rate': 0.2,
    'last_std': 1.0
  },
  'F1': {
    'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
    'train_imsize': 224, 'test_imsize': 320,
    'RA_level': '410', 'drop_rate': 0.3,
    'last_std': 1.0
  },
  'F2': {
    'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
    'train_imsize': 256, 'test_imsize': 352,
    'RA_level': '410', 'drop_rate': 0.4,
    'last_std': 1.0
  },
  'F3': {
    'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
    'train_imsize': 320, 'test_imsize': 416,
    'RA_level': '415', 'drop_rate': 0.4,
    'last_std': 1.0
  },
  'F4': {
    'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
    'train_imsize': 384, 'test_imsize': 512,
    'RA_level': '415', 'drop_rate': 0.5
  },
  'F5': {
    'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
    'train_imsize': 416, 'test_imsize': 544,
    'RA_level': '415', 'drop_rate': 0.5
  },
  'F6': {
    'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
    'train_imsize': 448, 'test_imsize': 576,
    'RA_level': '415', 'drop_rate': 0.5
  },
  'F7': {
    'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
    'train_imsize': 480, 'test_imsize': 608,
    'RA_level': '415', 'drop_rate': 0.5
  },
}


class NFNet(GenModule):
  def __init__(self, num_classes: int, in_channels: int=3, variant: str = 'F0', stochdepth_rate: float = 0.25,
         alpha: float = 0.2, se_ratio: float = 0.5, activation: str = 'crater', gamma: float=1.0):
    super(NFNet, self).__init__()

    if not variant in nfnet_params:
      raise RuntimeError(
        f"Variant {variant} does not exist and could not be loaded.")

    block_params = nfnet_params[variant]

    self.variant = variant
    self.train_imsize = block_params['train_imsize']
    self.test_imsize = block_params['test_imsize']
    self.activation = activation
    self.drop_rate = block_params['drop_rate']
    self.num_classes = num_classes

    # first few layers
    stem_layers_sizes = block_params.get('stem_layers', [16, 32, 64, 128])
    stem_layers = [
      ('conv0', WSConv2D(in_channels=in_channels, out_channels=stem_layers_sizes[0], kernel_size=3, stride=2, act=activation, gamma=gamma)),  # very first conv2d
    ]
    
    for ind in range(1, len(stem_layers_sizes)):
      is_last = ind == (len(stem_layers_sizes) - 1)
      cur_size = stem_layers_sizes[ind]
      prev_size = stem_layers_sizes[ind - 1]
      pad = 0
      if isinstance(stem_layers_sizes[ind - 1], tuple):
        prev_size, _ = stem_layers_sizes[ind - 1]
      
      if isinstance(stem_layers_sizes[ind], tuple):
        cur_size, pad = stem_layers_sizes[ind]
        
      stem_layers.append(
        (f'conv{ind}', WSConv2D(in_channels=prev_size, out_channels=cur_size, kernel_size=3, padding=pad, stride=2 if is_last else 1, act=None if is_last else activation, gamma=activation))
      )
    
    self.stem = SequentialModule(
      OrderedDict(stem_layers)
    )

    num_blocks, index = sum(block_params['depth']), 0

    blocks = []
    expected_std = 1.0
    in_channels = block_params['width'][0] // 2
    if WEIGHT_DEBUG:
      print('FIRST CHANNELS', in_channels)

    block_args = zip(
      block_params['width'],
      block_params['depth'],
      [0.5] * 4,  # bottleneck pattern
      block_params.get('group_sizes', [128] * 4),  # group pattern. Original groups [128] * 4
      block_params.get('strides', [1, 2, 2, 2])  # stride pattern
    )

    body_index = 0
    for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
      for block_index in range(stage_depth):
        beta = 1. / expected_std

        block_sd_rate = stochdepth_rate * index / num_blocks
        out_channels = block_width

        blocks.append((f'group{body_index}_block{block_index}',
          NFBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride if block_index == 0 else 1,
            alpha=alpha,
            beta=beta,
            se_ratio=se_ratio,
            group_size=group_size,
            stochdepth_rate=block_sd_rate,
            activation=activation)
        ))

        in_channels = out_channels
        index += 1

        if block_index == 0:
          expected_std = 1.0

        expected_std = (expected_std ** 2 + alpha**2)**0.5
      body_index += 1

    self.body = SequentialModule(OrderedDict(blocks))

    if WEIGHT_DEBUG:
      print('BODY', self.body)

    final_conv_channels = 2*in_channels
    self.final_conv = WSConv2D(
      in_channels=out_channels, out_channels=final_conv_channels, kernel_size=1, act=activation, gamma=activation)

    if self.drop_rate > 0.:
      self.dropout = nn.Dropout(self.drop_rate)

    self.linear = FinalLinear(final_conv_channels, self.num_classes, bias=True, act=None, gamma=activation)

    # update method to initialize final linear when using ensemble/fixed parameter
    def gen_initialized_params(lin=self.linear, dtype=None, device=None, gain=1.0):
      """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
      params = lin.gen_empty_params(dtype=dtype, device=device)
      nn.init.normal_(params['weight'], 0, block_params.get('last_std', 1.0))
      nn.init.zeros_(params['bias'])
      return params
    self.linear.gen_initialized_params = gen_initialized_params

  def is_generated(self):
    """ This module itself does not have any generated parameters but uses generated modules """
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()

    # order here matters! for anyone else looking at this
    mod['stem'] = self.stem.define_generated_modules()
    mod['body'] = self.body.define_generated_modules()
    mod['final_conv'] = self.final_conv.define_generated_modules()
    mod['linear'] = self.linear.define_generated_modules()
    return mod

  def forward(self, params, x):
    features = OrderedDict()
    features['stem'], out = self.stem(params['stem'], x)
    
    if WEIGHT_DEBUG:
      print('STEM OUT', out.shape)
      features['stem_out'] = out
    
    features['body'], out = self.body(params['body'], out)
    features['final_conv'], out = self.final_conv(params['final_conv'], out)

    # average across W,H
    # h, w = out.shape[-2:]  # know img size before for variance scaling
    pool = torch.mean(out, dim=(3, 4)) #   * math.sqrt(math.sqrt(h * w))

    # do dropout
    if self.training and self.drop_rate > 0.:
      pool = self.dropout(pool)

    # apply final linear
    features['linear'], output = self.linear(params['linear'], pool)

    # @TODO find consistent self convention...
    return features, output

  def exclude_from_weight_decay(self, name: str) -> bool:
    # Regex to find layer names like
    # "stem.6.bias", "stem.6.gain", "body.0.skip_gain",
    # "body.0.conv0.bias", "body.0.conv0.gain"
    regex = re.compile('stem.*(bias|gain)|conv.*(bias|gain)|skip_gain')
    return len(regex.findall(name)) > 0

  def exclude_from_clipping(self, name: str) -> bool:
    # Last layer should not be clipped
    return name.startswith('linear')


class NFBlock(GenModule):
  def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5,
         se_ratio: float = 0.5, stride: int = 1, beta: float = 1.0, alpha: float = 0.2,
         group_size: int = 1, stochdepth_rate: float = None, activation: str = 'crater'):

    super(NFBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.expansion = expansion
    self.se_ratio = se_ratio
    self.activation = Activation(activation, track=False)
    self.beta, self.alpha = beta, alpha
    self.group_size = group_size

    width = int(self.out_channels * expansion)
    self.groups = width // group_size
    self.width = group_size * self.groups
    self.stride = stride

    if WEIGHT_DEBUG:
      print('Block construction', 'i', self.in_channels, 'i2', self.width, 'o', self.out_channels, 'g', self.group_size)


    self.conv0 = WSConv2D(in_channels=self.in_channels,
                out_channels=self.width, kernel_size=1, act=activation, gamma=activation)
    self.conv1 = WSConv2D(in_channels=self.width, out_channels=self.width,
                kernel_size=3, stride=stride, padding=1, groups=self.groups, act=activation, gamma=activation)
    self.conv1b = WSConv2D(in_channels=self.width, out_channels=self.width,
                 kernel_size=3, stride=1, padding=1, groups=self.groups, act=activation, gamma=activation)
    self.conv2 = WSConv2D(in_channels=self.width,
                out_channels=self.out_channels, kernel_size=1, act=None, gamma=activation)

    self.use_projection = self.stride > 1 or self.in_channels != self.out_channels
    if self.use_projection:
      if stride > 1:
        stride_kern = 2
        self.shortcut_avg_pool = AvgPool2d(
          kernel_size=stride_kern, stride=2, padding=0 if self.in_channels == 1536 else 1, track='detach')
        
        # original impl doesn't correct avgpool variance
        # since kernel=2 then variance decreased by 4
        # let's rescale
        self.stride_var_rescale = stride_kern
      self.conv_shortcut = WSConv2D(
        self.in_channels, self.out_channels, kernel_size=1, act=None, gamma=activation)

    self.squeeze_excite = SqueezeExcite(
      self.out_channels, self.out_channels, se_ratio=self.se_ratio, activation=activation)
    
    # unfortunately shared amongst all generated members
    # need to test if it's better with or without
    self.skip_gain = nn.Parameter(torch.ones(()))  # torch.zeros(()))

    self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0. and stochdepth_rate < 1.
    if self.use_stochdepth:
      self.stoch_depth = StochDepth(stochdepth_rate)

  def is_generated(self):
    """ This module itself does not have any generated parameters but uses generated modules """
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()

    # order here matters! for anyone else looking at this
    if self.stride > 1:
      mod['shortcut_pool'] = self.shortcut_avg_pool.define_generated_modules()
      mod['shortcut'] = self.conv_shortcut.define_generated_modules()
    elif self.use_projection:
      mod['shortcut'] = self.conv_shortcut.define_generated_modules()

    # in order define gen modules
    mod['conv0'] = self.conv0.define_generated_modules()
    mod['conv1'] = self.conv1.define_generated_modules()
    mod['conv1b'] = self.conv1b.define_generated_modules()
    mod['conv2'] = self.conv2.define_generated_modules()
    mod['squeeze_excite'] = self.squeeze_excite.define_generated_modules()

    return mod

  def forward(self, params, x):
    # @TODO fix for activations other than crater
    features = OrderedDict()
    out = self.activation(None, x)[1] * self.beta
    
    if WEIGHT_DEBUG:
      features['before'] = x
      features['preact'] = out

      print('INPUT BLOCK', x.shape, 'EXPECTED CHANNELS', self.in_channels)

    if self.stride > 1:
      _, shortcut = self.shortcut_avg_pool(params['shortcut_pool'], out)
      shortcut = shortcut * self.stride_var_rescale
      features['shortcut_pool'], shortcut = shortcut.detach(), shortcut  # no need backprop on first
      features['shortcut'], shortcut = self.conv_shortcut(params['shortcut'], shortcut)
    elif self.use_projection:
      features['shortcut'], shortcut = self.conv_shortcut(params['shortcut'], out)
    else:
      shortcut = x

    features['conv0'], out = self.conv0(params['conv0'], out)
    features['conv1'], out = self.conv1(params['conv1'], out)
    features['conv1b'], out = self.conv1b(params['conv1b'], out)
    features['conv2'], out_conv = self.conv2(params['conv2'], out)
    
    if WEIGHT_DEBUG:
      print('CONV2 IN', out.shape, 'f', self.conv2.out_channels, 'CONV2 OUT', out_conv.shape)
    
    features['squeeze_excite'], out_squeeze = self.squeeze_excite(params['squeeze_excite'], out_conv)
    out = out_squeeze * out_conv

    if self.use_stochdepth:
      out = self.stoch_depth(out)

    if WEIGHT_DEBUG:
      features['PREPOST'] = out
      print('BLOCK OUT SHAPE', out.shape, 'BLOCK SKIP GAIN', self.skip_gain.shape, 'BLOCK SHORTCUT', shortcut.shape)
      features['POST'] = out * self.alpha
  
    out = out * self.alpha * self.skip_gain + shortcut

    # features['self'] = self.track_feature(out)
    return features, out


class SqueezeExcite(GenModule):
  def __init__(self, in_channels: int, out_channels: int, se_ratio: float = 0.5, activation: str = 'crater', track: bool=True):
    super(SqueezeExcite, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.se_ratio = se_ratio

    self.hidden_channels = max(1, int(self.in_channels * self.se_ratio))

    self.activation = activation
    self.linear = Linear(self.in_channels, self.hidden_channels, act=activation, gamma=1.0)  # after conv2 in block not needed
    self.linear_1 = Linear(self.hidden_channels, self.out_channels, act=None, gamma=activation)

  def is_generated(self):
    """ This module itself does not have any generated parameters but uses generated modules """
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()

    # order here matters! for anyone else looking at this
    mod['linear1'] = self.linear.define_generated_modules()
    mod['linear2'] = self.linear_1.define_generated_modules()
    return mod

  def forward(self, params, x):
    # avg pool
    features = OrderedDict()
    if WEIGHT_DEBUG:
      print('SQUEEZE EXCITE IN', x.shape)
      features['pre'] = x
    
    h, w = x.shape[-2:]  # know img size before for variance scaling
    out = torch.mean(x, (3, 4)) * math.sqrt(math.sqrt(h * w))
    if WEIGHT_DEBUG:
      print('SQUEEZE AVG SHAPE', out.shape, 'PREHW', h, w, 'HWS', math.sqrt(h * w))

      # approx variance preserve by rescaling by fixing avg
      print('PREFIX SQUEEZE', (x[:, 0]).mean(dim=1).mean(), (x[:, 0]).var(dim=1).mean())
      print('AVG SQUEEZE', h, (out[:, 0]).mean(dim=1).mean(), (out[:, 0]).var(dim=1).mean())

    features['linear1'], lin1 = self.linear(params['linear1'], out)
    _, out = self.linear_1(params['linear2'], lin1)

    # ideally we want a small cluster around 0.5 on sigmoid output
    # reason being it's easy to scale output to be close to 1.0
    # such that we don't have to adjust variance of multiplied features
    # note the magic number is to adjust output distribution to have expectation 1.0
    # @TODO find a constant for sigmoid to adjust input variance of squeeze selection
    out = (torch.sigmoid(0.4 * out) * 2.0) + 0.0339855078258
    features['linear2'] = out  # save for tracking

    b, ms, c, _, _ = x.size()
    output = out.view(b, ms, c, 1, 1).expand_as(x)
    # features['self'] = self.track_feature(output)

    if WEIGHT_DEBUG:
      print('SQUEEZE EXCITE OUT', output.shape)

    return features, output


class StochDepth(nn.Module):
  """ This module does not need to be generated. It's fully isolated """
  def __init__(self, stochdepth_rate: float):
    super(StochDepth, self).__init__()

    self.drop_rate = stochdepth_rate

  def forward(self, x):
    if not self.training:
      return x

    batch_size = x.shape[0]
    model_bs = x.shape[1]
    rand_tensor = torch.rand(batch_size, model_bs, 1, 1, 1).type_as(x).to(x.device)
    keep_prob = 1 - self.drop_rate
    binary_tensor = torch.floor(rand_tensor + keep_prob)

    return x * binary_tensor


""" ----  HYPERNETWORK SECTION --- """
""" defintions for layer code generators """
def build_nfnet_layer_generator(target: NFNet, code_size: int):

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
        mlp_dims=[128, 128],
        bias=True
      ),
      Linear: partial(
        MLPLayerGenerator,  # squeeze excite layers
          input_size=code_size,
          mlp_dims=[128, 128],
          norm_last=True,
          bias=True
      ),
      FinalLinear: partial(  # final class determinant
        MLPLayerGenerator,
          input_size=code_size,
          mlp_dims=[96, 96],
          norm_last=True,
          bias=True
      )
    },
    specific_generators={  # scale initial filter samples
      'stem.conv0': conv_build([112, 112]),
      'stem.conv1': conv_build([128, 128]),
      'stem.conv2': conv_build([148, 148]),
      'stem.conv3': conv_build([148, 148]),
      'final_conv': conv_build([164, 164])
    }
  )

  return layer_generator
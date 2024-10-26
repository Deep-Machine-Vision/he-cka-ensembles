# Modified to work with hypernetworks by David Smerkous
#
# Original source:
# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Union
from hyper.layers.linear import Linear, FinalLinear
from hyper.layers.conv import Conv2d
from hyper.layers.dropout import Dropout
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module
from collections import OrderedDict
from dataclasses import dataclass
import math

@dataclass
class DownSampleConfig:
  kernel_size: int = 2
  stride: int = 2
  padding: int = 0


@dataclass
class BlockConfig:
  kernel_size: int = 7
  padding: int = 3
  activation: str = 'gelu'
  layers: int = 3
  dims: int = 96
  dim_multiplier: int = 4
  track_every: int = 1
  downsample: DownSampleConfig = DownSampleConfig(kernel_size=2, stride=2)



def build_norm_layer(channels: int, groups: Union[int, str] = 'auto', auto_default: int = 64, min_chan_per_group: int=8, min_groups: int = 1, affine: bool=False, eps: float=1e-6):
  """ Construct the group norm and in auto mode tries to automatically determine an okay configuration using results described in
  The original GroupNorm paper https://arxiv.org/pdf/1803.08494.pdf
  """
  if isinstance(groups, str):
    if groups == 'auto':
      if (channels / auto_default) < min_chan_per_group:
        groups = math.ceil(channels / min_chan_per_group) 
      else:
        groups = auto_default
    else:
      raise ValueError(f'Invalid groupnorm option {groups}. Available is "auto"')

  # crude auto adjust if channels are not divisible by groups 
  passed = False
  for diff in range(0, 64):
    test_lower_group = groups - diff
    test_upper_group = groups + diff

    # diverge from expected number of groups until we get a match (prefer smaller group numbers)
    if test_lower_group > 0 and channels % test_lower_group == 0 and test_lower_group >= min_groups:
      groups = test_lower_group
      passed = True
    elif channels % test_upper_group == 0 and test_upper_group <= channels and (math.ceil(channels / test_upper_group) > min_chan_per_group):
      groups = test_upper_group
      passed = True

  # error if we fail
  if not passed:
    raise RuntimeError(f'Could not auto calculate groups for channels {channels} and selected target group {groups}')

  # construct groupnorm otherwise
  return GroupNorm(
    num_groups=groups,
    num_channels=channels,
    eps=eps,
    affine=affine
  )
  

class Block(GenModule):
  r""" ConvNeXt Block. There are two equivalent implementations:
  (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (P, N, C, H, W)
  (2) DwConv -> Permute to (P, N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
  We use (2) as we find it slightly faster in PyTorch
  
  Args:
      dim (int): Number of input channels.
      drop_path (float): Stochastic depth rate. Default: 0.0
      layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
  """
  def __init__(self, config: BlockConfig=BlockConfig(), drop_path=0., layer_scale_init_value=1e-6, track: bool=True):
    super().__init__(track=track)
    
    # unpack config
    dim = config.dims
    dim_mult = config.dim_multiplier
    
    self.dwconv = Conv2d(dim, dim, kernel_size=config.kernel_size, padding=config.padding, groups=dim, gamma=1.0, act=None) # depthwise conv
    self.norm = build_norm_layer(dim, groups='auto')
    self.pwconv1 = Conv2d(dim, dim_mult * dim, kernel_size=1, padding=0, stride=1, gamma=1.0, act=config.activation)   # Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
    self.pwconv2 = Conv2d(dim_mult * dim, dim, kernel_size=1, padding=0, stride=1, gamma=config.activation, act=None)  # Linear(4 * dim, dim)
    self.drop_path = drop_path
    
    # @TODO look into removed shared gain parameter across ensemble members
    self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, 1, dim, 1, 1)), requires_grad=True) if layer_scale_init_value > 0 else None
    # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
  
  def is_generated(self):
    return False
  
  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['dwconv'] = self.dwconv.define_generated_modules()
    mod['pwconv1'] = self.pwconv1.define_generated_modules()
    mod['pwconv2'] = self.pwconv2.define_generated_modules()
    return mod

  def forward(self, params, x):
    input = x
    
    feat = OrderedDict()
    
    # residual with drop path/stochdepth
    if self.training and self.drop_path > 0.0 and (random.random() < self.drop_path):
      x = input  # keep only residual
      feat['block'] = x.detach()  # previous layer taken care of
    else:
      # feat['prek_pre'] = x  # debug
      _, x = self.dwconv(params['dwconv'], x)
      _, x = self.norm(None, x)
      _, x = self.pwconv1(params['pwconv1'], x)
      _, x = self.pwconv2(params['pwconv2'], x)
      if self.gamma is not None:
        x = self.gamma * x
       
      # we only track post residual
      if self._track:
        feat['block'] = x
      
      # residual connection
      try:
        x = input + x
      except RuntimeError as err:
        print('Got runtime error', err)
        print('Shapes', input.shape, 'x', x.shape)
        raise Exception()
    return feat, x


class ConvNeXt(GenModule):
  r""" ConvNeXt
      A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

  Args:
      in_chans (int): Number of input image channels. Default: 3
      num_classes (int): Number of classes for classification head. Default: 1000
      depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
      dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
      drop_path_rate (float): Stochastic depth rate. Default: 0.
      layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
      head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
  """
  def __init__(self, in_chans=3, num_classes=1000,
                blocks: List[BlockConfig] = [
                  BlockConfig(layers=3, dims=96, downsample=DownSampleConfig(kernel_size=4, stride=2)),  # stem
                  BlockConfig(layers=3, dims=192),
                  BlockConfig(layers=9, dims=384),
                  BlockConfig(layers=3, dims=768)
                ],
                # num_groups: Union[int, str] = 'auto',
                gamma: float=1.0,  # initial variance scaling parameter
                drop_path_rate=0., layer_scale_init_value=1e-6
                ):
    super().__init__()

    # len(block) stages, each consisting of multiple residual blocks
    stages = []
    dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum([c.layers for c in blocks]))] 
    cur = 0
    self.blocks = blocks
    for i in range(len(blocks)):
      down_config = blocks[i].downsample
      if i == 0:  # stem
        down = [
          Conv2d(in_chans, blocks[0].dims, kernel_size=down_config.kernel_size, stride=down_config.stride, padding=down_config.padding, gamma=gamma),
          build_norm_layer(blocks[0].dims, groups='auto')
        ]
      else:  # 3 intermediate downsampling conv layers
        if down_config is None:
          down = []
        else:
          down = [
            build_norm_layer(blocks[i-1].dims, groups='auto'),
            Conv2d(blocks[i-1].dims, blocks[i].dims, kernel_size=down_config.kernel_size, stride=down_config.stride, padding=down_config.padding, gamma=1.0, track=False)
          ]
      
      # combining downsampling and stage layers from orig impl
      stage = SequentialModule(
        # downsampling layer
        *down,
       
        # stage layers 
        *[
          Block(config=blocks[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, track=(j % blocks[i].track_every == 0)) 
          for j in range(blocks[i].layers)
        ]
      )
      stages.append(stage)
      cur += blocks[i].layers

    self.stages = SequentialModule(*stages, track=True)
    self.norm = build_norm_layer(blocks[-1].dims, groups='auto') # final norm layer
    self.head = Linear(blocks[-1].dims, num_classes)

  def is_generated(self):
    return False
  
  def define_generated_modules(self):
    # NOTE: for anyone looking at this order DOES matter
    # all layercode based hypernetworks conditioned on previous layer codes need ordering
    mod = super().define_generated_modules()
    mod['stages'] = self.stages.define_generated_modules()
    mod['head'] = self.head.define_generated_modules()
    return mod

  def forward_features(self, params, x):
    feat, x = self.stages(params['stages'], x)
    _, x = self.norm(None, x.mean([-2, -1])) # global average pooling, (P, N, C, H, W) -> (P, N, C)
    return feat, x

  def forward(self, params, x):
    feat = OrderedDict()
    # feat['prek_input'] = x  # debug
    feat['stages'], x = self.forward_features(params, x)
    feat['head'], x = self.head(params['head'], x)
    return feat, x


class LayerNorm(GenModule):
  r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
  The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
  shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
  with shape (batch_size, channels, height, width).
  """
  def __init__(self, normalized_shape, eps=1e-6, affine=True, data_format="channels_last"):
    super().__init__()
    
    if affine:
      self.affine_weight = nn.Parameter(torch.ones(normalized_shape))
      self.affine_bias = nn.Parameter(torch.zeros(normalized_shape))
    else:
      self.affine_weight = None
      self.affine_bias = None
    self.affine = affine
    self.eps = eps
    self.data_format = data_format
    if self.data_format not in ["channels_last", "channels_first"]:
      raise NotImplementedError 
    self.normalized_shape = (normalized_shape, )
  
  def is_generated(self):
    return False
  
  def forward(self, params, x):
    if self.data_format == "channels_last":
      return None, F.layer_norm(x, self.normalized_shape, self.affine_weight, self.affine_bias, self.eps)
    elif self.data_format == "channels_first": # (P, N, C, H, W)
      u = x.mean(2, keepdim=True)
      s = (x - u).pow(2).mean(2, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.eps)
      
      if self.affine:
        if x.ndim == 5:  # Conv2d
          x = self.affine_weight[None, None, :, None, None] * x + self.affine_bias[None, None, :, None, None]
        else:  # Linear
          x = self.affine_weight[None, None, :] * x + self.affine_bias[None, None, :]
      return None, x


class GroupNorm(GenModule):
  r""" GroupNorm to support batched grouping
  """
  def __init__(self, num_groups, num_channels, eps=1e-6, affine: bool=False, track: bool=False):
    super().__init__(track=track)
    
    if affine:
      # self.affine_weight = nn.Parameter(torch.ones(normalized_shape))
      # self.affine_bias = nn.Parameter(torch.zeros(normalized_shape))
      raise NotImplementedError('TODO')
    else:
      self.affine_weight = None
      self.affine_bias = None
    self.num_groups = num_groups
    self.num_channels = num_channels
    
    if self.num_channels % self.num_groups != 0:
      raise ValueError('The provided num channels is not divisible by the number of groups!')
    self.affine = affine
    self.batch_gn = torch.func.vmap(F.group_norm)
    self.eps = eps
  
  def is_generated(self):
    return False
  
  def forward(self, params, x):
    # since it's channel-wise we can collapse batch and particle dim
    mbs, bs = x.shape[:2]
    other = x.shape[2:]
    y = F.group_norm(x.reshape(mbs * bs, *other), self.num_groups, weight=self.affine_weight, bias=self.affine_bias, eps=self.eps).reshape(mbs, bs, *other)
    return self.track_feature(y), y


def convnext_imsmall_tiny(**kwargs):
  downsample = DownSampleConfig(kernel_size=3, stride=2, padding=1)
  model = ConvNeXt(
    blocks=[
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=80, downsample=DownSampleConfig(kernel_size=5, stride=2, padding=2)),
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=160, downsample=downsample),
      # BlockConfig(kernel_size=3, padding=1, layers=5, dims=320, downsample=downsample),
      # BlockConfig(kernel_size=1, padding=0, layers=2, dims=640, downsample=DownSampleConfig(kernel_size=2, stride=2, padding=0))  
      
      # LARGER
      BlockConfig(kernel_size=3, padding=1, layers=0, dims=9, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1)),
      BlockConfig(kernel_size=3, padding=1, layers=3, dims=12, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1), track_every=2),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=24, downsample=downsample, track_every=2),
      BlockConfig(kernel_size=3, padding=1, layers=5, dims=48, downsample=downsample, track_every=1),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=96, downsample=downsample, track_every=1)  
    ],
    layer_scale_init_value=1e-4,
    **kwargs
  )
  return model


def convnext_imsmall_small(**kwargs):
  downsample = DownSampleConfig(kernel_size=3, stride=2, padding=1)
  model = ConvNeXt(
    blocks=[
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=80, downsample=DownSampleConfig(kernel_size=5, stride=2, padding=2)),
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=160, downsample=downsample),
      # BlockConfig(kernel_size=3, padding=1, layers=5, dims=320, downsample=downsample),
      # BlockConfig(kernel_size=1, padding=0, layers=2, dims=640, downsample=DownSampleConfig(kernel_size=2, stride=2, padding=0))  
      
      # LARGER
      BlockConfig(kernel_size=3, padding=1, layers=0, dims=18, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1)),
      BlockConfig(kernel_size=3, padding=1, layers=3, dims=36, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1)),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=62, downsample=downsample),
      BlockConfig(kernel_size=3, padding=1, layers=5, dims=144, downsample=downsample),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=288, downsample=downsample)  
    ],
    layer_scale_init_value=1e-4,
    **kwargs
  )
  return model



def convnext_imsmall_medium(**kwargs):
  downsample = DownSampleConfig(kernel_size=3, stride=2, padding=1)
  model = ConvNeXt(
    blocks=[
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=80, downsample=DownSampleConfig(kernel_size=5, stride=2, padding=2)),
      # BlockConfig(kernel_size=3, padding=1, layers=3, dims=160, downsample=downsample),
      # BlockConfig(kernel_size=3, padding=1, layers=5, dims=320, downsample=downsample),
      # BlockConfig(kernel_size=1, padding=0, layers=2, dims=640, downsample=DownSampleConfig(kernel_size=2, stride=2, padding=0))  
      
      # LARGER
      BlockConfig(kernel_size=3, padding=1, layers=0, dims=28, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1)),
      BlockConfig(kernel_size=3, padding=1, layers=3, dims=56, downsample=DownSampleConfig(kernel_size=3, stride=1, padding=1)),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=112, downsample=downsample),
      BlockConfig(kernel_size=3, padding=1, layers=5, dims=224, downsample=downsample),
      BlockConfig(kernel_size=3, padding=1, layers=4, dims=356, downsample=downsample)  
    ],
    layer_scale_init_value=1e-4,
    **kwargs
  )
  return model

# @TODO add these configs in
# def convnext_atto(**kwargs):
#     model = ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
#     return model

# def convnext_femto(**kwargs):
#     model = ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
#     return model

# def convnext_pico(**kwargs):
#     model = ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
#     return model

# def convnext_micro(**kwargs):
#     model = ConvNeXt(depths=[2, 2, 2, 2], dims=[64, 128, 256, 512], **kwargs)
#     return model

# def convnext_nano(**kwargs):
#     model = ConvNeXt(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
#     return model

# def convnext_tiny(**kwargs):
#     model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
#     return model

# def convnext_small(**kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     return model

# def convnext_base(**kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     return model

# def convnext_large(**kwargs):
#     model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     return model

# def convnext_xlarge(**kwargs):
#   model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#   return model

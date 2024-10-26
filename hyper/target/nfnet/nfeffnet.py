""" A batchable normalization free implementation of EfficientNet

Reference: https://raw.githubusercontent.com/shenghaoG/CIFAR10-ResNet18/main/models/efficientnet.py
Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
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
from hyper.target.nfnet import StochDepth, SqueezeExcite
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


# replace original WSConv2d with our version
WSConv2D = ScaledWConv2d
WEIGHT_DEBUG = True


def swish(x):
  return x * x.sigmoid()


# def drop_connect(x, drop_ratio):
#   keep_ratio = 1.0 - drop_ratio
#   mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
#   mask.bernoulli_(keep_ratio)
#   x.div_(keep_ratio)
#   x.mul_(mask)
#   return x


class SE(GenModule):
  def __init__(self, in_channels, se_channels, track: bool=True):
    super(SE, self).__init__(track=track)
    self.se1 = WSConv2D(in_channels, se_channels,
                          kernel_size=1, act='crater', gamma='crater', bias=True)
    self.se2 = WSConv2D(se_channels, in_channels,
                          kernel_size=1, act=None, gamma='crater', bias=True)

  def is_generated(self):
    return False
  
  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['se1'] = self.se1.define_generated_modules()
    mod['se2'] = self.se2.define_generated_modules()
    return mod

  def forward(self, params, x):
    feat = OrderedDict()
    h, w = x.shape[-2:]  # know img size before for variance scaling
    
    out = torch.mean(x, (3, 4)).reshape(*x.shape[:3], 1, 1) * math.sqrt(math.sqrt(h * w))
    if WEIGHT_DEBUG:
      feat['mean'] = out
    feat['se1'], out = self.se1(params['se1'], out)
    feat['se2'], out = self.se2(params['se2'], out)
    
    # ideally we want a small cluster around 0.5 on sigmoid output
    # reason being it's easy to scale output to be close to 1.0
    # such that we don't have to adjust variance of multiplied features
    # @TODO find a constant for sigmoid to adjust input variance of squeeze selection
    if WEIGHT_DEBUG:
      feat['se_dist'] = (torch.sigmoid(0.4 * out) * 2.0)
    out = x * (torch.sigmoid(0.4 * out) * 2.0)
    return self.track_feature(feat), out


class Block(GenModule):
  def __init__(self,
              in_channels,
              out_channels,
              kernel_size,
              stride,
              expand_ratio=1,
              se_ratio=0.,
              drop_rate=0.,
              gamma='crater'):
    super(Block, self).__init__()
    self.stride = stride
    self.drop_rate = drop_rate
    self.expand_ratio = expand_ratio

    # Expansion
    channels = expand_ratio * in_channels
    self.conv1 = WSConv2D(in_channels,
                            channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act='crater',
                            gamma=gamma,
                            bias=False)

    # Depthwise conv
    self.conv2 = WSConv2D(channels,
                            channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(1 if kernel_size == 3 else 2),
                            groups=channels,
                            act='crater',
                            gamma='crater',
                            bias=False)

    # SE layers
    se_channels = int(in_channels * se_ratio)
    self.se = SE(channels, se_channels)
    self.skip_gain = nn.Parameter(torch.zeros(()))
    self.stoch_d = StochDepth(drop_rate)

    # Output
    self.conv3 = WSConv2D(channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act=None,
                            gamma='crater',
                            bias=False)

    # Skip connection if in and out shapes are the same (MV-V2 style)
    self.has_skip = (stride == 1) and (in_channels == out_channels)

  def is_generated(self):
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['conv1'] = self.conv1.define_generated_modules()
    mod['conv2'] = self.conv2.define_generated_modules()
    mod['conv3'] = self.conv3.define_generated_modules()
    mod['squeeze_excite'] = self.se.define_generated_modules()
    return mod

  def forward(self, params, x):
    feat = OrderedDict()
    feat['conv1'], out = (None, x) if self.expand_ratio == 1 else self.conv1(params['conv1'], x)
    feat['conv2'], out = self.conv2(params['conv2'], out)
    feat['squeeze_excite'], out = self.se(params['squeeze_excite'], out)
    feat['conv3'], out = self.conv3(params['conv3'], out)
    if self.has_skip:
      if self.training and self.drop_rate > 0:
        out = self.stoch_d(out)
      out = (out * self.skip_gain) + x
    
    if WEIGHT_DEBUG:
      feat['residual'] = out
    return feat, out


class EfficientNet(GenModule):
  def __init__(self, cfg, in_channels=3, num_classes=10, gamma=1.0):
    super(EfficientNet, self).__init__()
    self.cfg = cfg
    self.conv1 = WSConv2D(in_channels,
                          32,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          act='crater',
                          gamma=gamma,
                          bias=False)
    self.layers = self._make_layers(in_channels=32)
    self.dropout = nn.Dropout(p=cfg['dropout_rate'])
    self.linear = FinalLinear(cfg['out_channels'][-1], num_classes, act=None, gamma='crater', bias=True)

  def is_generated(self):
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['conv1'] = self.conv1.define_generated_modules()
    mod['layers'] = self.layers.define_generated_modules()
    mod['linear'] = self.linear.define_generated_modules()
    return mod

  def _make_layers(self, in_channels):
    layers = []
    cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                  'stride']]
    b = 0
    blocks = sum(self.cfg['num_blocks'])
    gamma = 'crater'
    for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
      strides = [stride] + [1] * (num_blocks - 1)
      for stride in strides:
        drop_rate = self.cfg['drop_connect_rate'] * b / blocks
        layers.append(
          Block(in_channels,
                out_channels,
                kernel_size,
                stride,
                expansion,
                se_ratio=0.25,
                drop_rate=drop_rate,
                gamma=gamma)
        )
        gamma = None
        in_channels = out_channels
    return SequentialModule(*layers)

  def forward(self, params, x):
    feat = OrderedDict()
    feat['conv1'], out = self.conv1(params['conv1'], x)
    feat['layers'], out = self.layers(params['layers'], out)
    h, w = x.shape[-2:]
    out = torch.mean(out, (3, 4)) * math.sqrt(math.sqrt(h * w))
    out = self.dropout(out)
    out = out.view(out.shape[0], out.shape[1], -1)
    feat['linear'], out = self.linear(params['linear'], out)
    return self.track_feature(feat), out


def EfficientNetSmall():
  cfg = {
    'num_blocks': [1, 2, 2, 3],
    'expansion': [1, 6, 6, 6],
    'out_channels': [16, 24, 40, 80],
    'kernel_size': [3, 3, 5, 3],
    'stride': [1, 2, 2, 2],
    'dropout_rate': 0.2,
    'drop_connect_rate': 0.2,
  }
  return EfficientNet(cfg)


def EfficientNetB0():
  cfg = {
    'num_blocks': [1, 2, 2, 3, 3, 4, 1],
    'expansion': [1, 6, 6, 6, 6, 6, 6],
    'out_channels': [16, 24, 40, 80, 112, 192, 320],
    'kernel_size': [3, 3, 5, 3, 5, 5, 3],
    'stride': [1, 2, 2, 2, 1, 2, 1],
    'dropout_rate': 0.2,
    'drop_connect_rate': 0.2,
  }
  return EfficientNet(cfg)

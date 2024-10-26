""" A smallish (medium) normalization free resnet model used for CIFAR-10 classification. """

from hyper.layers.linear import Linear, FinalLinear
from hyper.layers.conv import ScaledWConv2d as WSConv2D
from hyper.layers.dropout import Dropout
from hyper.target.nfnet import StochDepth
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module
from collections import OrderedDict
import torch.nn as nn
import torch
import math


class Block(GenModule):
  expansion = 1
  
  def __init__(self, in_chan, out_chan, res: bool=True, stride=1, dropout=0.0, mc_dropout: bool=False, activation: str='crater', stochdepth_rate=0.25, track: bool = True):
    super().__init__(track)
    
    self.act = Activation(activation)
    self.pad = 1 if stride == 1 else 2
    self.conv1 = WSConv2D(in_chan, out_chan, kernel_size=3 if stride == 1 else 5, padding=self.pad, stride=stride, act=activation, gamma=activation, bias=False)  # , track=False)  # @REVIEW TRACKING?
    self.conv2 = WSConv2D(out_chan, out_chan, kernel_size=3, padding=1, act=None, gamma=activation, bias=False)
    self.dropout = Dropout(p=dropout, mc_dropout=mc_dropout, track=False)

    self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0. and stochdepth_rate < 1.
    if self.use_stochdepth:
      self.stoch_depth = StochDepth(stochdepth_rate)

    if self.use_stochdepth and not res:
      raise ValueError('Stochastic depth does not make sense in a non-residual network')

    self.res = res    
    if res:
      self.use_shortcut = stride != 1 or in_chan != self.expansion*out_chan 
      if self.use_shortcut:
        self.shortcut = WSConv2D(
          in_chan, self.expansion*out_chan, kernel_size=1, stride=stride, gamma=None, act=None, bias=False
        )
    else:
      self.use_shortcut = False

  def is_generated(self):
    return False

  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['conv1'] = self.conv1.define_generated_modules()
    mod['conv2'] = self.conv2.define_generated_modules()
    if self.res and self.use_shortcut:
      mod['shortcut'] = self.shortcut.define_generated_modules()
    return mod

  def forward(self, params, x):
    feat = OrderedDict()
    h, _ = x.shape[-2:]
    out = self.act(None, x)[1] 
    feat['conv1'] = out = self.conv1(params['conv1'], out)[1] * math.sqrt((h + 1.0) / h)
    h, _ = out.shape[-2:]
    feat['conv2'] = out = self.conv2(params['conv2'], out)[1]
    
    # zero out feats with prob stoch depth
    if self.use_stochdepth:
      out = self.stoch_depth(out)
    
    # use a residual connection or not
    if self.res:
      if self.use_shortcut:
        _, short = self.shortcut(params['shortcut'], x)
      else:
        short = x
      
      # initial variance preserving shortcut assuming at init both have unit variance
      out = (0.75 * out) + (0.75 * short)
    out = self.dropout(None, out)[1]
    # feat['residual'] = out # @REVIEW do we need this
    return feat, out


class MediumResNet(GenModule):
  def __init__(self, in_channels: int=3, num_classes: int=10, mc_dropout: bool=False, activation: str='crater', gamma: float=1.0, dropout: float=0.1, residual: bool = False, stochdepth_rate: float=0.25, track: bool = True):
    """ Create a batchable Medium resnet classification model

    Args:
      in_channels (int, optional): number of input Conv2d channels. Defaults to 3.
      num_classes (int, optional): number of output classes. Defaults to 10.
      flatten_dims (int, optional): input features to first Linear after flattening Conv2d layers. See lenet.py if confused. Defaults to 256.
      activation (str, optional): name of activation to use. See net/activation.py for options. Defaults to 'crater'.
      gamma (float, optional): input gamma adjustment for expected variance. Usually normalized data has gamma=1.0. Defaults to 1.0.
      track (bool, optional): whether or not to track the internal features of the modules. Defaults to True.
    """
    super().__init__(track)
    
    # zero out stochastic depth for non-residual
    if not residual:
      stochdepth_rate = 0.0
      dropout_add = 0.05  # add dropout
    else:
      dropout_add = 0.0
    
    self.conv1 = WSConv2D(in_channels, 32, kernel_size=5, stride=2, padding=2, act=None, gamma=gamma, bias=True)  # in 32x32 out 16x16
    # self.conv2 = WSConv2D(32, 64, kernel_size=3, stride=1, padding=1, act=activation, gamma=activation, bias=True)
    self.layers = SequentialModule(
      Block(32, 64, stride=2, dropout=dropout_add, activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),
      Block(64, 128, stride=2, dropout=dropout_add, activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),
      Block(128, 256, stride=2, dropout=dropout_add, activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),
      # Block(84, 128, stride=2, dropout=dropout_add, activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),  # in 16x16 out 8x8
      # Block(128, 256, stride=2, dropout=dropout_add, activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),  # in 8x8 out 4x4
      # Block(256, 256, stride=1, dropout=(2*dropout_add), activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),  # in 4x4 out 4x4
      # Block(256, 512, stride=2, dropout=(2*dropout_add), activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),  # in 4x4 out 2x2
      # Block(512, 768, stride=2, dropout=(2*dropout_add), activation=activation, res=residual, stochdepth_rate=stochdepth_rate, track=track),  # in 2x2 out 1x1
      #     [64, 64],
      # [64, 128],
      # [128, 256],
    )
    # self.dropout = nn.Dropout(p=0.4)
    self.lin_layers = SequentialModule(
      Dropout(p=dropout),
      Linear(256, 256, bias=True, act=activation, gamma=None),
      Dropout(p=dropout, mc_dropout=mc_dropout, track=False),
      FinalLinear(256, num_classes, bias=True, act=None, gamma=activation),
    )
    
  def is_generated(self):
    return False
  
  def define_generated_modules(self):
    mod = super().define_generated_modules()
    mod['conv1'] = self.conv1.define_generated_modules()
    # mod['conv2'] = self.conv2.define_generated_modules()
    mod['layers'] = self.layers.define_generated_modules()
    mod['lin_layers'] = self.lin_layers.define_generated_modules()
    return mod
  
  def forward(self, params, x):
    feat = OrderedDict()
    h, w = x.shape[-2:]
    # print(self.conv1.from_flat(params['conv1']['self'])['weight'].shape)
    feat['conv1'] = out = self.conv1(params['conv1'], x)[1] * math.sqrt((h + 1.0) / h)
    # feat['conv2'] = out = self.conv2(params['conv2'], out)[1]
    feat['layers'], out = self.layers(params['layers'], out)
    
    h, w = out.shape[-2:]
    out = torch.mean(out, dim=(3, 4)) * math.sqrt(math.sqrt(h * w))
    feat['lin_layers'], out = self.lin_layers(params['lin_layers'], out)
    return feat, out

    # classifier = Sequential(
    #   ScaledWConv2d(3, 64, 5, act=act, stride=2, padding=1, gamma=1.0, bias=bias),
    #   ScaledWConv2d(64, 64, 3, act=act, stride=1, padding=1, gamma=act, bias=bias),
    #   Dropout(p=0.15),
    #   ScaledWConv2d(64, 128, 5, act=act, stride=2, padding=1, gamma=act, bias=bias),
    #   Dropout(p=0.15),
    #   ScaledWConv2d(128, 128, 3, act=act, stride=1, padding=1, gamma=act, bias=bias),
    #   Dropout(p=0.3),
    #   ScaledWConv2d(128, 256, 3, act=act, stride=2, padding=1, gamma=act, bias=bias),
    #   Dropout(p=0.15),
    #   ScaledWConv2d(256, 256, 3, act=act, stride=2, padding=1, gamma=act, bias=bias),
    #   Dropout(p=0.3),
    #   Flatten(track=False),  # do not track this layer
    #   Linear(1024, 512, act=act, gamma=act, bias=bias),
    #   Dropout(p=0.4),
    #   FinalLinear(512, out_dim, act=None, gamma=act, bias=bias)
    # )
  
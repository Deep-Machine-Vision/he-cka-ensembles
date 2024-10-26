""" Creates the implicit conv filter layer sampler models """
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import numpy as np

from collections import OrderedDict

from ..conv import _ConvNd, Conv2d, ScaledWConv2d
from ..module import GenModule, ParametersGroup
from ...net import models, activation
from .base import BaseLayerGenerator, register_layer_generator


class SampledConvLayerGenerator(BaseLayerGenerator):
  """ Base class for a sampled conv layer generator """
  pass


@register_layer_generator('mlp_sampled_conv_generator')
class MLPSampledConvLayerGenerator(SampledConvLayerGenerator):
  def __init__(self, definition: _ConvNd, input_size: int, mlp_dims: Union[int, List[int]]=3, dim_multiplier: float=2, bias: bool=False, gamma: Union[float, str, object]=1.0, norm_before_last: bool=True, sn_coeff: float=3.0):
    """ Define a basic parameter generator using GELU, Linear, and GroupNorm with some flexibility for size

    Args:
        definition (_ConvNd): the model convolutional layer description to make a generator module for
        input_size (int): the size the generator should expect as an input (from some other hypernetwork or latent space)
        mlp_dims (Union[int, List[int]]): the center dimensions of the layer generator (including first layer excluding the last linear to project to parameter space). If it's an int it's just the number of layers to generate (look at mlp_multiplier), else it's a list of dimensions
        dim_multiplier (float, optional): if mlp_dims is an integer then use this multiplier to scale the number of parameters of each layer up linearly from latent_size*mlp_multiplier (only use if mlp_dims is int). Defaults to 2.
        bias (bool, optional): to use the bias term in the generators. Defaults to False.
        gamma (check nfnet/activation.py): calculates input gamma/scaling of weights based on previous output. Default 1.0 meaning expecting input to have variance of 1.0
    """
    super(MLPSampledConvLayerGenerator, self).__init__(definition, input_size)
    if not isinstance(definition, _ConvNd):
      raise ValueError('Attempted to make a sampled conv filter layer on a non-convolutional definition')

    if definition.transposed:
      raise NotImplementedError('handle Conv2d transpose please... @TODO')

    # build a basic MLP
    self.gen_bias = definition.bias
    self.filter_param = self.get_flat_size()
    self.generator = models.build_groupnorm_mlp(
      in_features=input_size,
      out_features=self.filter_param,
      mlp_dims=mlp_dims,
      dim_multiplier=dim_multiplier,
      gamma=gamma,
      bias=bias,
      norm_last=False,
      norm_before_last=norm_before_last,
      activation=torch.nn.GELU,
      norm_last_groups=1,  # output layer norm
      out_act=None,  # nothing on weight space (if changed look at init below for changes)
      coeff=sn_coeff  #3.5
    )
  
  def get_flat_size(self):
    """ Returns the dimension (number of parameters) of a single convolutional filter """
    # fix number of expected input (divide by groups is it is specified)
    in_chan = self.definition.in_channels // self.definition.groups

    # now do product of input channels and the kernel size
    in_dims = in_chan * np.prod(self.definition.kernel_size)
    return in_dims + (1 if self.gen_bias else 0)  # adds bias output to each filter generator 

  def get_expected_input_size(self):
    """ Tell mixer/hyper generator we expect this many codes
    
    Since we need out_channels worth of codes let's return that
    """
    return self.definition.out_channels

  def forward(self, codes):
    """ Specify foward from the latents
    
    Args:
      codes: generated codes from parent network/process in format [model batch size, get_expected_input_size(), code_size] 
    """
    # we're going to get [B, out_channels, code_size]
    # out filter generator will take in B*out_channels as batch
    B = codes.shape[0]
    merged = codes.reshape(B*self.get_expected_input_size(), -1)
    
    # pass through sampled layer generator
    all_filters = self.generator(merged)

    # reshape filter weights to be [B, out_channels * (flat_size + 1 if bias else 0)]
    # since we expect convolution to be [out_channels, in_channels, *kernel_size]
    # NOTE: this won't work on Conv2d transpose since the vectorized form will not align with the "unview"
    num_filter = self.filter_param    
    all_filters = all_filters.view(B, self.get_expected_input_size(), num_filter)
    
    # extract bias terms and aggregate them to the end
    # which is going to be AFTER the weight term
    # if you're running into shape issues ensure bias is defined after weight in your module
    # then it'll take that as the last "out_channels" see module.py for ParametersGroup definition that unflattens this view
    if self.gen_bias:
      all_bias = all_filters[:, :, -1].view(B, self.get_expected_input_size())  # bias is aggregated to be sent to the end
      all_weights = all_filters[:, :, :-1].reshape(B, self.get_expected_input_size()*(num_filter - 1))  # flatten kernels
      all_filters = torch.concat([all_weights, all_bias], dim=1).contiguous()  # combine along model weights
    
    return all_filters.reshape(B, -1)  # flattent dims

""" Module that defines the batched group linear operator and its respective generator """
from typing import Union
from .module import ParametersGroup, register_gen_module
from ..net import activation

import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch
import math


@register_gen_module('linear')
class Linear(ParametersGroup):
  def __init__(self, in_features: int, out_features: int, bias: bool=True, act: str=None, gamma: Union[str, float, object]=1.0, param_scale: str='torch', track: Union[bool, str]=True):
    """ Creates a batched linear layer operator

    Args:
      in_features (int): expected number of input features per model
      out_features (int): expected number of output features per model
      bias (bool, optional): include bias term (out_features worth). Defaults to True.
      act (str, optional): the activation to use after the linear layer. Default is None.
      param_scale (str): weight initialization/scaling method. Default is torch module equivalent.
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
    """
    # specify expected parameter shapes
    self.in_features = in_features
    self.out_features = out_features
    self.weight_shape = (out_features, in_features)
    self.bias_shape = (out_features,)
    self.param_scale = param_scale
    self.bias = bias
    
    # handle parameter scaling and signal gain by preceding activation function
    if self.param_scale == 'gamma':
      self.invsq = math.sqrt(1 / in_features)
      self.gamma = activation.activation_gamma(gamma)
      self.scale = self.gamma * self.invsq
      self.bias_scale = self.invsq
    elif self.param_scale == 'torch':  # attempt to match init by torch init module
      gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5.0))
      std = gain / math.sqrt(in_features)
      self.scale = math.sqrt(std)
      self.bias_scale = math.sqrt(1.0 / math.sqrt(in_features))
    else:
      raise ValueError(f'Invalid paramater scaling method {self.param_scale}')
    
    
    if (not act is None) and isinstance(act, str):
      if hasattr(F, act):
        self.act = getattr(F, act)
      elif hasattr(activation, act):
        self.act = getattr(activation, act)
      else:
        raise ValueError(f'Could not find activation {act} in either torch.nn.functional or nfnet.activation')
    else:
      self.act = act

    if (not act is None) and isinstance(self.act, str):
      raise ValueError(f'The activation {act} is not in torch.nn.functional')

    # construct parameters indexing via ParametersGroup constructor
    super(Linear, self).__init__(shapes=[
      ('weight', self.weight_shape),
      ('bias', self.bias_shape)
    ] if bias else [
      ('weight', self.weight_shape)
    ], track=track)
  
  def gen_initialized_params(self, dtype=None, device=None, gain=1.0):
    """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
    if gain is None or not isinstance(gain, float):
      raise ValueError('Gain must be a float')

    # these are autoscaled already by _param_scale
    params = self.gen_empty_params(dtype=dtype, device=device)
    with torch.no_grad():
      if self.param_scale == 'gamma':
        init.normal_(params['weight'], 0.0, 1.0)
        # init.normal_(params['weight'], 0.0, self.invsq**2.0)
        if self.bias:
          # init.zeros_(params['bias'])
          # init.normal_(params['bias'], 0.0, self.scale**2.0)
          init.normal_(params['bias'], 0.0, 1.0)
      elif self.param_scale == 'torch':
        init.uniform_(params['weight'], -math.sqrt(3.0), math.sqrt(3.0))
        if self.bias:
          init.uniform_(params['bias'], -math.sqrt(3.0), math.sqrt(3.0))
        # init.kaiming_uniform_(params['weight'], a=math.sqrt(5))
        # if self.bias:
        #   fan_in, _ = init._calculate_fan_in_and_fan_out(params['weight'])
        #   bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #   init.uniform_(params['bias'], -bound, bound)
        
    return params

  def _param_scale(self, weight, bias):
    """ Applies unit gaussian normalization across the batched parameters and gain on weight parameters and simple scaling on bias

    Args:
        weight (torch.Tensor): tensor of the weight to correctly shift/scale and gain
        bias (torch.Tensor): tensor of the bias to correctly scale
    """
    weight = self.scale * weight
    if bias is not None:
      bias = self.bias_scale * bias
    return weight, bias

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Handles the batched linear forward operation
    
    Args:
      viewed (torch.Tensor): correct view weights/biases in single vector
      x (torch.Tensor): set of features to forward. Accepts (*, input_features)
    """

    # apply scaling to parameters
    if self.bias:
      viewed['weight'], viewed['bias'] = self._param_scale(viewed['weight'], viewed['bias'])
    else:
      viewed['weight'], _ = self._param_scale(viewed['weight'], None)

    # fix inputs to be batched
    # got modified implementation from ALF
    weight_batch = viewed['weight'].shape[0]
    if x.ndim == 2:
      assert x.shape[1] == self.in_features, (
        f'Input inputs has wrong shape {x.shape}. Expecting (feature batch, {self.in_features})'
      )

      # change to [N, B, D]
      x = x.unsqueeze(0).expand(weight_batch, *x.shape)
    elif x.ndim == 3:  # we should expect [N, B, D]
      assert (
        x.shape[0] == weight_batch
        and x.shape[2] == self.in_features), (
          f'Input inputs has wrong shape {x.shape}. Expecting ({weight_batch}, B, {self.in_features})'
      )
    else:
      raise ValueError(f'Badx dimension. Got {x.ndim}')

    # fix bias to work across batch dim and not model dim
    # @TODO figure out tiny numerical difference (1e-7) between baddm and linear
    if self.bias:
      bias = viewed['bias'].unsqueeze(1) # broadcastable across feature batch
      
      # now expected shape is [B, examples, out_features]
      y = torch.baddbmm(bias, x, viewed['weight'].transpose(1, 2))
      # y = torch.stack(
      #   [
      #     F.linear(x[i], weight=viewed['weight'][i], bias=viewed['bias'][i]) for i in range(viewed['weight'].shape[0])
      #   ]
      # )
    else:
      y = torch.bmm(x, viewed['weight'].transpose(1, 2))
      # y = torch.stack(
      #   [
      #     F.linear(x[i] + 1, weight=viewed['weight'][i]) for i in range(viewed['weight'].shape[0])
      #   ]
      # )

    if self.act is None:
      return y
    y = self.act(y)

    # add following to unit tests @TODO
    # print('in weight', viewed['weight'])
    # print('in shape', x.shape, 'out shape', y.shape, 'x0', x[0].shape, 'y0', y[0].shape, 've', viewed['weight'][0].shape, 'lin', F.linear(x[0], viewed['weight'][0]).shape)
    # print('diff x', torch.sum(torch.abs(x[0] - x[1])))
    # print(y[0], 'and', self.act(F.linear(x[0], viewed['weight'][0], viewed['bias'][0])))
    # print('diff',  torch.sum(torch.abs(y[0] - self.act(F.linear(x[0], viewed['weight'][0], viewed['bias'][0])))))
    # print()
    return y

  ''' OLD IMPL
  def _model_batch_second_forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Handles the batched linear forward operation
    
    Args:
      viewed (torch.Tensor): correct view weights/biases in single vector
      x (torch.Tensor): set of features to forward. Accepts (*, input_features)
    """

    # apply scaling to parameters
    if self.bias:
      viewed['weight'], viewed['bias'] = self._param_scale(viewed['weight'], viewed['bias'])
    else:
      viewed['weight'], _ = self._param_scale(viewed['weight'], None)

    # fix inputs to be batched
    # got modified implementation from ALF
    weight_batch = viewed['weight'].shape[0]
    if x.ndim == 2:
      assert x.shape[1] == self.in_features, (
        f'Input inputs has wrong shape {x.shape}. Expecting (B, {self.in_features})'
      )

      # change to [B, N, D]
      x = x.unsqueeze(0).expand(weight_batch, *x.shape)
    elif x.ndim == 3:  # we should expect [N, B, D]
      assert (
        x.shape[1] == weight_batch
        and x.shape[2] == self.in_features), (
          f'Input inputs has wrong shape {x.shape}. Expecting (B, {weight_batch}, {self.in_features})'
      )
      x = x.transpose(0, 1)  # convert to [B, N, D]
    else:
      raise ValueError(f'Badx dimension. Got {x.ndim}')

    # fix bias to work across batch dim and not model dim
    if self.bias:
      bias = viewed['bias'].unsqueeze(1) # broadcastable across feature batch

      # now expected shape is [B, examples, out_features]
      y = torch.baddbmm(bias, x, viewed['weight'].transpose(1, 2))
    else:
      y = torch.bmm(x, viewed['weight'].transpose(1, 2))

    # reshape back to [N, B, D]
    y = y.transpose(0, 1)

    if self.act is None:
      return y
    y = self.act(y)

    # add following to unit tests @TODO
    # print('in weight', viewed['weight'])
    # print('in shape', x.shape, 'out shape', y.shape, 'x0', x[0].shape, 'y0', y[0].shape, 've', viewed['weight'][0].shape, 'lin', F.linear(x[0], viewed['weight'][0]).shape)
    # print('diff x', torch.sum(torch.abs(x[0] - x[1])))
    # print(y[0], 'and', self.act(F.linear(x[0], viewed['weight'][0], viewed['bias'][0])))
    # print('diff',  torch.sum(torch.abs(y[0] - self.act(F.linear(x[0], viewed['weight'][0], viewed['bias'][0])))))
    # print()
    return y
  '''
  
  def extra_repr(self) -> str:
    return f'in_features={self.in_features}, out_features={self.out_features}, activation={str(self.act)}'


@register_gen_module('final_linear')
class FinalLinear(Linear):
  """ Used for default generator to specify a special generator for the output layer. Generally you need this for nfnets """
  pass


@register_gen_module('scaled_w_linear')
class ScaledWLinear(Linear):
  def __init__(self, in_features: int, out_features: int, bias: bool=True, act: str=None, gamma: [str, float, object]=1.0, affine: bool=True, eps: float=1e-5, track: Union[bool, str]=True):
    """ Creates a scaled standard deviation weight batched linear layer operator

    Args:
      in_features (int): expected number of input features per model
      out_features (int): expected number of output features per model
      bias (bool, optional): include bias term (out_features worth). Defaults to True.
      act (str, optional): the activation to use after the linear layer. Default is None.
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
      
    """
    # construct parameters indexing via ParametersGroup constructor
    super(ScaledWLinear, self).__init__(
      in_features=in_features,
      out_features=out_features,
      bias=bias,
      act=act,
      track=track
    )

    # learnable gain parameter same gain is applied across the batch
    self.affine = affine
    self.affine_weight = nn.Parameter(torch.ones(1, out_features, 1)) if affine else None
    # self.affine_bias = nn.Parameter(torch.zeros(1, out_features, 1)) if affine else None

    # calculate gain (from expected previous activation/gain) and scale by input features
    self.eps = eps
  
  # def gen_initialized_params(self, dtype=None, device=None, gain=1.0):
  #   """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
  #   if gain is None or not isinstance(gain, float):
  #     raise ValueError('Gain must be a float')

  #   # by default we just initialize with unit gaussian
  #   # as we scale them later
  #   params = self.gen_empty_params(dtype=dtype, device=device)
  #   init.normal_(params['weight'], 0.0, 1.0)
  #   if self.bias:
  #     init.normal_(params['bias'], 0.0, 1.0)
  #   return params

  def _param_scale(self, weight, bias):
    """ Applies unit gaussian normalization across the batched parameters and gain on weight parameters and simple scaling on bias

    Args:
        weight (torch.Tensor): tensor of the weight to correctly shift/scale and gain
        bias (torch.Tensor): tensor of the bias to correctly scale
    """
    # mean and std along the input dimension
    mean = torch.mean(weight, dim=2, keepdim=True)
    std = torch.std(weight, dim=2, keepdim=True, unbiased=False)
    weight = self.scale * ((weight - mean) / (std + self.eps))
    if self.affine:
      weight = (weight * self.affine_weight) # + self.affine_bias
    bias = self.invsq * bias  # scale bias var by 1/n
    return weight, bias


@register_gen_module('final_scaled_w_linear')
class FinalScaledWLinear(ScaledWLinear):
  """ Used for default generator to specify a special generator for the output layer. Generally you need this for nfnets """
  pass

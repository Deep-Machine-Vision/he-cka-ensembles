""" Module that defines the batched group linear operator and its respective generator """
from typing import Union, Tuple, Optional, Callable
from .module import ParametersGroup, register_gen_module
from ..net import activation

from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch
import math


class _ConvNd(ParametersGroup):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: Tuple[int, ...],
      stride: Tuple[int, ...],
      padding: Tuple[int, ...],
      dilation: Tuple[int, ...],
      transposed: bool,
      output_padding: Tuple[int, ...],
      groups: int,
      bias: bool,
      padding_mode: str,
      act: str=None,
      gamma: Union[str, float, object]=1.0,
      param_scale: str='torch',
      track: Union[bool, str]=True):
    """ Creates a batched convolution layer operator

    Args:
      in_channels (int): expected number of input features per model
      out_channels (int): expected number of output features per model
      kernel_size (int): size of the kernel
      dilation (int): dilation of kernel
      bias (bool, optional): include bias term (out_channels worth). Defaults to True.
      act (str, optional): the activation to use after the linear layer. Default is None.
      param_scale (str): weight initialization/scaling method. Default is torch module equivalent.
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
    """

    # most of this is referenced from https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
    if groups <= 0:
      raise ValueError('groups must be a positive integer')
    if in_channels % groups != 0:
      raise ValueError('in_channels must be divisible by groups')
    if out_channels % groups != 0:
      raise ValueError('out_channels must be divisible by groups')
    valid_padding_strings = {'same', 'valid'}
    if isinstance(padding, str):
      if padding not in valid_padding_strings:
        raise ValueError(
            f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
      if padding == 'same' and any(s != 1 for s in stride):
        raise ValueError("padding='same' is not supported for strided convolutions")

    valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
    if padding_mode not in valid_padding_modes:
      raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")

    # specify expected parameter shapes
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.transposed = transposed
    self.output_padding = output_padding
    self.groups = groups
    self.padding_mode = padding_mode
    self.param_scale = param_scale

    if isinstance(self.padding, str):
      self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
      if padding == 'same':
        for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
          total_padding = d * (k - 1)
          left_pad = total_padding // 2
          self._reversed_padding_repeated_twice[2 * i] = left_pad
          self._reversed_padding_repeated_twice[2 * i + 1] = (
              total_padding - left_pad)
    else:
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

    # define expected shapes
    if transposed:
      self.weight_shape = (in_channels, out_channels // groups, *kernel_size)
    else:
      self.weight_shape = (out_channels, in_channels // groups, *kernel_size)
    self.bias_shape = (out_channels,)
    self.bias = bias

    # number of input features and scaling features for conv
    self.fan_in = (in_channels // groups) * np.prod(kernel_size)
    if self.param_scale == 'gamma':
      self.invsq = math.sqrt(1 / self.fan_in)
      self.gamma = activation.activation_gamma(gamma)
      self.scale = self.gamma * self.invsq
      self.bias_scale = self.invsq
    elif self.param_scale == 'torch':  # attempt to match init by torch init module
      gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5.0))
      std = gain / math.sqrt(self.fan_in)
      self.scale = math.sqrt(std)
      self.bias_scale = math.sqrt(1.0 / math.sqrt(self.fan_in))
    else:
      raise ValueError(f'Invalid paramater scaling method {self.param_scale}')
    
    # get the activation function if defined
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
    super(_ConvNd, self).__init__(shapes=[
      ('weight', self.weight_shape),
      ('bias', self.bias_shape)
    ] if bias else [
      ('weight', self.weight_shape)
    ], track=track)
  
  def gen_initialized_params(self, dtype=None, device=None, gain=1.0):
    """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
    if gain is None or not isinstance(gain, float):
      raise ValueError('Gain must be a float')

    # by default we just initialize with unit gaussian
    # as we scale them later
    params = self.gen_empty_params(dtype=dtype, device=device)
    
    if self.param_scale == 'gamma':
      init.normal_(params['weight'], 0.0, 1.0)
      if self.bias:
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
      # nonlinearity = 'relu'
      # a = math.sqrt(5.0)
      # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(params['weight'])
      # gain = torch.nn.init.calculate_gain(nonlinearity, a)
      # std = gain / math.sqrt(fan_in)
      
      # # params['weight'] = params['weight'] * std
      
      # # if self.bias:
      # #   params['bias'] = params['bias'] * (1.0 / math.sqrt(fan_in))
    
    return params

  def _param_scale(self, viewed):
    """ Applies unit gaussian normalization across the batched parameters and gain on weight parameters and simple scaling on bias

    Args:
        viewed (OrderedDict): tensors of the weight/bias to correctly shift/scale and gain
    """
    viewed['weight'] = self.scale * viewed['weight']
    if self.bias:
      viewed['bias'] = self.bias_scale * viewed['bias']
      
    return viewed

  def _conv_forward(self, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ Runs convolution function on correctly variance adjusted weight and bias

    Args:
        weight (torch.Tensor): assume correctly adjusted for weight
        bias (torch.Tensor): assume correctly adjust for bias

    Returns:
        torch.Tensor: output of convolution
    """
    raise NotImplementedError('_ConvNd does not have a convolution type defined. Please use sub-class')

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Handles the batched linear forward operation
    
    Args:
      viewed (torch.Tensor): correct view weights/biases in single vector
      x (torch.Tensor): set of features to forward. Accepts (*, input_features)
    """

    # apply scaling to parameters
    viewed = self._param_scale(viewed)
    
    # run through convolution
    y = self._conv_forward(x, viewed['weight'], viewed['bias'] if self.bias else None)
    return y

CONV2D_DEFAULT_POOLING_ARGS = {
  'kernel_size': 2
} 


@register_gen_module('conv2d')
class Conv2d(_ConvNd):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_2_t,
      stride: _size_2_t=1,
      padding: Union[str, _size_2_t]=0,
      dilation: _size_2_t=1,
      groups: int=1,
      bias: bool=True,
      padding_mode: str='zeros',
      act: str=None,
      pooling: Union[str, Callable]=None,
      pooling_kwargs: dict=CONV2D_DEFAULT_POOLING_ARGS,
      gamma: Union[str, float, object]=1.0,
      param_scale: str='torch',
      track: Union[bool, str]=True):
    """ Creates a batched 2D convolution layer operator

    Args:
      in_channels (int): expected number of input features per model
      out_channels (int): expected number of output features per model
      kernel_size (int): size of the kernel
      dilation (int): dilation of kernel
      bias (bool, optional): include bias term (out_channels worth). Defaults to True.
      act (str, optional): the activation to use after the linear layer. Default is None.
      pooling (str|Callable): method to pool. Options can be 'max', 'avg' or a function like F.max_pool2d. Default is None (ie no pooling).
      pooling_kwargs (dict): by default the kernel size is 2. You can pass any options to the pool function here.
      gamma (float|str|method): variance scaling correction coefficient. Autocalculated for common activations, default is no scaling ie 1.0. 
      param_scale (str): weight initialization/scaling method. Default is torch module equivalent.
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
    
      Note: applying pooling in this module will spare you a reshape (with copy) if calling pooling in a separate module which will require tensor copies.
    """
    kernel_size_ = _pair(kernel_size)
    stride_ = _pair(stride)
    padding_ = padding if isinstance(padding, str) else _pair(padding)
    dilation_ = _pair(dilation)

    self.pooling = pooling
    if isinstance(self.pooling, str):
      try:
        pooling_map = {
          'max': F.max_pool2d,
          'avg': F.avg_pool2d,
          'adaptive_max': F.adaptive_avg_pool2d,
          'adaptive_avg': F.adaptive_avg_pool2d
        }
        self.pooling = pooling_map[self.pooling]
      except KeyError:
        raise KeyError(f'Provided pooling "{self.pooling}" was not found. Available keys: {", ".join(pooling_map.keys())}')

    self.pooling_kwargs = pooling_kwargs

    super(Conv2d, self).__init__(
      in_channels,
      out_channels,
      kernel_size_,
      stride_,
      padding_,
      dilation_,
      False,  # transposed
      _pair(0),  # output_padding
      groups,
      bias,
      padding_mode,
      act,
      gamma,
      param_scale,
      track
    )
    
    # create batched conv2d
    # self.batched_conv = torch.func.vmap(self._apply_conv)

  def _apply_conv(self, input, weights, biases, index, repeat):
    """ Internal application of a single conv2d """
    # repeating input or copies already provided?
    if repeat:
      x = input
    else:
      x = input[index]

    if biases is None:
      bias = None
    else:
      bias = biases[index]

    # apply padding if necessary/run conv
    if self.padding_mode != 'zeros':
      x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
      o = F.conv2d(
        input=x.contiguous(),
        weight=weights[index],
        bias=bias,
        stride=self.stride,
        padding=_pair(0),
        dilation=self.dilation,
        groups=self.groups
      )
    else:
      # uncomment to test weight scaling standalone
      # # useful for future ref
      # weight = weights[index]
      # mean = torch.mean(weight, axis=[1,2,3], keepdims=True)
      # var = torch.var(weight, axis=[1,2,3], keepdims=True)
      # scale = torch.rsqrt(torch.maximum(var * self.fan_in, torch.tensor(self.eps, device=weight.device)))
      # weight = (weight - mean) * scale
      
      # run through 2d convolution
      o = F.conv2d(
        input=x,
        weight=weights[index],
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups
      )
    
    # apply activation if applicable
    if self.act is not None:
      o = self.act(o)
    
    # apply pooling method if applicable
    if self.pooling is not None:
      # apply pooling
      o = self.pooling(o, **self.pooling_kwargs)
      
    return o

  def _conv_forward_groups(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ Runs convolution function on correctly variance adjusted weight and bias

    NOTE: empirically I found using Conv2d with group for batched conv is slower than just looping over convs/individual calls
    most likely due to having to permute input and contiguous copies when individually that is not required.

    Args:
      input (torch.Tensor): input to feed through conv expected for 2d [B, N, C_in, H_in, W_in] or [N, C_in, H_in, W_in]
      weight (torch.Tensor): assume correctly adjusted for weight
      bias (torch.Tensor): assume correctly adjust for bias

    Returns:
        torch.Tensor: output of convolution [B, N, C_out, H_out, W_out]
    """
    # implementation inspired from ALF framework
    # if we don't have batched input. ie [N, C_in, H_in, W_in] (for example passing input image) we need to repeat B times for each model
    # to get [N, B*C_in, H_in, W_in]
    model_bs = weight.shape[0]  # weights are [B, out_channels, in_channels // groups, K_w, K_h]
    
    # input will be either just a instance with (C_in channels) or already a batched channel input (B * C_in)
    if input.ndim == 4:
      if input.shape[1] == self.in_channels:  # repeat across input channels
        input = input.repeat(1, model_bs, 1, 1) # now [N, B*C_in, H_in, W_in]
      else:
        assert input.shape[1] == (model_bs * self.in_channels), (
          f'Conv2d input x has wrong shape {input.shape}. Expecting (N, H, W) or (N, {model_bs * self.in_channels}, H, W)'
        )
      # assert input.shape[1] == self.in_channels, 'Input channel count mismatch'
    elif input.ndim == 5:
      assert input.shape[0] == model_bs and input.shape[2] == self.in_channels, (
        f'Conv2d input x has wrong shape {input.shape}, expect [{model_bs}, N, {self.in_channels}, H, W]'
      )
      
      # now change input from [B, N, C_in, H_in, W_in] to [N, B*C_in, H_in, W_in]
      mbs, n, c = input.shape[:3]
      other = input.shape[3:]
      input = input.transpose(0, 1).reshape(n, mbs * c, *other)
    else:
      raise RuntimeError(f'Invalid input shape to batched Conv2d {input.shape} expecting ndim to be 4 or 5')

    device = weight.device
    # reshape weights/bias for conv2d operation from [B, C_out, C_in, K_h, K_w] to [B*C_out, C_in, K_h, K_w]
    weight = weight.reshape(model_bs * self.out_channels, self.in_channels // self.groups, *self.kernel_size)
    if self.bias:
      bias = bias.reshape(model_bs * self.out_channels)

    # apply padding to input features
    if self.padding_mode != 'zeros':
      input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
      y = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=self.stride,
        padding=_pair(0),
        dilation=self.dilation,
        groups=self.groups*model_bs
      )
    else:
      # run through 2d convolution
      y = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups*model_bs
      )

    # apply activation if applicable
    if self.act is not None:
      y = self.act(y)

    # apply max pooling if applicable
    if self.pooling is not None:
      # apply pooling
      y = self.pooling(y, **self.pooling_kwargs)

    # current y is [N, B*C_out, H_out, W_out]
    # reshape back into expected [B, N, C_out, H_out, W_out]
    y = y.view(y.shape[0], model_bs, self.out_channels, y.shape[2], y.shape[3]).transpose(0, 1).contiguous()

    # should be [B, N, C_out, W_out]
    return y

  def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """ Runs convolution function on correctly variance adjusted weight and bias

    Args:
      input (torch.Tensor): input to feed through conv expected for 2d [B, N, C_in, H_in, W_in] or [N, C_in, H_in, W_in]
      weight (torch.Tensor): assume correctly adjusted for weight
      bias (torch.Tensor): assume correctly adjust for bias

    Returns:
        torch.Tensor: output of convolution [B, N, C_out, H_out, W_out]
    """
    model_bs = weight.shape[0]  # weights are [B, out_channels, in_channels // groups, K_w, K_h]
    
    # input in this case is either just a single image with (N, C_in channels) or already a batched channel input (B, N, C_in)
    repeat = False
    if input.ndim == 4:
      repeat = True
      assert input.shape[1] == self.in_channels, 'Input channel mismatch!'
      
      # repeat input across models
      # input = input.unsqueeze(0).repeat(model_bs, 1, 1, 1, 1) # now [B, N, C_in, H_in, W_in]
    elif input.ndim == 5:
      assert input.shape[0] == model_bs, (
        f'Conv2d input x has wrong shape {input.shape}'
      )

    # reshape weights/bias for conv2d operation to [B, C_out, C_in, K_h, K_w]
    weight = weight.reshape(model_bs, self.out_channels, self.in_channels // self.groups, *self.kernel_size)
    if self.bias:
      bias = bias.reshape(model_bs, self.out_channels)
    else:
      bias = None

    device = weight.device
    out = []  # outputs from convs

    # apply cpu or cuda
    # BUG: this pattern is broken past Pytorch 1.9 :(
    # https://pytorch.org/docs/stable/notes/cuda.html#bwd-cuda-stream-semantics
    # if input.is_cuda and False:
    #   streams = [torch.cuda.Stream(device) for _ in range(model_bs)]
    #   for ind, stream in enumerate(streams):
    #     with torch.cuda.stream(stream):
    #       out.append(self._apply_conv(
    #         weights=weight,
    #         biases=bias,
    #         input=input,
    #         index=ind,
    #         repeat=repeat
    #       ))
      
    #   # sync streams
    #   cur_stream = torch.cuda.current_stream()
    #   for stream in streams:
    #     cur_stream.wait_stream(stream)
    #   torch.cuda.synchronize()
    # else:
    # apply standard linear order otherwise
    for ind in range(model_bs):
      out.append(self._apply_conv(
        weights=weight,
        biases=bias,
        input=input,
        index=ind,
        repeat=repeat
      ))

    # stack results
    y = torch.stack(out)
    
    # if self.bias:
    #   y = self.batched_conv(input, weight, bias) # torch.stack(out)
    # else:
    #   y = self.batched_conv(input, weight)

    # @TODO still look into parallel conv2d
    # both methods currently aren't ideal
    # above is so bad. Took me forever to find the bug!
    ''' OLD IMPLEMENTATION DUE TO groups=n ACTUALLY BEING SLOWER!

    Still an issue with pytorch v2.1.0 
    # reshape weights/bias for conv2d operation from [B, C_out, C_in, K_h, K_w] to [B*C_out, C_in, K_h, K_w]
    weight = weight.reshape(model_bs * self.out_channels, self.in_channels // self.groups, *self.kernel_size)
    if self.bias:
      bias = bias.reshape(model_bs * self.out_channels)

    print('REPEAT\n', input, input.shape)

    # mean = torch.mean(weight, axis=[1,2,3], keepdims=True)
    # var = torch.var(weight, axis=[1,2,3], keepdims=True)

    # fan_in = weight.shape[1:].numel()
    # weight = (weight - mean) * torch.rsqrt(torch.maximum(var * fan_in, self.eps))

    # mean = torch.mean(weight, axis=[1,2,3], keepdims=True)
    # var = torch.var(weight, axis=[1,2,3], keepdims=True)

    # print(mean.mean())
    # print(var.mean())
    print('preweight', weight.shape)
    # @TODO FIGURE THIS PART OUT :()
    # weight =  F.batch_norm(
    #         weight.reshape(1, self.out_channels, -1), None, None,
    #         weight=(torch.ones((self.out_channels, 1, 1, 1)).to(weight.device) * math.sqrt(1.0 / self.fan_in)).view(-1),
    #         training=True, momentum=0., eps=1e-4).reshape_as(weight)
    # input = torch.normal(0.0, 1.0, size=input.shape)
    # weight = torch.normal(0.0, math.sqrt(1.0 / self.fan_in), size=weight.shape).cuda()
    
    # if self.bias:
    #   bias = torch.zeros_like(bias)
    print('postweight', weight.shape, self.fan_in)

    # apply padding to input features
    if self.padding_mode != 'zeros':
      input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
      y = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=self.stride,
        padding=_pair(0),
        dilation=self.dilation,
        groups=self.groups*model_bs
      )
    else:
      # run through 2d convolution
      y = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        groups=self.groups*model_bs
      )

    print(self.stride, self.padding, self.dilation, self.groups)
    print('VAR', torch.var(y, dim=1).mean())
    print('MEAN', torch.mean(y, dim=1).mean())

    # apply activation if applicable
    if self.act is not None:
      y = self.act(y)

    # apply max pooling if applicable
    if self.pooling is not None:
      # apply pooling
      y = self.pooling(y, **self.pooling_kwargs)

    # current y is [N, B*C_out, H_out, W_out]
    # reshape back into expected [N, B, C_out, H_out, W_out]
    y = y.reshape(y.shape[0], model_bs, self.out_channels, y.shape[2], y.shape[3])
    '''

    # should be [B, N, C_out, W_out]
    return y

  def extra_repr(self) -> str:
    return f'in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, groups={self.groups}'


# helper class for final layer
@register_gen_module('final_conv2d')
class FinalConv2d(Conv2d):
  pass


@register_gen_module('scaled_w_conv2d')
class ScaledWConv2d(Conv2d):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: _size_2_t,
      stride: _size_2_t=1,
      padding: Union[str, _size_2_t]=0,
      dilation: _size_2_t=1,
      groups: int=1,
      bias: bool=True,
      padding_mode: str='zeros',
      act: str=None,
      pooling: Union[str, Callable]=None,
      pooling_kwargs: dict=CONV2D_DEFAULT_POOLING_ARGS,
      gamma: Union[str, float, object]=1.0,
      affine: bool=True,
      eps: float=1e-5, 
      track: Union[bool, str]=True):
    """ Creates a batched 2D convolution layer operator

    Args:
      in_channels (int): expected number of input features per model
      out_channels (int): expected number of output features per model
      kernel_size (int): size of the kernel
      dilation (int): dilation of kernel
      bias (bool, optional): include bias term (out_channels worth). Defaults to True.
      act (str, optional): the activation to use after the linear layer. Default is None.
      pooling (str|Callable): method to pool. Options can be 'max', 'avg' or a function like F.max_pool2d. Default is None (ie no pooling).
      pooling_kwargs (dict): by default the kernel size is 2. You can pass any options to the pool function here.
      gamma (float|str|method): variance scaling correction coefficient. Autocalculated for common activations, default is no scaling ie 1.0. 
      affine (bool): apply linear transform after weight normalization
      eps (float): normalization to prevent zero std.
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
    
      Note: applying pooling in this module will spare you a reshape (with copy) if calling pooling in a separate module which will require tensor copies.
    """
    super(ScaledWConv2d, self).__init__(
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      bias,
      padding_mode,
      act,
      pooling,
      pooling_kwargs,
      gamma,
      track
    )

    # learnable gain parameter same gain is applied across the batch
    # print('GOT AFFINE', affine)
    # self.TEST = nn.Linear(10, 10)
    self.affine = affine
    self._affine_weight = torch.ones(out_channels, requires_grad=affine)
    if affine:
      self.affine_weight = nn.Parameter(self._affine_weight, requires_grad=True)
    else:
      self.register_buffer('affine_weight', self._affine_weight)

    # self.affine_bias = nn.Parameter(torch.zeros(1, out_channels, 1)) if affine else None

    # calculate gain (from expected previous activation/gain) and scale by input features
    self.register_buffer('eps', torch.tensor(eps, dtype=torch.float, requires_grad=False))

    # create weight normalization
    # weights are in [model bs, out, in, h, w] we want to normalize over (in, h, w)
    # self.weight_norm = nn.LayerNorm(
    #   [self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]],
    #   elementwise_affine=affine,
    #   bias=affine,
    #   eps=eps**2.0
    # )

  def _param_scale(self, viewed):
    """ Applies unit gaussian normalization across the batched parameters and gain on weight parameters and simple scaling on bias

    Args:
        weight (torch.Tensor): tensor of the weight to correctly shift/scale and gain
        bias (torch.Tensor): tensor of the bias to correctly scale
    """
    # mean and std along the input dimension
    weight = viewed['weight']


    # normalize weights
    model_bs, out_chan = weight.shape[:2]
    prev_shape = weight.shape
    weight = F.group_norm(
      input=weight.reshape(model_bs, out_chan, -1),
      num_groups=out_chan,
      weight=self.affine_weight * self.scale,
      bias=None,
      eps=self.eps
    ).reshape(prev_shape)
    #
    # batch norm method
    #
    # for ind in range(weight.shape[0]):
    #   weight[ind] =  F.batch_norm(
    #     weight[ind].reshape(1, self.out_channels, -1), None, None,
    #     weight=(torch.ones((self.out_channels, 1, 1, 1)).to(weight.device) * math.sqrt(1.0 / self.fan_in)).view(-1),
    #     training=True, momentum=0., eps=1e-4
    #   ).reshape_as(weight[ind])
    
    # NAIVE (I've verified this works) method
    # model_bs, out_chan = weight.shape[:2]
    # weights = []
    # for i in range(model_bs):
    #   mean = torch.mean(weight[i], axis=[1,2,3], keepdims=True)
    #   var = torch.var(weight[i], axis=[1,2,3], keepdims=True)
    #   scale = torch.rsqrt(torch.maximum(var * self.fan_in, torch.tensor(self.eps, device=weight.device)))
    #   weight_i = (weight[i] - mean) * scale
    #   # weight[i] = weight
    #   weights.append(weight_i)
    # weight = self.gamma * torch.stack(weights)

    # normalize weights
    # var, mean = torch.var_mean(weight, dim=[2, 3, 4], correction=1, keepdim=True)
    # scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
    # weight = (weight - mean) * scale

    viewed['weight'] = weight  # self.weight_norm(weight) * self.scale
    if self.bias:
      viewed['bias'] = self.invsq * viewed['bias']  # scale bias var by 1/n
    return viewed


@register_gen_module('final_scaled_w_conv2d')
class FinalScaledWConv2(ScaledWConv2d):
  """ Used for default generator to specify a special generator for the output layer. Generally you need this for nfnets """
  pass

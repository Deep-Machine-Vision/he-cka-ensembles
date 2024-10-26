""" Module that defines the batched group linear operator and its respective generator """
from typing import Union, Tuple, Optional, Callable
from .module import ParametersGroup, GenModule
from ..net import activation

from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torch
import math


class _AvgPoolNd(GenModule):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad']

    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class AvgPool2d(_AvgPoolNd):
  r"""Applies a 2D average pooling over an input signal composed of several input
  planes.

  In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
  output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
  can be precisely described as:

  .. math::

      out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                              input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

  If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
  for :attr:`padding` number of points.

  Note:
      When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
      or the input. Sliding windows that would start in the right padded region are ignored.

  The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

      - a single ``int`` -- in which case the same value is used for the height and width dimension
      - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
        and the second `int` for the width dimension

  Args:
      kernel_size: the size of the window
      stride: the stride of the window. Default value is :attr:`kernel_size`
      padding: implicit zero padding to be added on both sides
      ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
      count_include_pad: when True, will include the zero-padding in the averaging calculation
      divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.


  Shape:
      - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
      - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
              \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
              \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor

  Examples::

      >>> # pool of square window of size=3, stride=2
      >>> m = nn.AvgPool2d(3, stride=2)
      >>> # pool of non-square window
      >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
      >>> input = torch.randn(20, 16, 50, 32)
      >>> output = m(input)
  """
  __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override', 'track']

  kernel_size: _size_2_t
  stride: _size_2_t
  padding: _size_2_t
  ceil_mode: bool
  count_include_pad: bool
  track: bool

  def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, track: bool=True) -> None:
    super().__init__(track=track)
    self.kernel_size = kernel_size
    self.stride = stride if (stride is not None) else kernel_size
    self.padding = padding
    self.ceil_mode = ceil_mode
    self.count_include_pad = count_include_pad
    self.divisor_override = divisor_override

  def forward(self, viewed: torch.Tensor, input: Tensor) -> Tensor:
    # input is expected to be [N, B, C, H, W]
    assert input.ndim == 5, 'AvgPool2d expects input dim shape of [N, B, C, H, W] where N is input batch size and B is model batch size'

    N, B, C = input.shape[0], input.shape[1], input.shape[2]
    
    # view correctly
    x = input.reshape(N, B*C, input.shape[3], input.shape[4])
    
    # apply pooling
    x = F.avg_pool2d(x, self.kernel_size, self.stride,
                        self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

    # reshape back from [N, B*C, H_out, W_out]
    H_out, W_out = x.shape[2], x.shape[3]
    x = x.reshape(N, B, C, H_out, W_out)
    return self.track_feature(x), x

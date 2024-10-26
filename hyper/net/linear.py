""" Standard deviation weight normalized linear layers

NOTE: this is used in models.py and not in the hypernetwork architecture. Please
see layers.linear for the param generated version!

As described from:
1) https://arxiv.org/pdf/2101.08692.pdf
and
2) https://arxiv.org/pdf/2102.06171.pdf
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from .activation import activation_gamma


class ScaledWLinear(nn.Linear, nn.Module):
  def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, affine=True, gamma=1.0, eps=1e-5):
    nn.Module.__init__(self)
    factory_kwargs = {'device': device, 'dtype': dtype}

    # learnable gain parameter
    self.affine = affine
    if affine:
      self.affine_weight = nn.Parameter(torch.empty(out_features, 1, **factory_kwargs))
      self.affine_bias = nn.Parameter(torch.empty(out_features, 1, **factory_kwargs))
    else:
      self.register_parameter('affine_weight', None)
      self.register_parameter('affine_bias', None)
    
    # scaling of weights
    self.invsq = math.sqrt(1.0 / in_features)
    self.scale = activation_gamma(gamma) * self.invsq
    self.eps = eps

    # common linear init stuff
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
        self.register_parameter('bias', None)
    self.reset_parameters()
    self.weight.data = self._param_weight()

  def reset_parameters(self) -> None:
    """ Since we're parameterizing the weights we'll initialize them normally """
    # initialize linear parameters
    init.normal_(self.weight, 0.0, 1.0)
    if self.bias is not None:
      init.normal_(self.bias, 0.0, self.invsq)
      # init.zeros_(self.bias)

    # initialize affine parameters
    if self.affine:
      init.ones_(self.affine_weight)
      init.zeros_(self.affine_bias)

  def _param_weight(self):
    mean = torch.mean(self.weight, dim=1, keepdim=True)
    std = torch.std(self.weight, dim=1, keepdim=True, unbiased=False)
    weight = self.scale * (self.weight - mean) / (std + self.eps)
    if self.affine:
      weight = (weight * self.affine_weight) + self.affine_bias 
    return weight

  def forward(self, x):
    return F.linear(x, self._param_weight(), self.bias)

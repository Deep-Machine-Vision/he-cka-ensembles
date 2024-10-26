""" Adds dropout. Luckily, we can just use the built-in dropout layer. """
from hyper.layers.module import GenModule, register_gen_module
import torch.nn.functional as F
from functools import partial
import torch


@register_gen_module('dropout')
class Dropout(GenModule):
  def __init__(self, p: float=0.5, inplace: bool=False, mc_dropout: bool=False, track: bool=True):
    """ Wraps some torch module and handles tracking of features """
    super(Dropout, self).__init__(track=track)
    self.p = p
    self.inplace = inplace
    self.mc_dropout = mc_dropout

  def extra_repr(self) -> str:
    return f'p={self._module.p}, inplace={self._module.inplace}'

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Call dropout when training or if using mc dropout"""
    return F.dropout(
      x,
      p=self.p,
      training=self.training or self.mc_dropout,
      inplace=self.inplace
    )


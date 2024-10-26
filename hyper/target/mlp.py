""" A module that builds on top of the base to provide a basic MLP target """
from ..layers.module import SequentialModule, register_gen_module
from ..layers.linear import Linear, FinalLinear
from typing import List, Tuple, Union, Optional


@register_gen_module('mlp')
class MLPModule(SequentialModule):
  def __init__(self, layers=List[dict], activation: str='relu', act_last: bool=False, track: bool=True):
    """ Defines an MLP module """
    
    super(MLPModule, self).__init__(
      modules=[
        (Linear if not layer.get('last', False) else FinalLinear)(
          in_features=layer['in'],
          out_features=layer['out'],
          bias=layer.get('bias', True),
          act=layer.get('act', None if (ind == len(layers) - 1 and not act_last) else activation),
          gamma=layer.get('gamma', None if ind == 0 else activation),
          track=layer.get('track', True)
        ) for ind, layer in enumerate(layers)
      ],
      track=track
    )
  
  @staticmethod
  def from_config(config: dict):
    """ Builds a model from a configuration """
    return config

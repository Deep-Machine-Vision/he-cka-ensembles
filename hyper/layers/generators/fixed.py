""" Module to handle fixed ensemble parameter groups """

from typing import List, Tuple, Union
import torch
import torch.nn as nn
from collections import OrderedDict

from .base import BaseLayerGenerator, register_layer_generator
from ..module import GenModule, ParametersGroup


@register_layer_generator('ensemble_parameter_generator')
class EnsembleParametersGroupGenerator(BaseLayerGenerator):
  def __init__(self, definition: ParametersGroup, fixed_size: int):
    """ Defines a fixed size parameter ensemble layer that keeps track of a set of parameters

    Args:
        definition (GenModule): the model layer description to make a generator module for
        fixed_size (int): the number of parameters to collect
    """
    if not isinstance(definition, ParametersGroup):
      raise ValueError('The EnsembleParametersGroupGenerator object only supports known size parameters, which requires a ParametersGroup object')

    super(EnsembleParametersGroupGenerator, self).__init__(definition, input_size=fixed_size)
    self.fixed_size = fixed_size
    if self.fixed_size <= 0 or isinstance(self.fixed_size, float):
      raise ValueError('Fixed size must be a non-negative integer')

    # construct the ensemble set of parameters
    self.fixed_parameters = nn.ParameterDict()
    params_list = [definition.gen_initialized_params() for _ in range(self.fixed_size)]  # uses layer definition to automatically initialize parameters with right gain
    
    # create the list for each parameter name
    for name in params_list[0].keys():
      self.fixed_parameters[name] = nn.Parameter(torch.stack([params_list[ind][name] for ind in range(self.fixed_size)]), requires_grad=True)

    # calculate flat size
    self.flat_size = 0
    for value in self.fixed_parameters.values():
      self.flat_size += int(value[0].numel())

  def random(self, num, std=1.0, device=None):
    """ Specify foward from the "latents" which in this case is just an index
    
    Args:
      num: num models to generate [model batch size, get_expected_input_size(), ] 
    """
    
    params = torch.empty(num, self.flat_size, device=device)
    torch.nn.init.normal_(params, mean=0.0, std=std)
    return params

  def forward(self, codes):
    """ Specify foward from the "latents" which in this case is just an index
    
    Args:
      codes: generated codes from parent network/process in format [model batch size, get_expected_input_size(), ] 
    """
    indices = codes.long().ravel()

    # this seems weird but what we're doing
    # is selecting the respective tensors associated with the particle
    # then flattening (in the right order) the parameters
    selected_params = [torch.index_select(value, 0, indices).view(len(indices), -1) for value in self.fixed_parameters.values()]

    # @TODO verify by adding to unit tests
    # recon = torch.concat([torch.index_select(value, 0, indices).view(len(indices), -1) for value in [self.fixed_parameters['weight'], self.fixed_parameters['bias']]], dim=1) 
    # print(recon.shape)
    # weights = recon[:, :self.fixed_parameters['weight'][0].numel()].view_as(self.fixed_parameters['weight'])
    # print(weights.shape, self.fixed_parameters['weight'][0].numel())
    # bias = recon[:, self.fixed_parameters['weight'][0].numel():].view_as(self.fixed_parameters['bias'])
    # print(bias.shape)
    # print(torch.norm(weights - self.fixed_parameters['weight']), torch.norm(bias - self.fixed_parameters['bias']))

    return torch.concat(selected_params, dim=1)  # combine into flattened view along parameter dimension


class EnsembleBufferGroupGenerator(BaseLayerGenerator):
  def __init__(self, definition: ParametersGroup, fixed_size: int):
    """ Defines a fixed size buffer ensemble layer that keeps track of a set of buffers

    Args:
        definition (GenModule): the model layer description to make a generator module for
        fixed_size (int): the number of parameters to collect
    """
    if not isinstance(definition, ParametersGroup):
      raise ValueError('The EnsembleBufferGroupGenerator object only supports known size buffers, which requires a ParametersGroup object')

    super(EnsembleBufferGroupGenerator, self).__init__(definition, input_size=fixed_size)
    self.fixed_size = fixed_size
    if self.fixed_size <= 0 or isinstance(self.fixed_size, float):
      raise ValueError('Fixed size must be a non-negative integer')

    # construct the ensemble set of parameters
    params_list = [definition.gen_initialized_params() for _ in range(self.fixed_size)]  # uses layer definition to automatically initialize parameters with right gain
    
    # create the list for each parameter name
    self.names = list(params_list[0].keys())
    for name in self.names:
      for ind in range(self.fixed_size):
        self.register_buffer(f'mod-{ind}-{name}', params_list[ind][name])

    # calculate flat size
    self.flat_size = 0
    for value in self.buffers(recurse=False):
      self.flat_size += int(value[0].numel())

  def random(self, num, std=1.0, device=None):
    """ Specify foward from the "latents" which in this case is just an index
    
    Args:
      num: num models to generate [model batch size, get_expected_input_size(), ] 
    """
    
    # just use the default buffer values when creating a random model
    all = OrderedDict()
    for name in self.names:
      all[name] = [self.definition.gen_initialized_params() for ind in range(self.fixed_size)]
    return OrderedDict(all)

  def forward(self, codes):
    """ Specify foward from the "latents" which in this case is just an index
    
    Args:
      codes: generated codes from parent network/process in format [model batch size, get_expected_input_size(), ] 
    """
    indices = codes.long().ravel().cpu().tolist()
    all = OrderedDict()
    for name in self.names:
      all[name] = [self.get_buffer(f'mod-{ind}-{name}') for ind in indices]
    return OrderedDict(all)

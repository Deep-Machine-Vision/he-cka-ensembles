""" A model generator that is just an ensemble of models """
from .base import ModelGenerator, LayerCodeModelGenerator, GenModule, register_generator
from ..layers.module import GenModule, ParametersGroup
from ..layers.generators.fixed import EnsembleParametersGroupGenerator, EnsembleBufferGroupGenerator
from ..layers.linear import Linear, ScaledWLinear, FinalLinear, FinalScaledWLinear
from ..layers.conv import Conv2d, ScaledWConv2d
from ..layers.norm import _NormAffine, _NormBuffers
from ..util.collections import DefaultOrderedDict
from functools import partial
import torch


@register_generator('ensemble')
class FixedEnsembleModel(LayerCodeModelGenerator):
  def __init__(self, target: GenModule, ensemble_size: int):
    """ Creates a base generator that just describes the target module with a certain ensemble size

    Args:
      target (GenModule): the target module that defines the structure of the target model and specifies which parameters are "generated" for ensemble it's a fixed set of particles

    """
    super(FixedEnsembleModel, self).__init__(
      target=target,
      code_size=1,  # just index of model
      default_generators={
        GenModule: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        ParametersGroup: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        Linear: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        FinalLinear: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        Conv2d: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        FinalScaledWLinear: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        ScaledWLinear: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        ScaledWConv2d: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        
        # norm specific modules
        _NormAffine: partial(EnsembleParametersGroupGenerator, fixed_size=ensemble_size),
        _NormBuffers: partial(EnsembleBufferGroupGenerator, fixed_size=ensemble_size)
      }
    )
    self.size = ensemble_size

  def sample_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    if size is None:  # sample all models
      return torch.arange(0, self.size, 1, device=device, dtype=torch.long)
    elif isinstance(size, (list, set)):
      return torch.tensor(size, device=device, dtype=torch.long)  # explicitly specify indices
    return torch.randperm(self.size, device=device, dtype=torch.long)[:size]

  def sample_random_params(self, size: int, device=None):
    """ Returns a sample of random input to parameters """
    return str(size)  # hacky way but it works (see forward_params for impl)

  def sample_local_best(self, data, num_samples: int, num_iters: int, lr: float, device=None):
    raise NotImplementedError('Ensemble models do not support local best sampling')

  def forward_params(self, codes, device=None):
    """ Handles taking the generated codes (from some other model) and forwarding them through the individual generators

    Args:
        codes (torch.Tensor): A tensor of [B, total codes, code size] values to generate parameters from
    """
    is_rand = isinstance(codes, str)
    
    if not is_rand:
      if isinstance(codes, int):  # number to sample
        codes = torch.arange(0, codes, 1, device=device, dtype=torch.long)
      elif codes is None:
        codes = torch.arange(0, self.size, 1, device=device, dtype=torch.long)
      assert len(codes.shape) <= 1, 'Invalid number of code dims. Expecting 1'

      # assert right indices
      if torch.any(codes < 0) or torch.any(codes >= self.size):
        raise ValueError(f'Invalid model index in codes. Please only select members from indices 0 to {self.size}')
    else:
      # random codes
      num_rand = int(codes)
      codes = torch.arange(0, self.size, 1, device=device, dtype=torch.long)
    
    # add dim if single model
    if len(codes.shape) == 0:
      codes = codes.unsqueeze(0)
    
    # convert to long/right index
    indices = codes.long()

    # keep track of the current code offset
    code_offset = 0

    # creates an easy way to assign parameters
    params = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      # run through layer generator with expected number of codes
      if is_rand:
        params.set_subitems(names, layer_generator.random(num_rand, device=device))
      else:
        params.set_subitems(names, layer_generator(indices))
        
      # move linearly through codes
      code_offset += layer_num_code

    return params
  
  @staticmethod
  def from_config(config: dict):
    config = ModelGenerator.from_config(config)
    return config

""" Base module to share functions across layer generator modules """
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import copy
from collections import OrderedDict

from ..linear import Linear
from ..module import GenModule, ParametersGroup, build_gen_module
from ...net import models, activation


AVAILABLE_LAYER_GENERATORS = {}

def register_layer_generator(name: str):
  """ Decorator to register a layer generator """
  def decorator(cls):
    AVAILABLE_LAYER_GENERATORS[name] = cls
    return cls
  return decorator


def build_layer_generator(config: dict, module: GenModule):
  """ Builds a model """
  config = copy.deepcopy(config)
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')

  cls = AVAILABLE_LAYER_GENERATORS[name]
  config = cls.from_config(config)
  return cls(definition=module, **config)


EXPECTED_STD = 0.5039466   # expected STD from layer generators
# we want to correct gain of output layers to match the expected
# target distribution

def null_from_flat(shape):
  raise ValueError('Cannot get shape flat for undefined layer')


class BaseLayerGenerator(nn.Module):
  def __init__(self, definition: GenModule, input_size: int):
    super(BaseLayerGenerator, self).__init__()
    self.definition = definition
    self.input_size = input_size

  def get_flat_size(self):
    """ Gets the flat number of elements for the given layer """
    return self.definition.single_flat_size()

  def get_expected_input_size(self):
    """ Tells the hyper network how big the input size should be as either the latent size or an integer of number of input size """
    return 1

  def get_parameter_shapes(self) -> OrderedDict:
    """ Gets the shapes of the parameters for the layer. Must inherit from ParametersGroup """
    if isinstance(self.definition, ParametersGroup):
      return self.definition.shapes
    return OrderedDict()

  def random(self, num, std=1.0, device=None):
    """ Specify foward from the "latents" which in this case is just an index
    
    Args:
      num: num models to generate [model batch size, get_expected_input_size(), ] 
    """
    
    params = torch.empty(num, self.get_flat_size(), device=device)
    torch.nn.init.normal_(params, mean=0.0, std=std)
    return params

  def from_flat(self):
    """ Gets the shape viewers for the layer. Must inherit from ParametersGroup """
    if isinstance(self.definition, ParametersGroup):
      return self.definition.from_flat
    return null_from_flat

  def from_shapes(self):
    """ Get the from shapes indices """
    if isinstance(self.definition, ParametersGroup):
      return self.definition.from_shapes
    return []

  @staticmethod
  def from_config(config: dict):
    """ Builds the layer generator from the configuration """
    if 'definition' in config:
      config['definition'] = build_gen_module(config['definition'])
    return config

class Scaler(nn.Module):
  def __init__(self, val):
    super(Scaler, self).__init__()
    self.val = val
  
  def forward(self, x):
    return self.val * x


@register_layer_generator('mlp_layer_generator')
class MLPLayerGenerator(BaseLayerGenerator):
  def __init__(self, definition: GenModule, input_size: int, mlp_dims: Union[int, List[int]]=3, dim_multiplier: float=2, bias: bool=False, gamma: Union[float, str, object]=1.0, norm_last: bool=True, norm_before_last: bool=False, act=activation.Crater, affine_last: bool=True, sn_coeff: float=3.0):
    """ Define a basic parameter generator using GELU, Linear, and GroupNorm with some flexibility for size

    Args:
        definition (GenModule): the model layer description to make a generator module for
        input_size (int): the size the generator should expect as an input (from some other hypernetwork or latent space)
        mlp_dims (Union[int, List[int]]): the center dimensions of the layer generator (including first layer excluding the last linear to project to parameter space). If it's an int it's just the number of layers to generate (look at mlp_multiplier), else it's a list of dimensions
        dim_multiplier (float, optional): if mlp_dims is an integer then use this multiplier to scale the number of parameters of each layer up linearly from latent_size*mlp_multiplier (only use if mlp_dims is int). Defaults to 2.
        bias (bool, optional): to use the bias term in the generators. Defaults to False.
        gamma (check nfnet/activation.py): calculates input gamma/scaling of weights based on previous output. Default 1.0 meaning expecting input to have variance of 1.0,
        norm_last (bool): apply normalization on the weights (recommended on hidden layers, not recommended on last layer of target network). Default True.
        norm_before_last (bool): apply normalization on the linear layer before projection  (recommended on hidden layers, not recommended on last layer of target network). Default True.
    """
    super(MLPLayerGenerator, self).__init__(definition, input_size)

    # build a basic MLP
    self.generator = models.build_groupnorm_mlp(
      in_features=input_size,
      out_features=self.get_flat_size(),
      mlp_dims=mlp_dims,
      dim_multiplier=dim_multiplier,
      gamma=gamma,
      bias=bias,
      norm_last=norm_last,
      norm_before_last=norm_before_last,
      activation=torch.nn.GELU,
      # activation=act,
      affine_last=affine_last,
      norm_last_groups=1,  # output layer norm
      out_act=None,  # nothing on weight space (if changed look at init below for changes)
      coeff=sn_coeff  #3.5
    )

    """ @TODO remove after testing
    # apply scaling to linear parameters
    # due to weight normalization
    if isinstance(definition, Linear):
      with torch.no_grad():
        fan_in = definition.shapes['weight'][1]  # get the number of input features

        # specify expected initial distribution on weights
        # k = math.sqrt(2) * math.sqrt(1.0 / in_feat)
        
        # calculate difference on bounds
        exp_var = 0.95  # EXPECTED_STD ** 2
        diff = math.sqrt(1.0 / (exp_var * (in_feat) * (1.0 if in_feat == 2 else 0.789196473029)))  # (1.0 / EXPECTED_STD) * (k**2)

        # commments on weirdness
        # 1.5 represents some empircal fix for linear STD projection
        # (1.0 / EXPECTED_STD) represents correction gain to 1 STD after initial STD fix
        # the k is as followed for kaiming_normal init
        # std = 1.0 * (1.0 / EXPECTED_STD) * k # new expected distribution
        # p_i = in_hyper_feat
        # rng = 1.3 * math.sqrt(6*4) / math.sqrt(((p_i + out_hyper_feat) * exp_var * (in_feat + out_feat)))
        # init.uniform_(self.generator[-1].weight, -rng, rng)
        self.generator[-1].weight *= diff
        if self.generator[-1].bias is not None:
          self.generator[-1].bias *= diff
        # std = math.sqrt(2.0 / ((in_feat + out_feat) * (EXPECTED_STD**2)))
        # std = math.sqrt(2.0 / (p_i * exp_var * (in_feat + out_feat)))
        # init.normal_(self.generator[-1].weight, 0.0, std)
        # if self.generator[-1].bias is not None:
        #   init.zeros_(self.generator[-1].bias)

        # init.normal_(self.generator[-1].weight, 0.0, k)
        # self.generator[-1].weight *= std

        # if self.generator[-1].bias is not None:
        #   self.generator[-1].bias *= k

      # self.generator.append(
      #   Scaler(diff)
      # )


      # product of W=XY (https://en.wikipedia.org/wiki/Distribution_of_the_product_of_two_random_variables)
      # Var(XY) = (sx^2 + mux^2)(sy^2 +muy^2)-mux^2muy^2
      # Ideally we want balance between target layers to have
      # Var(W) = 2 / (n_i + n_i+1) where n_i and n_i + 1 are target model feature num http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
      # Assume all mu are approx zero 
      # Now Var(W) = 2/(n_i + n_i+1) = (sx^2)(sy^2)
      # we want to find sx and sy 
      
      # assume we know sy empirically
      # So we want to set sx = sqrt(2/(n_i + n_+1)sy^2)
    """
  
  def forward(self, codes):
    """ Specify foward from the latents
    
    Args:
      codes: generated codes from parent network/process in format [model batch size, get_expected_input_size(), ] 
    """
    out_codes = self.generator(codes.squeeze(1))  # we squeeze layer codes dimension as we only accept one code
    return out_codes
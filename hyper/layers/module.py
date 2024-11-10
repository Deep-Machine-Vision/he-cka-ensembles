""" Basic layer definitions for a target model. Essentially telling the model generator which layers it expects to be generated """
from typing import List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from hyper.net.activation import crater, leakysoftstep
from collections import OrderedDict
import copy

AVAILABLE_GEN_MODULES = {}

def register_gen_module(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_GEN_MODULES[name] = cls
    return cls
  return decorator


def build_gen_module(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_GEN_MODULES[name]
  
  # some methods/classes do not define a conversion from a dictionary
  if hasattr(cls, 'from_config'):
    config = cls.from_config(config)
  
  return cls(**config)


@register_gen_module('gen_module')
class GenModule(nn.Module):
  def __init__(self, track: bool=True):
    """ Base batched layer object that just returns identity from generator
    
    Args:
      track (bool|str): track the internal feature. Use 'detached' to track without backprop. Default is True
    """
    super(GenModule, self).__init__()
    self._track = track

  def is_generated(self):
    """ Tells any implementing model/hyper generator that this particular module can have something assigned to it

    To see a use case of this look at ParametersGroup which does not define any sub-modules but itself can be generated
    """
    return False
  
  def define_generated_modules(self):
    """ Define the generated layers for this target model

    This gives you flexibility to define the type of layers the hyper network should generate
    
    NOTE: any submodules will be assumed to be generated AFTER the fact 
          if you need that changed ie a layer generator to generate children
          and then self then sub-class this/override this method
    """
    base = OrderedDict() # a parametersgroup does define itself as generated
    if self.is_generated():
      base['self'] = self
    return base

  def single_flat_size(self) -> int:
    """ Gets the size of vector for a single model """
    raise NotImplementedError('GenModule.single_flat_size is not implemented. Please consider using a sub-class')

  def from_flat(self, params: torch.Tensor):
    """ Function that returns some shape given a flat representation (output of generator) of it.

    Args:
      params (torch.Tensor): flattened representation of layer [B, P] where B is number of models

    Returns:
      torch.Tensor: reshaped tensor (base is identity) [B, P]
    """
    return params
  
  def track_feature(self, feat: torch.Tensor):
    """ Handles the track/detach portion of the feature return

    Args:
        feat (torch.Tensor): the internal feature to track or possibly tracked detached
    """
    if self._track == 'detach':
      return feat.detach()
    elif not self._track:
      # print('NOT TRACKING', self)
      return None
    # print('TRACKING', self)
    return feat

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Internal forward function that will automatically have the right view and features tracked """
    return x

  def forward(self, params: torch.Tensor, x: torch.Tensor):
    """ Forwards the x features through params """
    viewed = self.from_flat(params)
    y = self._forward(viewed, x)
    return self.track_feature(y), y

  @staticmethod
  def from_config(config: dict):
    """ Builds a model from a configuration """
    return config


class WrapGen(GenModule):
  def __init__(self, module: nn.Module, name: str=None, track: bool=True):
    """ Wraps some torch module and handles tracking of features 
    
    NOTE: when calling forward you could POTENTIALLY pass in a normal module like nn.GELU or any activation
    that applies elementwise since that's independent of model batch. However, you can't just throw in modules
    like nn.Linear, nn.Conv2D as those don't support model batches. Expects any module to have input [model batch size, *] or be completely element-wise
    
    """
    self._module = module
    self._name = name if not (name is None) else str(module.__class__.__name__) 
    super(WrapGen, self).__init__(track=track)

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Wraps the original module in a batched generated one naively """
    return self._module(x)


@register_gen_module('shared_linear')
class SharedLinear(GenModule):
  def __init__(self, in_feat, out_feat, act=None, name: str=None, track: bool=True):
    """ Wraps some torch module and handles tracking of features 
    
    NOTE: when calling forward you could POTENTIALLY pass in a normal module like nn.GELU or any activation
    that applies elementwise since that's independent of model batch. However, you can't just throw in modules
    like nn.Linear, nn.Conv2D as those don't support model batches. Expects any module to have input [model batch size, *] or be completely element-wise
    
    """
    super(SharedLinear, self).__init__(track=track)
    self._module = nn.Linear(in_feat, out_feat)
    self._outf = out_feat
    common = {
      'relu': F.relu,
      'relu6': F.relu6,
      'sigmoid': F.sigmoid,
      'tanh': F.tanh,
      'softmax': F.softmax,
      'mish': F.mish,
      'elu': F.elu,
      'gelu': F.gelu,
      'crater': crater,
      'leakysoftstep': leakysoftstep
    }
    if act is not None:
      self.act = common[act]
    else:
      self.act = None

  def _forward(self, viewed: torch.Tensor, x: torch.Tensor):
    """ Wraps the original module in a batched generated one naively """
    mbs, bs, inf = x.shape
    x = x.reshape(mbs * bs, inf)
    x = self._module(x)
    x = x.reshape(mbs, bs, self._outf)

    if self.act is not None:
      x = self.act(x)
    
    return x

@register_gen_module('reshape')
class Reshape(GenModule):
  def __init__(self, shape: Tuple[int], track: bool = True):
    """ Simple layer that reshapes/views the data passing through that can be trackable

    Args:
      shape (tuple): the reshape to make (includes batch dimension). let element 'B' indicate batch dimension
      track (bool, optional): _description_. Defaults to True.
    """
    super(Reshape, self).__init__(track)

    self.has_batch = 'B' in shape
    self.batch_dim = shape.index('B') if self.has_batch else None
    
    if self.has_batch:
      if self.batch_dim > 0:
        self.shape = (shape[:self.batch_dim], shape[self.batch_dim + 1:])
      else:
        self.shape = shape[1:]  # it's first elem
    else:
      self.shape = shape

  def _forward(self, params: torch.Tensor, x: torch.Tensor):
    """ Forwards the x features through params """
    if self.has_batch:
      if self.batch_dim > 0:
        return x.reshape(*self.shape[0], x.shape[0], *self.shape[1])
      return x.reshape(x.shape[0], *self.shape)
    return x.reshape(*self.shape)


@register_gen_module('flatten')
class Flatten(GenModule):
  def __init__(self, track: bool = True):
    """ Simple layer that flattens the data along no-batch dim passing through that can be trackable

    Assumes input is [N, B, ....] where B is model batch size, N is sample batch size
    
    Args:
      shape (tuple): the reshape to make
      track (bool, optional): _description_. Defaults to True.
    """
    super(Flatten, self).__init__(track)

  def _forward(self, params: torch.Tensor, x: torch.Tensor):
    """ Forwards the x features through params """
    return x.reshape(x.shape[0], x.shape[1], -1)


@register_gen_module('parameter_group')
class ParametersGroup(GenModule):
  def __init__(self, shapes: Union[List[Tuple[str, tuple]], OrderedDict], shape_type=None, track: bool=True):
    """ Parameters layer object that takes the batched model parameters and uses them
    
    Args:
      shapes (List[tuple] or OrderedDict): name and shapes of parameters to generate for the target model
      track (bool): track the internal feature. Use 'detach' if you wish to track but not backprop. Default is True
    """
    super(ParametersGroup, self).__init__(
      track=track
    )

    if not isinstance(shapes, OrderedDict):
      self.shapes = OrderedDict(shapes)
    else:
      self.shapes = shapes
    self.shape_type = shape_type
  
    # ensure correctness
    if len(self.shapes) == 0:
      raise ValueError('shapes must not be empty')
    
    # verify parameters!
    for n, s in self.shapes.items():
      if n is None:
        raise ValueError('shape name must not be None')
      elif not isinstance(n, str):
        raise ValueError('shape name must be a string')
      
      if s is None:
        raise ValueError('shape size must not be None')
      elif not isinstance(s, (list, tuple)):
        raise ValueError('shape size must be a list of values/tuple')
      elif len(s) == 0:
        raise ValueError('shape size must not be empty!')
      elif any([v == 0 for v in s]):
        raise ValueError('no shape dimension may be zero!')

    # precompute shape offsets for from_flat to reduce compute each time (minor opt)
    flats = self.get_param_flats()
    offset = 0
    self.from_shapes = []
    for name, shape, flat in zip(self.shapes.keys(), self.shapes.values(), flats.values()):
      self.from_shapes.append((name, shape, offset, offset + flat))
      offset += flat

  def is_generated(self):
    """ Tells any implementing model/hyper generator that this particular module can have something assigned to it

    Anything sub-classing a ParametersGroup will now be seen to any hypernetwork/generator as a valid module
    """
    return True

  def names(self):
    """ Return a list of parameter names """
    return list(self.shapes.keys())

  def gen_empty_params(self, dtype=None, device=None, requires_grad=True):
    """ By default we generate parameters but sometimes it's useful to generate an empty version, ie for an ensemble, of the parameters  """
    params = OrderedDict()
    for name, shape in self.shapes.items():
      params[name] = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return params

  def gen_initialized_params(self, dtype=None, device=None, gain=1.0):
    """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
    params = self.gen_empty_params(dtype=dtype, device=device)
    for name, shape in self.shapes.items():
      try:
        init.xavier_normal_(params[name], gain=gain)
      except ValueError:
        init.uniform_(params[name], -0.1, 0.1)
    return params

  def get_shapes(self) -> OrderedDict:
    """ Gets the names and shapes of parameters """
    return self.shapes

  def get_param_flats(self) -> OrderedDict:
    """ Gets the number of flattened params per shape """
    ret = OrderedDict()
    for n, s in self.shapes.items():
      ret[n] = int(np.prod(s))
    return ret

  def single_flat_size(self) -> int:
    """ Gets the size of vector for a single model """
    return sum(v for v in self.get_param_flats().values())
  
  def from_flat(self, params: torch.Tensor) -> OrderedDict:
    """ Function that returns some shape given a flat representation (output of generator) of it.

    Args:
      params (torch.Tensor): flattened representation of layer [B, P] where B is number of models and P is the flattened representation

    Returns:
      OrderedDict[str, torch.Tensor]: each parameter with a reshaped tensor [B, name's shape...]
    """
    
    # DO NOT MODIFY UNLESS YOU KNOW WHAT YOU ARE DOING!
    # a lot of modules depend on this exact structure/ordering
    rep = OrderedDict()
    batch_dim = params.shape[0]
    for name, shape, from_index, to_index in self.from_shapes:
      rep[name] = params[:, from_index:to_index].view((batch_dim,) + shape)  # view from flat to expected shape for 
    return rep

  def _forward(self, params: torch.Tensor, x: torch.Tensor):
    """ Forwards the x features through params """
    raise NotImplementedError('_forward for ParametersGroup is ill-defined. Please consider using a sub-class')

  def forward(self, params: torch.Tensor, x: torch.Tensor):
    """ Forwards the x features through params """
    # since we are a is_generated module
    # we must split parameters for self and sub-modules
    # if you wish to access sub-module parameters then override this function
    viewed = self.from_flat(params['self'])
    y = self._forward(viewed, x)
    return self.track_feature(y), y


@register_gen_module('activation')
class Activation(WrapGen):
  def __init__(self, name: str, track: bool=True):
    """ Wraps some torch module and handles tracking of features """
    self._name = name

    # wrap some common ones
    common = {
      'relu': F.relu,
      'relu6': F.relu6,
      'leaky_relu': F.leaky_relu,
      'sigmoid': F.sigmoid,
      'tanh': F.tanh,
      'softmax': F.softmax,
      'mish': F.mish,
      'elu': F.elu,
      'gelu': F.gelu,
      'silu': F.silu,
      'crater': crater,
      'leakysoftstep': leakysoftstep
    }
    
    if isinstance(name, str) and name not in common:
      raise ValueError(f'invalid activation provided. please select from {list(common.keys())}')
    elif isinstance(name, str):
      mod = common[name]
    else:
      mod = name

      # convert name to something more useful/friendly
      if name.__class__ == 'function':
        name = name.__name__
      else:
        name = name.__class__.__name__ 

    super(Activation, self).__init__(module=mod, name=name,track=track)


@register_gen_module('sequential')
class SequentialModule(GenModule, nn.Sequential):
  def __init__(self, *args, track: bool=True, modules: List[GenModule]=None):
    """ Very similar to nn.Sequential except that it includes sub-features

    Args:
      track (bool): whether to track features for this module/sub-modules
    """

    # initialize base module
    GenModule.__init__(self, track=track)

    # initialize sequential (has to come after)
    if len(args) > 0:
      nn.Sequential.__init__(self, *args)
    elif modules is not None:
      nn.Sequential.__init__(self, *modules)
    else:
      nn.Sequential.__init__(self)  # no modules added yet

  def define_generated_modules(self):
    """ Define the generated layers for this target model

    This gives you flexibility to define the type of layers the hyper network should generate.
    """
    mods = GenModule.define_generated_modules(self)
    for name, module in self._modules.items():
      # add all generated sub-modules
      if isinstance(module, GenModule):
        mods[name] = module.define_generated_modules()
    return mods

  def forward(self, params, x):
    """ Custom forward that also keeps track of internal features """
    features = OrderedDict()
    current = x
    for name, module in self._modules.items():
      # get the generated parameters for the sub-module
      if isinstance(module, GenModule):
        p = params[name]
      else:
        p = None  # pass in nothing
      
      # pass in generated parameters and current features
      sub_features, current = module(p, current)

      # update module via sub-features
      if self._track:
        features[name] = sub_features
    return features, current

  @staticmethod
  def from_config(config: dict):
    """ Builds a model from a configuration """
    
    # need to build sequential modules recursively potentially
    modules = []
    for mod in config['modules']:
      if 'modules' in mod and mod['name'] == 'sequential':
        modules.append(SequentialModule.from_config(mod))
      else:
        modules.append(build_gen_module(mod))
    
    config['modules'] = modules
    return config


# for ease of use people can also just use Module
Module = GenModule
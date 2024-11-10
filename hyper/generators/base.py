""" Defines the base model generator """
from typing import List, Union, Tuple, Dict
from functools import partial
import torch
import torch.nn as nn
from hyper.diversity import hyperspherical_energy, pairwise_cossim

from hyper.diversity.kernels import batched_gram
from ..layers.generators.base import BaseLayerGenerator, build_layer_generator
from ..layers.module import GenModule, build_gen_module, AVAILABLE_GEN_MODULES
from ..util.collections import DefaultOrderedDict, flatten_keys
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt


AVAILABLE_MODELS = {}


def register_generator(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_MODELS[name] = cls
    return cls
  return decorator


def build_generator(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_MODELS[name]
  config = cls.from_config(config)
  return cls(**config)


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


@register_generator('base')
class ModelGenerator(nn.Module):
  def __init__(self, target: GenModule=None):
    """ Creates a base generator that just describes the target module
    
    You can subclass this class to be able to define more flexible model generators
    """
    super(ModelGenerator, self).__init__()
    self.target = target

    # scan target model and assign default generators
    self.gen_defined = self.define_generated_modules()

  def get_target(self) -> nn.Module:
    """ Get the target module """
    return self.target

  def target_parameters(self):
    """ Return the parameters of the target network"""
    return self.get_target().parameters()

  def define_generated_modules(self):
    """ Get the generated modules defined by the target module"""
    if self.target is None:
      return GenModule.define_generated_modules(self)
    return self.target.define_generated_modules()

  def _iter_bfs_defined(self, definitions=None, target_dict=None, target_dict_type=OrderedDict):
    """ (INTERNAL: use iter_bfs) A (python) generator [too many things are called generators...] on the generated defined ordered dict tree in a way s.t all dependent
          defined parameters appear after.
    """
    for name, defined in definitions.items():
      # now yield all children
      if isinstance(defined, (dict, OrderedDict, nn.ModuleDict)):
        if target_dict is not None and name not in target_dict:
          target_dict[name] = target_dict_type()  # create new instance of target dictionary
        yield from self._iter_bfs(definitions[name], target_dict=target_dict[name] if target_dict is not None else None, target_dict_type=target_dict_type)
      else:
        yield name, defined, target_dict
  
  def iter_bfs_defined(self, target_dict=None, target_dict_type=OrderedDict):
    """ A (python) generator [too many things are called generators...] on the generated defined ordered dict tree in a way s.t all dependent
          defined parameters appear after.
    """
    return self._iter_bfs_defined(self.define_generated_modules(), target_dict, target_dict_type)

  def sample_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    raise NotImplementedError('Sample params not defined for base class model. Use subclass')

  def sample_random_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    raise NotImplementedError('Sample params not defined for base class model. Use subclass')
  
  def sample_forward(self, size: int, x=None):
    """ Samples a set of models of specified size and feeds through data

    Args:
        size (int): number of models to randomly sample
    """
    param_init = self.sample_params(size, device=x.device)
    return self.forward(self.forward_params(param_init), x)

  def sample_local_optim(self, data, num_samples: int, objective, gamma, lr: float=1e-3, device='cuda', max_iter=80):
    """ Samples a set of models of specified size and feeds through data

    Args:
        x (torch.Tensor): the input data to the model
        num_samples (int): number of models to randomly sample
        num_iters (int): number of iterations to run the local search
        lr (float): the learning rate for the local search
    """
    if self.training:
      self.eval()
      was_training = True
    else:
      was_training = False
    
    # sample initial parameters
    param_init = self.sample_params(num_samples, device=device)
    
    # make grad copy
    if isinstance(param_init, list):
      params = [p.clone().detach().requires_grad_(True) for p in param_init]
      multi = True
    elif isinstance(param_init, torch.Tensor):
      params = param_init.clone().detach().requires_grad_(True)
      multi = False

    # search for best model in parameter space
    optim = torch.optim.SGD((params if multi else [params]), lr=lr, nesterov=True, momentum=0.9)
    
    # do local search against objective
    num_train = len(data)
    tq_train = tqdm(data, desc='Finding local best', total=num_train, colour='green')
    init_loss = None
    for b, (X, Y) in enumerate(tq_train):
      if b > max_iter:
        print('Hit max iter')
        break
      X_o = X.cuda()
      Y_o = Y.cuda()
      
      # feed params through/run target network and backprop to init codes
      target_params = self.forward_params(params)
      feats, pred = self.forward(target_params, X_o)
      feats = flatten_keys(feats)
      
      # make feature matrices
      features = []
      for ind, feat in enumerate(feats.values()):
        if feat is None:
          continue  # skip any non-tracked features
        mbs, bs = feat.shape[:2]
        features.append(feat.reshape(mbs, pred.shape[1], -1))
      
      first = features[0]
      with torch.no_grad():
        weights = torch.linspace(1.0, 10.0, len(features), device=first.device, dtype=first.dtype)
        # last layer has first layer's CKA weight
        # weights[-1] = weights[int(len(weights) / 2)]
        weights /= weights.sum()
      
      d_loss = 0.0
      for ind, features in enumerate(features):
        grams = batched_gram(features, features, kernel='linear', detach_diag=True, center=True)
        mhe = hyperspherical_energy(grams, half_space=False, s=5.5, arc_eps=0.05, eps=1e-7)
        d_loss += weights[ind] * mhe
      
      # do step
      optim.zero_grad()
      loss = objective(num_samples, X_o, Y_o, pred) + (gamma * d_loss)
      loss.backward()
      
      # print(params.grad)
      optim.step()
      
      if init_loss is None:
        init_loss = loss.item()
      
      # print('OPT?')
      tq_train.set_description(f'Local best loss: {loss.item()}')
    
    # report improvement
    if init_loss is not None:
      print('Loss Improvement', init_loss, loss.item())
    
    # put back into train if that was the previous state
    if was_training:
      self.train()
    
    # return locally optimal parameters
    return params

  def _uHSIC(self, K, L):
    """
    Computes the unbiased estimate of HSIC metric.

    Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    """
    N = K.shape[0]
    ones = torch.ones(N, 1).to(K.device)
    result = torch.trace(K @ L)
    result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
    result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
    return (1 / (N * (N - 3)) * result).item()


  def unbiased_cka(self, data, num_samples: int=5, device='cuda', save_path=None, title=None):
    if self.training:
      self.eval()
      was_training = True
    else:
      was_training = False
    
    # sample initial parameters
    params = self.sample_params(num_samples, device=device)
    
    # do local search against objective
    num_train = len(data)
    tq_train = tqdm(data, desc='Comparing features', total=num_train, colour='green')
    init_loss = None
    hsic_matrix = None
    for b, (X, Y) in enumerate(tq_train):
      X_o = X.cuda()
      Y_o = Y.cuda()
      
      # feed params through/run target network and backprop to init codes
      target_params = self.forward_params(params)
      feats, pred = self.forward(target_params, X_o)
      feats = flatten_keys(feats)
      
      # make feature matrices
      features = []
      for ind, feat in enumerate(feats.values()):
        if feat is None:
          continue  # skip any non-tracked features
        mbs, bs = feat.shape[:2]
        features.append(feat.reshape(mbs, pred.shape[1], -1))
      
      if hsic_matrix is None:
        hsic_matrix = torch.zeros((mbs, mbs, len(features), len(features), 3), dtype=torch.double, device=device)
      
      for m_i in range(mbs):
        for m_j in range(mbs):
          if m_j <= m_i:
            continue

          for layer_i in range(len(features)):
            feat1 = features[layer_i][m_i]
            if feat1.ndim < 2:
              continue
            K = feat1 @ feat1.t()
            K.fill_diagonal_(0.0)
            for layer_j in range(len(features)):
              if layer_i < layer_j:
                continue
            
              feat2 = features[layer_j][m_j]
              if feat2.ndim < 2:
                continue
              
              L = feat2 @ feat2.t()
              L.fill_diagonal_(0.0)
              
              kk = self._uHSIC(K, K) / num_train
              kl = self._uHSIC(K, L) / num_train
              ll = self._uHSIC(L, L) / num_train
              
              # print('MODEL', m_i, m_j, 'LAYER', layer_i, layer_j, 'VALS',  kk, kl, ll)
              
              hsic_matrix[m_i, m_j, layer_i, layer_j, 0] += kk
              if layer_i != layer_j:
                hsic_matrix[m_i, m_j, layer_j, layer_i, 0] += kk
              hsic_matrix[m_i, m_j, layer_i, layer_j, 1] += kl
              if layer_i != layer_j:
                hsic_matrix[m_i, m_j, layer_j, layer_i, 1] += kl
              hsic_matrix[m_i, m_j, layer_i, layer_j, 2] += ll
              if layer_i != layer_j:
                hsic_matrix[m_i, m_j, layer_j, layer_i, 2] += ll
              
      # break
      # if b > 10:
      #   break
    # calculate ckas
    # print('ONE', hsic_matrix[:, :, :, :, 1])
    # print('TWO', (torch.sqrt(torch.abs(hsic_matrix[:, :, :, :, 0])) * torch.sqrt(torch.abs(hsic_matrix[:, :, :, :, 2])) + 1e-8))
    cka_matrix = (hsic_matrix[:, :, :, :, 1] + 1e-8) / (torch.sqrt(torch.abs(hsic_matrix[:, :, :, :, 0])) * torch.sqrt(torch.abs(hsic_matrix[:, :, :, :, 2])) + 1e-8)
    
    # get upper triangular indices
    cka_avg = torch.zeros(len(features), len(features), dtype=torch.double, device=device)
    count = 0
    for i in range(mbs):
      for j in range(mbs):
        if j > i:
          cka_avg += cka_matrix[i, j]
          count += 1
    cka_avg /= count
    
    # put back into train if that was the previous state
    if was_training:
      self.train()
    
    if save_path:
      fig, ax = plt.subplots()
      cka_mat = cka_avg.detach().cpu().numpy()
      # print(cka_mat)
      im = ax.imshow(cka_mat, origin='lower', cmap='magma', vmin=0, vmax=1.0)

      # Loop over data dimensions and create text annotations.
      for i in range(cka_mat.shape[0]):
          for j in range(cka_mat.shape[1]):
              try:
                  value = cka_mat[i, j].item()
              except:
                  value = float(cka_mat[i, j])
              text = ax.text(j, i, '%.2f' % float(value),
                          ha="center", va="center", color=("w" if value < 0.2 else "black"), fontsize=12)

      ax.set_xlabel(f"Layer", fontsize=15)
      ax.set_ylabel(f"Layer", fontsize=15)

      if title is not None:
          ax.set_title(f"{title}", fontsize=18)
      # else:
      #     ax.set_title(f"Average CKA of Ensemble", fontsize=18)

      add_colorbar(im)
      plt.tight_layout()

      if save_path is not None:
          plt.savefig(save_path, dpi=300)

      # plt.show()
    
    # return locally optimal parameters
    return cka_avg

  def forward_params(self, x=None):
    raise NotImplementedError('Forward params not defined for the base model generator class. Consider using a sub-class')

  def forward(self, params: Union[torch.Tensor, dict, OrderedDict, DefaultOrderedDict], x: torch.Tensor, sample_params: bool=False, ret_params: bool=False, feature_split: bool=False, **split_args: dict):
    """ Takes the generated/specified parameters and runs through target model also returning model's tracked features

    Args:
        params (Union[torch.Tensor, dict, OrderedDict, DefaultOrderedDict]): the parameters for each parameter layer generated by method forward_params
        x (torch.Tensor): the input to the batched target network 

    Note: if params is a Tensor it is assumed to be a set of codes to be forwarded through forward_params
    otherwise it is assumed to be an already generated parameter set

    Returns:
        tuple: (tracked features as an OrderedDict, final return features from target model)
    """

    # generate parameters if not already specified
    # if isinstance(params, torch.Tensor):
    if sample_params:
      # hacky but it works
      if hasattr(self, 'forward_random_params') and isinstance(params, str):
        params = self.forward_random_params(int(params))
      else:
        sparam = self.sample_params(params, device=x.device)   # handle sampling codes for the parameters
        params = self.forward_params(sparam)
    elif not isinstance(params, OrderedDict):   # assume not already sampled parameters in layer code generator (NOTE THIS DOES NOT WORK FOR OTHERS @TODO only specify in layer code generator)
        params = self.forward_params(params)
    
    # if doing a feature split then specify args as well
    if feature_split:
      res = self.forward_split(params, x, **split_args)
      if ret_params:  # add params to return dictionary
        res['params'] = params
      return res
    else:
      # forward through target model
      outs = self.target(params, x)

      # normal tuple return with params as first return
      if ret_params:
        return params, outs
      return outs
    
  def feature_split(self, feats, pred=None, ood_N=None, skip_empty: bool=True, detach_feats: List[int]=None, split_pred_only: bool=False, already_flat: bool=False):
    """ Splits the features into two groups. Usually used for inlier and outlier points
    
    If ood_N is specified then the last N samples are considered outlier and the rest are inliers. Otherwise, all samples are considered inliers
    
    NOTE: This function will returned a **flattened** view of feature dictionary. Use unflatten from util.collections to get the original structure
    
    Args:
        feats (OrderedDict): the tracked features from the model
        pred (torch.Tensor): the predictions from the model
        ood_N (int): the number of out of distribution samples to use
        skip_empty (bool): if True, will not add any empty features to output. Default is True
        detach_feats (List[int]): a list of indices to detach the features from graph at. Default is None
        split_pred_only (bool): if True, will only split the predictions and not the features. Default is False
        already_flat (bool): if True, will assume the features are already flattened. Default is False
    """
    
    # flatten structure
    if already_flat:
      flat_feats = feats
    else:
      flat_feats = flatten_keys(feats)
    
    # specify feature lists in the layer code order
    features_in = OrderedDict()
    features_ood = OrderedDict() if ood_N is not None and ood_N > 0 else None
    
    # make feature matrices
    for ind, (key, feat) in enumerate(flat_feats.items()):
      # skip any non-tracked features
      if feat is None and skip_empty:
        continue

      # detach features if specified
      if detach_feats is not None and ind in detach_feats:
        feat = feat.detach()
  
      # separate out the ood features
      if features_ood is not None and not split_pred_only:
        feat_ood = feat[:, -ood_N:]
        feat = feat[:, :-ood_N]
        features_ood[key] = feat_ood
      features_in[key] = feat
    
    # define ood predictions
    pred_in = pred
    pred_ood = None
    if pred is not None and ood_N is not None and ood_N > 0:
      pred_in = pred[:, :-ood_N]
      pred_ood = pred[:, -ood_N:]
    
    # return the split
    return features_in, pred_in, features_ood, pred_ood

  def forward_split(self, params: dict, x: torch.Tensor, ood_N: int=0, skip_empty: bool=True, detach_feats: List[int]=None, split_pred_only: bool=False):
    """ Forward through hyper and do the feature and prediction split
    
    NOTE: returns a flattened view of the features. Use unflatten from util.collections to get the original structure
    
    Args:
      hyper (LayerCodeModelGenerator): The hyper model definition (could also be an ensemble)
      params (dict): The generated parameters of the target model 
      x (torch.Tensor): The input data
      ood_N (int, optional): The number of ood samples if any
      skip_empty (bool): if True, will not add any empty features to output. Default is True
      detach_feats (List[int]): a list of indices to detach the features from graph at. Default is None
      split_pred_only (bool): if True, will only split the predictions and not the features. Default is False
    """
    # forward through target models
    all_feats, all_pred = self.forward(params, x, sample_params=False, ret_params=False, feature_split=False)
    
    # flatten features
    all_feats = flatten_keys(all_feats)
    
    # if there are outlier samples then apply the feature split
    # function automatically handles case when there are no ood samples
    feats, pred, ood_feats, ood_pred = self.feature_split(
      feats=all_feats,
      pred=all_pred,
      ood_N=ood_N,
      skip_empty=skip_empty,
      detach_feats=detach_feats,
      split_pred_only=split_pred_only,
      already_flat=True  # optional/saves recalling flatten_keys
    )
      
    return {
      'feats': all_feats,
      'pred': all_pred,
      'feats_ind': feats,  # could be equal to all_feats
      'pred_ind': pred,    # could be equal to all_pred
      'feats_ood': ood_feats,
      'pred_ood': ood_pred
    }
 
  @staticmethod
  def from_config(config: dict):
    """ Create a model generator from a configuration """
    if isinstance(config['target'], (OrderedDict, dict)):
      target = build_gen_module(config['target'])
      config['target'] = target
    return config


@register_generator('layer_generator')
class LayerCodeModelGenerator(ModelGenerator):
  def __init__(self, target: GenModule, code_size: int, default_generators: Dict[GenModule, BaseLayerGenerator], specific_generators: Dict[str, BaseLayerGenerator]=None):
    """ Creates a model generator that works on the principal of layer codes. Thus for each layer definition there is are latent code(s)
    that is assigned a layer generator defined by default_generators and specific_generators

    Args:
        target (GenModule): the target batched model definition with layer definitions
        code_size (int): the latent size of the codes
        default_generators (Dict[GenModule, BaseLayerGenerator]): define a dict that maps a specifc type of layer, for example a Linear, then passes any layer that matches that key with the generator defined as the value. 
        specific_generators (Dict[str, BaseLayerGenerator], optional): same as default_generator but now accepts a specific path for example "0.conv2d" to have a generator defined in value. Defaults to None.
    """
    # get target model definition as described in model generator
    super(LayerCodeModelGenerator, self).__init__(target)

    # specify layer default
    self.default_gen = default_generators
    self.specific_gen = specific_generators
    self.code_size = code_size

    # initialize the layer generator list
    self.total_codes = 0
    self.total_layers = 0
    self.layer_code_num = OrderedDict()
    self.layer_generators = nn.ModuleDict()

    # recursively assign/generate the individual layer generators
    self.assign_layer_generators(self.gen_defined, self.layer_generators, self.layer_code_num, prefix='')

  def assign_layer_generators(self, definitions, generators, code_nums, prefix):
    """ This takes the expected generated parameters defined by the target module
      and creates a generator for each expected parameter/paramter group.
    """
    # recursively walk through all definitions to create the generators
    for name, layer in definitions.items():
      mod_def = definitions[name]

      # append '.' to name if we're in a sub-module 
      if prefix == '':
        full_name = name
      else:
        full_name = f'{prefix}.{name}'
      
      # ensure order by first doing any sub-module
      if isinstance(mod_def, (dict, OrderedDict, nn.ModuleDict)):
        # initialize sub-module module dict and its code numbers
        generators[name] = nn.ModuleDict()
        code_nums[name] = OrderedDict()

        # recursively add layer generators
        self.assign_layer_generators(mod_def, generators[name], code_nums[name], full_name)
      else:
        # see if we can find a specific generator first
        print('Building layer...', full_name)
        if self.specific_gen is not None and full_name in self.specific_gen:
          generators[name] = self.specific_gen[full_name](mod_def)
        elif self.specific_gen is not None and f"{full_name.replace('.self', '')}" in self.specific_gen:
          generators[name] = self.specific_gen[f"{full_name.replace('.self', '')}"](mod_def)
          # print('assigned', full_name, 'comment out line 115 gen/base.py')
        elif mod_def.__class__ in self.default_gen:
          generators[name] = self.default_gen[mod_def.__class__](mod_def)
        else:
          raise ValueError(f'the definition class {mod_def.__class__.__name__} for layer {full_name} does not have a respective generator. Available defaults {[key.__name__ for key in self.default_gen.keys()]}')

        # get the expected number of codes and save the flat offset
        expected = int(generators[name].get_expected_input_size())
        code_nums[name] = expected  # save the number of codes for later use
        self.total_codes += expected
        self.total_layers += 1

  def get_total_codes(self) -> int:
    """ Returns expected total number of codes required for layer generators

    Returns:
        int: the number of total codes required for layer generators
    """
    return self.total_codes

  def get_total_layers(self) -> int:
    """ Returns expected total number of generated layers

    Returns:
        int: number of layers to generate
    """
    return self.total_layers

  def _iter_bfs(self, names=None, generators=None, code_nums=None):
    """ (INTERNAL: use iter_bfs) A (python) generator [too many things are called generators...] on the layer generator ordered dict tree in a way s.t all dependent
          generators appear after.
    """
    for name, gen in generators.items():
      if isinstance(gen, (dict, OrderedDict, nn.ModuleDict)):
        yield from self._iter_bfs(names + [name], generators[name], code_nums[name])
      else:
        yield names + [name], gen, code_nums[name]

  def iter_bfs(self):
    """ A (python) generator [too many things are called generators...] on the layer generator ordered dict tree in a way s.t all dependent
          generators appear after.
    """
    return self._iter_bfs([], self.layer_generators, self.layer_code_num)

  def get_parameter_shapes(self) -> DefaultOrderedDict:
    """ Returns the expected parameter shapes for the layer generators """
    shapes = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      shapes.set_subitems(names, layer_generator.get_parameter_shapes())
    return shapes

  def get_parameter_from_flats(self) -> DefaultOrderedDict:
    """ Returns functions to unflatten parameters from a flat vector """
    fflats = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      fflats.set_subitems(names, layer_generator.from_flat())
    return fflats

  def get_parameter_from_shapes(self) -> DefaultOrderedDict:
    """ Returns list of definitions/indices to unflatten a paramter def """
    fshapes = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      fshapes.set_subitems(names, layer_generator.from_shapes())
    return fshapes

  def forward_params(self, codes):
    """ Handles taking the generated codes (from some other model) and forwarding them through the individual generators

    Args:
        codes (torch.Tensor): A tensor of [B, total codes, code size] values to generate parameters from
    """
    assert len(codes.shape) == 3, 'Invalid number of code dims. Expecting 3'
    assert codes.shape[1] == self.total_codes, f'Expecting a total of {self.total_codes} codes in axes 1. Got instead {codes.shape[1]}'
    assert codes.shape[2] == self.code_size, f'Expecting a code size of {self.code_size} in axes 2. Got instead {codes.shape[2]}'

    # keep track of the current code offset
    code_offset = 0

    # creates an easy way to assign parameters
    params = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      # run through layer generator with expected number of codes
      params.set_subitems(names, layer_generator(codes[:, code_offset: code_offset + layer_num_code, :]))
      
      # move linearly through codes
      code_offset += layer_num_code

    return params

  def forward_random_params(self, batch_size, device='cuda'):
    """ Handles generating a set of random parameters for the layer generators
    Args:
        codes (torch.Tensor): A tensor of [B, total codes, code size] values to generate parameters from
    """
    # creates an easy way to assign parameters
    params = DefaultOrderedDict()
    for names, layer_generator, layer_num_code in self.iter_bfs():
      # run through layer generator with expected number of codes
      params.set_subitems(names, layer_generator.random(batch_size, device=device))

    return params

  @staticmethod
  def from_config(config: dict):
    """ Create a model generator from a configuration """
    config = ModelGenerator.from_config(config)
    
    default_generators = {}
    for key, val in config['default_generators'].items():
      default_generators[AVAILABLE_GEN_MODULES[key]] = partial(build_layer_generator, val)
    
    specific_generators = None
    if 'specific_generators' in config:
      specific_generators = {}
      for key, val in config['specific_generators'].items():
        specific_generators[AVAILABLE_GEN_MODULES[key]] = partial(build_layer_generator, val)
    
    config['default_generators'] = default_generators
    config['specific_generators'] = specific_generators
    return config


@register_generator('generated_layer_generator')
class GeneratedLayerCodeModelGenerator(ModelGenerator):
  def __init__(self, latent_size: int, layer_code_generator: LayerCodeModelGenerator, code_generator: nn.Module=None):
    """ A base class that can be sub-classes for any generator that creates codes for a layer code generator (ik a lot of generators... need better words)

    Args:
        latent_size (int): the expected input latent code size for the code generator
        layer_code_generator (LayerCodeModelGenerator): an expected layer code generator to feed generated codes through
        code_generator (nn.Module): a module that takes a batch of <latent_size> input codes, create codes for the layer_code_generator, and passes them through. This is optional as you could also sub-class and override the forward method. Default is None.
    """
    # get target model definition as described in model generator
    super(GeneratedLayerCodeModelGenerator, self).__init__(layer_code_generator.get_target())

    # specify layer default
    self.latent_size = latent_size
    self.layer_gen = layer_code_generator
    self.code_gen = code_generator

  def set_code_generator(self, code_generator: nn.Module):
    """ Sets the generator to be used for the code generator

    Args:
        code_generator (nn.Module): a module that accepts some code and generates multiple/mixed codes
    """
    self.code_gen = code_generator

  def get_parameter_shapes(self) -> OrderedDict:
    """ Returns the expected parameter shapes for the layer generators """
    return self.layer_gen.get_parameter_shapes()

  def get_parameter_from_flats(self) -> OrderedDict:
    """ Returns functions to unflatten parameters from a flat vector """
    return self.layer_gen.get_parameter_from_flats()

  def get_parameter_from_shapes(self) -> OrderedDict:
    """ Returns functions that define parameter unflattening """
    return self.layer_gen.get_parameter_from_shapes()

  def forward_layer_params(self, codes):
    """ Identity to the LayerCodeModelGenerator.forward_params(codes) with a simple reshape to ensure any layer code generator gets the right input """
    # in some cases generators, like an MLP, flatten the codes. So we'll view it as the layer_code generator expects it to
    if codes.shape[1] != self.layer_gen.total_codes:
      codes = codes.view(codes.shape[0], self.layer_gen.total_codes, self.layer_gen.code_size)
    
    if codes.shape[2] != self.layer_gen.code_size:
      raise ValueError('Invalid code size passed to forward_layers')
    return self.layer_gen.forward_params(codes)
  
  def forward_params(self, x, device='cuda'):
    """ Forward input sample of latent_size through the code generator and then the layer codes to generate the parameters """
    # handle case of randomly sampling weights
    if isinstance(x, str):
      return self.layer_gen.forward_random_params(int(x), device=device)
    
    # other case of feeding through generator
    if self.code_gen is None:
      raise NotImplementedError('The base class requires a code_generator module. Please set it using set_code_generator or sub-class and change the forward method in hyper/generators/base.py')
    codes = self.code_gen(x)
    return self.forward_layer_params(codes)

  def sample_random_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    return str(size)  # hacky way right now but works


  def forward(self, params: Union[torch.Tensor, dict, OrderedDict, DefaultOrderedDict], x: torch.Tensor, sample_params: bool=False, ret_params: bool=False, feature_split: bool=False, **split_args: dict):
    """ Takes the generated/specified parameters and runs through target model also returning model's tracked features

    Args:
        params (Union[torch.Tensor, dict, OrderedDict, DefaultOrderedDict]): the parameters for each parameter layer generated by method forward_params
        x (torch.Tensor): the input to the batched target network 

    Note: if params is a Tensor it is assumed to be a set of codes to be forwarded through forward_params
    otherwise it is assumed to be an already generated parameter set

    Returns:
        tuple: (tracked features as an OrderedDict, final return features from target model)
    """

    # generate parameters if not already specified
    # if isinstance(params, torch.Tensor):
    if sample_params or isinstance(params, int):
      # hacky but it works
      if hasattr(self, 'forward_random_params') and isinstance(params, str):
        params = self.forward_random_params(int(params))
      else:
        sparam = self.sample_params(params, device=x.device)   # handle sampling codes for the parameters
        params = self.forward_params(sparam)
    elif not isinstance(params, OrderedDict):   # assume not already sampled parameters in layer code generator (NOTE THIS DOES NOT WORK FOR OTHERS @TODO only specify in layer code generator)
        params = self.forward_params(params)
    
    # if doing a feature split then specify args as well
    if feature_split:
      res = self.forward_split(params, x, **split_args)
      if ret_params:  # add params to return dictionary
        res['params'] = params
      return res
    else:
      # forward through target model
      outs = self.layer_gen.target(params, x)

      # normal tuple return with params as first return
      if ret_params:
        return params, outs
      return outs

  @staticmethod
  def from_config(config: dict, code_gen: nn.Module=None):
    """ Create a model generator from a configuration """
    config['layer_code_generator'] = build_generator(config['layer_code_generator'])
    
    if code_gen is not None:
      config['code_generator'] = code_gen
    return config
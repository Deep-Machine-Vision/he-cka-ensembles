""" Functions useful in calculating kernels on models including for SVGD and HE """
from typing import List, Union, Optional
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from hyper.util.collections import flatten_keys, unflatten_keys
from hyper.diversity import pairwise_cossim, hyperspherical_energy, unit_vector_rows
from hyper.diversity.kernels import EMA, KERN_PARAMS, MAPS, batched_gram, center_gram
from hyper.diversity.layer_weighting import LayerWeighting, build_weighting


HE_VALUE_CACHE = {}  # cache he min/max values
AVAILABLE_KERNELS = {}

def register_model_kernel(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_KERNELS[name] = cls
    return cls
  return decorator


def build_model_kernel(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_KERNELS[name]
  config = cls.from_config(config)
  return cls(**config)


class BaseModelKernel(nn.Module):
  def __init__(self) -> None:
    """ Base class for model kernels """
    super(BaseModelKernel, self).__init__()
    pass

  def forward(self, params, feats, pred, track, layer=None):
    pass
  
  @staticmethod
  def from_config(config: dict):
    """ Create a kernel from a configuration dictionary """
    return config


@register_model_kernel("feature")
class FeatureKernel(BaseModelKernel):
  def __init__(self, kernel: str, detach_diag: bool=False, readd_diag: bool=True, params: dict=None, learnable: bool=False, *args, **kwargs) -> None:
    """ Kernel supporting batched features
    
    Args:
      kernel (str): The name of kernel to use. See kernels.py for available kernels
      detach_diag (bool, optional): Whether to detach the diagonal of the kernel matrix from the graph
      readd_diag (bool, optional): Whether to readd the diagonal of the kernel matrix. If detach_diag is False this does nothing. If detach_diag is True and if this is False, the diagonal will be zero
      params (dict): The parameters for the kernel (usually like epsilon or param for RBF)
      *args: Additional arguments for the kernel (see batched_gram)
      **kwargs: Additional keyword arguments for the kernel (see batched_gram)
    """
    super(FeatureKernel, self).__init__()
    self.kernel = kernel
    self.detach_diag = detach_diag
    self.readd = readd_diag
    self.args = args
    self.kwargs = kwargs
    
    assert self.kernel in MAPS, "Kernel {} not found. See kernels.py for available kernels".format(self.kernel)
    
    # now we keep track of other variables here
    self.dparams = KERN_PARAMS.get(self.kernel, {})
    if params is not None:
      self.dparams.update(params)  # update user specified parameters
    self.params = nn.ParameterDict(OrderedDict((k, nn.Parameter(torch.tensor(float(v)), requires_grad=learnable)) for k, v in self.dparams.items()))
    # self.median_ma = EMA(0.99)
    
  def batched_kernel(self, feats: torch.Tensor, flat_second: bool=True, grad_vars: Optional[torch.Tensor]=None):
    # flatten the features
    if feats.ndim < (3 if flat_second else 2):
      raise RuntimeError("Features must be have at least 2/3 dimensions (depending on flat second): (model bs, bs, dim1) or (model bs, bs). See model_kernels.py")
    
    # flatten along first or second dimension
    if flat_second:
      mbs, bs = feats.shape[:2]
      feats = feats.reshape(mbs, bs, -1)  # flatten features
    else:
      mbs = feats.shape[0]
      feats = feats.reshape(mbs, -1)  # flatten features
    
    # fix to single batch item if not batched already
    if feats.ndim == 2:
      feats = feats.unsqueeze(0)
      single = True
    else:
      single = False
    
    gram = batched_gram(
      X=feats, Y=None,  # some impl are faster with Y=None
      kernel=self.kernel,
      center=False,
      detach_diag=self.detach_diag,
      readd=self.readd,
      *self.args,
      **self.params,
      **self.kwargs
    )
    
    # if single batch item then remove the batch dimension
    if single:
      gram = gram.squeeze(0)
    
    # if including the kernel gradient then return that as well
    if grad_vars is not None:
      return gram, torch.autograd.grad(gram.sum(), grad_vars, retain_graph=True)[0]
  
    return gram

  def forward(self, params, feats, preds, track, layer=None):
    """ Assumes feats have already been flattened, but can be flattened "automatically" """
    return self.batched_kernel(feats)


@register_model_kernel("function")
class FunctionKernel(FeatureKernel):
  def __init__(self, kernel: str, detach_diag: bool=False, readd_diag: bool=True, params: dict=None, *args, **kwargs) -> None:
    """ Kernel supporting batched function space kernels. Just the output of the model
    
    Args:
      kernel (str): The name of kernel to use. See kernels.py for available kernels
      detach_diag (bool, optional): Whether to detach the diagonal of the kernel matrix from the graph
      readd_diag (bool, optional): Whether to readd the diagonal of the kernel matrix. If detach_diag is False this does nothing. If detach_diag is True and if this is False, the diagonal will be zero
      params (dict): The parameters for the kernel (usually like epsilon or param for RBF)
      *args: Additional arguments for the kernel (see batched_gram)
      **kwargs: Additional keyword arguments for the kernel (see batched_gram)
    """
    super(FunctionKernel, self).__init__(kernel, detach_diag, readd_diag, params, *args, **kwargs)

  def forward(self, params, feats, preds, track, layer=None):
    """ Assumes feats have already been flattened """
    return self.batched_kernel(preds, flat_second=False)


@register_model_kernel("weight")
class WeightKernel(FeatureKernel):
  def __init__(self, kernel: str, detach_diag: bool=False, readd_diag: bool=True, params: dict=None, *args, **kwargs) -> None:
    """ Kernel supporting batched function space kernels. Just the output of the model
    
    Args:
      kernel (str): The name of kernel to use. See kernels.py for available kernels
      detach_diag (bool, optional): Whether to detach the diagonal of the kernel matrix from the graph
      readd_diag (bool, optional): Whether to readd the diagonal of the kernel matrix. If detach_diag is False this does nothing. If detach_diag is True and if this is False, the diagonal will be zero
      params (dict): The parameters for the kernel (usually like epsilon or param for RBF)
      *args: Additional arguments for the kernel (see batched_gram)
      **kwargs: Additional keyword arguments for the kernel (see batched_gram)
    """
    super(WeightKernel, self).__init__(kernel, detach_diag, readd_diag, params, *args, **kwargs)

  @staticmethod
  def flatten_weights(params):
    # do not include shared target parameters
    flat_par_params, par_empty_params = flatten_keys(params, include_empty=True)
    par_params = list(flat_par_params.values())
    
    if len(par_params) == 0:
      raise ValueError("No parameters found in the model")
    
    # get model batch size
    model_bs = par_params[0].shape[0]  # typically parameters are batched via first dimension
    flats = [p.reshape(model_bs, -1) for p in par_params]
    pset = torch.concat(flats, dim=1)
    
    return {
      'params': flat_par_params,
      'empty': par_empty_params,
      'flat_params': pset,
      'model_bs': model_bs,
      '__flat__': True
    }

  def forward(self, params, feats, preds, track, layer=None):
    """ Assumes feats have already been flattened """

    # assumes params have been flattened but we can try to flatten ourselves otherwise
    if isinstance(params, (OrderedDict, dict)) and '__flat__' not in params:
      params = self.flatten_weights(params)

    return self.batched_kernel(params['flat_params'], flat_second=False)


@register_model_kernel("cka")
class CKAModelKernel(BaseModelKernel):
  def __init__(self, feature_kernel: FeatureKernel, eps: float=1e-6, center: bool=True, reduction: str='off_diag', abs_vals: bool=False, detach_right: bool=False, learnable: bool=False) -> None:
    """ Kernel based on centered kernel alignment
    
    Args:
      feature_kernel (FeatureKernel): The feature kernel to use for the CKA kernel
      eps (float, optional): The epsilon value for the kernel
      center (bool, optional): Whether to center the kernel
      reduction (str, optional): How to reduce the kernel matrix. Options are 'off_diag' (default) or 'none'/None 
      abs_vals (bool, optional): Whether to take the absolute value of the kernel matrix
      detach_right (bool, optional): Whether to detach the right side of the kernel matrix when calculating pairwise cossim
      args: Additional arguments for the kernel (see batched_gram)
      kwargs: Additional keyword arguments for the kernel (see batched_gram)
    """
    super(CKAModelKernel, self).__init__()
    self.feature_kernel = feature_kernel
    self.center = center
    self.abs_vals = abs_vals
    self.reduction = reduction
    self.detach_right = detach_right
    self.eps = nn.Parameter(torch.tensor(eps), requires_grad=learnable)

  def forward(self, params, feats, preds, track, layer=None):
    """ Assumes feats have already been flattened """
    gram = self.feature_kernel.batched_kernel(feats)
    
    if self.center:
      gram = center_gram(gram)
    
    # track gram matrix if we want to estimate CKAs later
    track['gram'] = gram
    
    # calculate pairwise cosine similarity
    pcosim = pairwise_cossim(
      grams=gram,
      eps=self.eps,
      reduction=self.reduction,
      detach_right=self.detach_right
    )
    
    return pcosim

  @staticmethod
  def from_config(config: dict):
    """ Create a kernel from a configuration dictionary """
    config = BaseModelKernel.from_config(config)
    config['feature_kernel'] = build_model_kernel(config['feature_kernel'])
    return config


@register_model_kernel("he")
class HEModelKernel(BaseModelKernel):
  def __init__(self, feature_kernel: FeatureKernel, he_s: float=2.0, eps: float=1e-6, arc_eps=1e-3, center: bool=True, reduction: str='off_diag', abs_vals: bool=False, detach_right: bool=False, use_exp: bool=False, normalize: bool=False, learnable: bool=False) -> None:
    """ Kernel based on hyperspherical energy based centered kernel alignment
    
    Args:
      feature_kernel (FeatureKernel): The feature kernel to use for the HE kernel
      he_s (float, optional): The s parameter for the HE kernel
      eps (float, optional): The epsilon value for the kernel
      arc_eps (float, optional): The epsilon value for the arccosine function
      center (bool, optional): Whether to center the kernel
      reduction (str, optional): How to reduce the kernel matrix. Options are 'off_diag' (default) or 'none'/None 
      abs_vals (bool, optional): Whether to take the absolute value of the kernel matrix
      detach_right (bool, optional): Whether to detach the right side of the kernel matrix when calculating pairwise cossim
      use_exp (bool, optional): Whether to use the exponential version of the hyperspherical energy
      args: Additional arguments for the kernel (see batched_gram)
      kwargs: Additional keyword arguments for the kernel (see batched_gram)
    """
    super(HEModelKernel, self).__init__()
    self.feature_kernel = feature_kernel
    self.center = center
    self.abs_vals = abs_vals
    self.reduction = reduction
    self.detach_right = detach_right
    self.he_s = he_s  # nn.Parameter(torch.tensor(he_s), requires_grad=learnable)
    self.eps = nn.Parameter(torch.tensor(eps), requires_grad=learnable)
    self.arc_eps = nn.Parameter(torch.tensor(arc_eps), requires_grad=learnable)
    self.use_exp = use_exp
    self.normalize_he = normalize
    self.min_he_energy = 0.0
    self.max_he_energy = 1.0

  def get_smoothing(self, detach=False, asfloat=False):
    """ Return the smoothing terms """
    if detach:
      arc, eps = torch.abs(self.arc_eps).detach(), torch.abs(self.eps).detach()
    else:
      arc, eps = torch.abs(self.arc_eps), torch.abs(self.eps)
    
    if asfloat:
      return arc.item(), eps.item()
    return arc, eps

  @torch.enable_grad()
  def precompute(self, model_bs, bs, device='cuda'):
    # if not self.training:  # only precompute during training
    #   return self.min_he_energy, self.max_he_energy
    
    # we need to presolve min hyperspherical energy
    print('Precomputing min/max hyperspherical energy')
    
    # ensure batched gram calculation is sim to before
    fake_grams = torch.empty(model_bs, bs, bs, requires_grad=True, dtype=torch.float32, device=device)
    torch.nn.init.normal_(fake_grams)
      
    # maximize energy for each method
    # now minimize hyperspherical energy
    fake_optim = torch.optim.Adam([fake_grams], lr=0.05)
    fake_sched = torch.optim.lr_scheduler.StepLR(fake_optim, step_size=10, gamma=0.1)
    arc, eps = self.get_smoothing(detach=True, asfloat=True)
    
    for i in range(100):
      fake_optim.zero_grad()

      with torch.no_grad():
        fake_grams.data = unit_vector_rows(fake_grams.data, eps=0.0)

      # test the energy
      energy = hyperspherical_energy(
        fake_grams,
        half_space=False,
        s=self.he_s,
        arc_eps=arc,
        eps=eps,
        use_exp=self.use_exp,
        remove_diag=self.reduction == 'off_diag',
        detach_right=self.detach_right,
        abs_vals=self.abs_vals
      )
      energy = energy.mean()
      energy.backward()
      fake_optim.step()
      fake_sched.step()
    self.min_he_energy = energy.min().item()
    
    # now calculate "max" via a projected gaussian far away
    # simulating grams really close together
    with torch.no_grad():
      torch.nn.init.normal_(fake_grams, mean=1000.0, std=0.1)
      max_energy = hyperspherical_energy(
        fake_grams,
        half_space=False,
        s=self.he_s,
        arc_eps=arc,
        eps=eps,
        use_exp=self.use_exp,
        remove_diag=self.reduction == 'off_diag',
        detach_right=self.detach_right,
        abs_vals=self.abs_vals
      )
      self.max_he_energy = max_energy.max().item()  # approximate "max"
    
    print('Completed hyperspherical energy max and min calc')
    print('FINAL ITER', i, 'MIN ENERGY', self.min_he_energy, 'MAX ENERGY', self.max_he_energy)
    
    # store into cache
    if self.normalize_he:
      HE_VALUE_CACHE[(model_bs, bs, arc, eps)] = (self.min_he_energy, self.max_he_energy)
    return self.min_he_energy, self.max_he_energy

  def forward(self, params, feats, preds, track, layer=None):
    """ Assumes feats have already been flattened """
    gram = self.feature_kernel.batched_kernel(feats)
    
    if self.center:
      gram = center_gram(gram)
    
    # track gram matrix if we want to estimate CKAs later
    track['gram'] = gram
    
    # # ensure the kernel matrix is positive
    # if self.abs_vals:
    #   gram = torch.abs(gram)

    arc, eps = self.get_smoothing()
    energy = hyperspherical_energy(
      gram,
      half_space=False,
      s=self.he_s,
      arc_eps=arc,
      eps=eps,
      use_exp=self.use_exp,
      remove_diag=self.reduction == 'off_diag',
      detach_right=self.detach_right,
      abs_vals=self.abs_vals
    )
    
    # potentially normalize he
    # use values from cache as to not recompute min/max
    if self.normalize_he:
      min_he, max_he = HE_VALUE_CACHE.get((gram.shape[0], gram.shape[1], arc.item(), eps.item()), (None, None))
      if min_he is None or max_he is None:
        # run precomputation for the following HE set
        min_he, max_he = self.precompute(gram.shape[0], gram.shape[1], device=gram.device)
      
      # normalize 0-1 using min and max energy
      energy = ((energy - min_he) / (max_he - min_he))
    
    return energy

  @staticmethod
  def from_config(config: dict):
    """ Create a kernel from a configuration dictionary """
    config = BaseModelKernel.from_config(config)
    config['feature_kernel'] = build_model_kernel(config['feature_kernel'])
    return config


@register_model_kernel("model_layers")
class ModelLayersKernel(BaseModelKernel):
  def __init__(self, layer_kernel: Union[BaseModelKernel, List[BaseModelKernel]], layer_weighting: LayerWeighting, reduction: str='mean') -> None:
    """ Kernel based on the layers of a model and kernel weighted based on a LayerWeighting scheme
    
    Args:
      layer_kernel (BaseModelKernel): The kernel to use for each layer. Possibly a list of kernels for each layer
      layer_weighting (LayerWeighting): The weighting scheme for the layers
      reduction (str, optional): How to reduce the kernel matrix. Options are 'mean' (default), 'sum' or 'none'/None. This is a reduction on each layer then weighted across all layers using the specified weighting scheme 
    
    """
    super(ModelLayersKernel, self).__init__()
    
    # we can use the same layer kernel for all layers or different ones
    self.different_kernels = isinstance(layer_kernel, list)
    if self.different_kernels:
      self.layer_kernel = nn.ModuleList(layer_kernel)
    else:
      self.layer_kernel = layer_kernel
    self.layer_weighting = layer_weighting
    self.reduction = reduction
  
  def apply_reduction(self, model_grams):
    if self.reduction == 'mean':
      return model_grams.mean()
    elif self.reduction == 'sum':
      return model_grams.sum()
    return model_grams
  
  def forward(self, params, feats, pred, track):
    """ Assumes feats have already been flattened """
    total = len(feats)
    if total == 0:
      return None
    
    self.layer_weighting.reset_total()  # save the layer weighting for the current total number of feat
    self.layer_weighting.save_total(total)  # reset the total variable (useful for learned layer weighting)
    
    # track keys
    feat_layers = list(feats.values())
    layer_track = [OrderedDict() for _ in range(total)]

    # same or different kernels
    if self.different_kernels:
      kernel = self.layer_kernel[0](params, feat_layers[0], pred, layer_track[0])
    else:
      kernel = self.layer_kernel(params, feat_layers[0], pred, layer_track[0])
    
    # get the weights for each layer
    gram = self.layer_weighting.get_weight(0, total) * self.apply_reduction(kernel)
    index = 1
    while index < total:
      if self.different_kernels:
        kernel = self.layer_kernel[index](params, feat_layers[index], pred, layer_track[index])
      else:
        kernel = self.layer_kernel(params, feat_layers[index], pred, layer_track[index])
      gram = gram + self.layer_weighting.get_weight(index, total) * self.apply_reduction(kernel)
      index += 1
    
    track['layer_track'] = layer_track
    
    return gram

  @staticmethod
  def from_config(config: dict):
    """ Create a kernel from a configuration dictionary """
    config = BaseModelKernel.from_config(config)
    if isinstance(config['layer_kernel'], list):
      config['layer_kernel'] = [build_model_kernel(k) for k in config['layer_kernel']]
    else:
      config['layer_kernel'] = build_model_kernel(config['layer_kernel'])
    
    # build the weighting scheme
    config['layer_weighting'] = build_weighting(config['layer_weighting'])
    return config

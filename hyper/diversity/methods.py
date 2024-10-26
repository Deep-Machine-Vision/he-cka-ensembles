""" Various diversity approaches like SVGD and HE 

Specify any custom diversity/repuslive particle methods here.
"""
from typing import List, Union, Dict
from collections import OrderedDict
import traceback

import torch
import torch.nn as nn

from hyper.util.collections import flatten_keys, unflatten_keys
from hyper.diversity.model_kernels import BaseModelKernel, FunctionKernel, ModelLayersKernel, WeightKernel, build_model_kernel
from hyper.diversity.losses import LossFunction, build_loss
from hyper.diversity.ssge import SpectralSteinEstimator
import copy

AVAILABLE_METHODS = {}

def register_method(name: str):
  """ Decorator to register a method """
  def decorator(cls):
    AVAILABLE_METHODS[name] = cls
    return cls
  return decorator


def build_method(config: dict):
  """ Builds a method """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_METHODS[name]
  config = cls.from_config(config)
  return cls(**config)


@register_method("basic")
class ParticleMethods(nn.Module):
  def __init__(self, num: int, model_kernel: BaseModelKernel=None, ind_loss: LossFunction=None, ood_loss: LossFunction=None, beta_ind: float=1.0, beta_ood: float=1.0, l2_reg: float=None, learnable_beta: bool=False) -> None:
    """ Class to hold particle methods """
    super(ParticleMethods, self).__init__()

    self.num = num
    self.ind_loss = ind_loss
    self.ood_loss = ood_loss
    self.model_kernel = model_kernel
    self.l2_reg = (l2_reg / 2.0) if l2_reg is not None else None
    
    # weighting between ind and ood losses
    self.beta_ind = nn.Parameter(torch.tensor(beta_ind, dtype=torch.float32), requires_grad=learnable_beta)
    self.beta_ood = nn.Parameter(torch.tensor(beta_ood, dtype=torch.float32), requires_grad=learnable_beta)

  def grads(self, var, grad_var, retain_graph=True, create_graph=False, **kwargs):
    """ Calculates gradients for kernel
    
    NOTE: huge caveat being this will only calculate grad within the same process. DDP must be independent then
    """
    return torch.autograd.grad(var.sum(), grad_var, retain_graph=retain_graph, create_graph=create_graph, **kwargs)

  def batch_start(self, trainer, X, Y):
    """ Handle the start of the batch/any special preprocessing """
    trainer.optim_hyper.zero_grad()
    trainer.optim_method.zero_grad()

  def batch_end(self, trainer, X, Y):
    """ Handle end of the batch/optim stepping or clipping desired """

    # use typical clipping
    try:
      if trainer.clip is not None:
        torch.nn.utils.clip_grad.clip_grad_norm_(trainer.hyper.parameters(), trainer.clip, norm_type=2, error_if_nonfinite=True)
      
      # take step on success
      trainer.optim_hyper.step()
      trainer.optim_method.step()
    except RuntimeError as err:
      traceback.print_exc()
      print(f'Clip error! {str(err)}')

  def forward(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor, ignore_ood_loss: bool=False, step: int=0):
    """ Forward pass for the method. Basic is just forwarding through both losses """
    track['ind_loss'] = self.ind_loss(self.num, x, track['pred_ind'], gt) if self.ind_loss is not None else None
    track['loss'] = torch.abs(self.beta_ind) * track['ind_loss']
    
    # in some cases like learned rollouts we prefer to not include a learned ood loss
    if not ignore_ood_loss:
      if self.ood_loss is not None and track['pred_ood'] is None:
        raise RuntimeError("OOD loss was defined but no OOD predictions were passed")

      if self.ood_loss is not None:
        # include ood loss if needed
        track = self.forward_ood(trainer, params, shared_params, track, x, gt)
    
    # add l2 reg loss
    if self.l2_reg is not None:
      flat_params = flatten_keys(params)
      l2_loss = 0.0
      for pname, pvalue in flat_params.items():
        if 'affine' in pname or 'bias' in pname or 'skip' in pname:  # ignore shared/affine/norm parameters
          continue
        
        # add to l2 loss on each model
        l2_loss += torch.square(pvalue).view(pvalue.shape[0], -1).mean(dim=1).mean()
      track['loss'] = track['loss'] + self.l2_reg * l2_loss
    
    return track 

  def forward_ood(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor):
    """ Forward pass for the method. However only adds ood loss """
    track['ood_loss'] = self.ood_loss(self.num, x, track['pred_ood'], gt)
    ood_loss = torch.abs(self.beta_ood) * track['ood_loss']
    if 'loss' in track:
      track['loss'] = track['loss'] + ood_loss
    else:
      track['loss'] = ood_loss 
    return track

  @torch.no_grad()
  def test(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor):
    """ Calculates the loss for the in distribution samples
    
    Args:
      params (OrderedDict): The parameters of the target models (batched)
      shared_params (list): The shared parameters, a list of them, of the target models
      feats (OrderedDict): The features of the target models (batched)
      track (torch.Tensor): The features/predictions of the target models
      x (torch.Tensor): The input data (could be used for internal loss)
      gt (torch.Tensor): The ground truth labels/outputs
      retain_graph (bool, optional): Whether to retain the graph during backward
      create_graph (bool, optional): Whether to create a graph during backward
    """
    return track['loss'], {}

  def backward(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor, retain_graph: bool=False, create_graph: bool=False, inputs=None):
    """ Update the particles using a method. Assuming driving force is already calculated
    
    Args:
      params (OrderedDict): The parameters of the target models (batched)
      shared_params (list): The shared parameters, a list of them, of the target models
      track (torch.Tensor): The features/predictions of the target models
      x (torch.Tensor): The input data (could be used for internal loss)
      gt (torch.Tensor): The ground truth labels/outputs
      retain_graph (bool, optional): Whether to retain the graph during backward
      create_graph (bool, optional): Whether to create a graph during backward
    """
    track['loss'].backward(retain_graph=retain_graph, create_graph=create_graph, inputs=inputs)
    return track['loss'], {}

  @staticmethod
  def from_config(config: dict):
    """ Create a method from a configuration dictionary """
    config['model_kernel'] = build_model_kernel(config['model_kernel']) if 'model_kernel' in config else None
    
    if 'ind_loss' in config:
      config['ind_loss'] = build_loss(config['ind_loss'])
    
    if 'ood_loss' in config:
      config['ood_loss'] = build_loss(config['ood_loss'])
    return config


@register_method("svgd")
class SVGD(ParticleMethods):
  def __init__(self, num: int, model_kernel: Union[ModelLayersKernel, WeightKernel], ind_loss: LossFunction, kde: bool=False) -> None:
    """ Weight Space Stein Variational Gradient Descent """
    super(SVGD, self).__init__(num, model_kernel, ind_loss)
    
    if self.ood_loss is not None:
      raise ValueError("OOD loss is not supported for SVGD")
    
    self.weight_space = isinstance(model_kernel, WeightKernel)
    self.kde = kde
  
  def backward(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor, retain_graph: bool=False, create_graph: bool=False, inputs=None):
    """ Update the particles using SVGD. Assuming driving force is already calculated
    
    Args:
      params (OrderedDict): The parameters of the target models (batched)
      shared_params (list): The shared parameters, a list of them, of the target models
      feats (OrderedDict): The features of the target models (batched)
      track (torch.Tensor): The features/predictions of the target models
      x (torch.Tensor): The input data (could be used for internal loss)
      gt (torch.Tensor): The ground truth labels/outputs
      retain_graph (bool, optional): Whether to retain the graph during backward
      create_graph (bool, optional): Whether to create a graph during backward
    """
    
    # flatten parameters
    flat_params = flatten_keys(params)
    all_flat_params = list(flat_params.values()) + shared_params
    
    # get in distribution samples
    loss = track['ind_loss']  # get unweighted ind loss
    pred = track['pred_ind']
    feats = track['feats_ind']
    
    # this is driving force gradient
    drive_grads = self.grads(-loss, grad_var=all_flat_params, retain_graph=True, create_graph=create_graph, allow_unused=True)

    # weight kernel is special since we need to flatten the weights
    # into a single vector before passing into the kernel
    if self.weight_space:
      param_data = WeightKernel.flatten_weights(params)
      kern = self.model_kernel(param_data, feats, pred, track)
    else:
      kern = self.model_kernel(flat_params, feats, pred, track)
    
    # calculate kernel gradient
    kern_grads = self.grads(kern, grad_var=all_flat_params, retain_graph=True, create_graph=create_graph, allow_unused=True)
    
    # update parameter gradients then backprop further
    for index, (p, grad_drive, grad_kern) in enumerate(zip(all_flat_params, drive_grads, kern_grads)):
      if grad_drive is not None:
        
        if self.kde:
          grad = (grad_drive - (grad_kern / kern.sum(1, keepdim=True)))
        else:
          grad = (torch.matmul(kern, grad_drive) - grad_kern) / self.num
        p.backward(inputs=inputs, gradient=-grad, retain_graph=True, create_graph=create_graph)
    
    return loss, {}


@register_method("function_svgd")
class FunctionSVGD(ParticleMethods):
  def __init__(self, num: int, model_kernel: FunctionKernel, ind_loss: LossFunction, exp_std: float=1.0, eta: float=0.05, kde: bool=False) -> None:
    """ Weight Space Stein Variational Gradient Descent
    
    Based off of implementation here https://github.com/ratschlab/repulsive_ensembles/blob/master/methods
    """
    super(FunctionSVGD, self).__init__(num, model_kernel, ind_loss)
    
    self.ssge = SpectralSteinEstimator(eta)
    self.exp_std = exp_std
    self.kde = kde
    
    if not isinstance(model_kernel, FunctionKernel):
      raise ValueError("Weight kernels/feature kernels are not supported for function SVGD. Use normal SVGD for that")
  
  def backward(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor, retain_graph: bool=False, create_graph: bool=False, inputs=None):
    """ Update the particles using SVGD. Assuming driving force is already calculated
    
    Args:
      params (OrderedDict): The parameters of the target models (batched)
      shared_params (list): The shared parameters, a list of them, of the target models
      feats (OrderedDict): The features of the target models (batched)
      track (torch.Tensor): The features/predictions of the target models
      x (torch.Tensor): The input data (could be used for internal loss)
      gt (torch.Tensor): The ground truth labels/outputs
      retain_graph (bool, optional): Whether to retain the graph during backward
      create_graph (bool, optional): Whether to create a graph during backward
    """
    
    # flatten parameters
    flat_params = flatten_keys(params)
    all_flat_params = list(flat_params.values()) + shared_params
    
    # get in distribution samples
    loss = track['ind_loss']  # get unweighted ind loss
    pred = track['pred_ind']
    feats = track['feats_ind']
    
    # retain gradient for predictions
    pred.retain_grad()
    
    # scoring function gradient
    pred_grad = self.grads(-loss/(self.exp_std**2), grad_var=pred, retain_graph=True, create_graph=create_graph)[0]
    score_func = pred_grad.reshape(self.num, -1)
    
    # gradient functional prior (normal prior)
    module = trainer.hyper.module if trainer.ddp else trainer.hyper
    sparam_prior = module.sample_random_params(self.num, device=pred.device)
    params_prior = module.forward_params(sparam_prior, device=pred.device)
    
    # forward pass through the hypernetwork
    if trainer.ood_N > 0:
      X_ind = x[:-trainer.ood_N]
    else:
      X_ind = x
    _, pred_prior = module.forward(params_prior, X_ind)
    flat_pred_prior = pred_prior.reshape(self.num, -1)
    
    # gradient prior
    grad_prior = self.ssge.compute_score_gradients(pred.reshape(self.num, -1), flat_pred_prior)
    
    # apply kernel on the driving force
    kern = self.model_kernel(flat_params, feats, pred, track)
    
    # calculate kernel gradient (zero out pred gradients)
    kern_grads = self.grads(kern, grad_var=pred, retain_graph=True, create_graph=create_graph, allow_unused=True)[0]
    flat_kern_grads = kern_grads.reshape(self.num, -1)
    
    # use KDE or not
    if self.kde:
      grad = (score_func + grad_prior) - ((flat_kern_grads / kern.sum(1, keepdim=True))) / self.num
    else:
      grad = torch.matmul(kern, score_func + grad_prior) - flat_kern_grads / self.num
    
    # calculate jacob to weight space and run backward
    pred.backward(gradient=-grad.reshape(pred.shape), retain_graph=retain_graph, create_graph=create_graph)
    return loss, {}


@register_method("repulsive_kernel")
class RepulsiveKernel(ParticleMethods):
  def __init__(self, num: int, model_kernel: ModelLayersKernel, ind_loss: LossFunction, ood_loss: LossFunction=None, gamma: float=1.0, model_kernel_ood: ModelLayersKernel=None, beta_ind: float=1.0, beta_ood: float=1.0, gamma_ood: float=1.0, warmup: int=0, l2_reg: float=None, learnable_beta: bool=False, learnable: bool=False) -> None:
    """ Used for CKA or HE based feature repulsion """
    super(RepulsiveKernel, self).__init__(num, model_kernel, ind_loss, ood_loss, beta_ind=beta_ind, l2_reg=l2_reg, beta_ood=beta_ood, learnable_beta=learnable_beta)
    
    # specify gamma/repulsive force
    self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=learnable)
    
    # if using ood
    self.use_ood = ood_loss is not None or model_kernel_ood is not None
    if self.use_ood:
      self.gamma_ood = nn.Parameter(torch.tensor(gamma_ood, dtype=torch.float32), requires_grad=learnable)  
    self.model_kernel_ood = model_kernel_ood
    self.warmup = warmup
      
    if isinstance(model_kernel, WeightKernel):
      raise ValueError("Weight kernels are not supported for feature repulsion")
  
  def forward(self, trainer, params: dict, shared_params: list, track: dict, x: torch.Tensor, gt: torch.Tensor, ignore_ood_loss: bool=False, step: int=0):
    """ Forward pass for the method. Basic is just forwarding through both losses """
    track = super(RepulsiveKernel, self).forward(trainer, params, shared_params, track, x, gt, ignore_ood_loss=ignore_ood_loss, step=step)
    flat_params = flatten_keys(params)
    
    # handle warmup steps linearly. Repulsive force is scaled by gamma
    if trainer.training and self.warmup is not None and step is not None and step < self.warmup:
      warmup_coeff = (1e-8 + step) / self.warmup
    else:
      warmup_coeff = 1.0
    
    # # inlier repulsive loss always defined
    track['ind_kernel'] = OrderedDict()
    track['ind_rep_loss'] = self.model_kernel(flat_params, track['feats_ind'], track['pred_ind'], track['ind_kernel'])
    track['loss'] = track['loss'] + warmup_coeff * torch.abs(self.gamma) * (track['ind_rep_loss'] if not track['ind_rep_loss'].isnan() else 0.0)
    
    # include ood repulsive loss if needed
    if self.use_ood and self.model_kernel_ood is not None:
      track['ood_kernel'] = OrderedDict()
      track['ood_rep_loss'] = self.model_kernel_ood(flat_params, track['feats_ood'], track['pred_ood'], track['ood_kernel']) if self.use_ood else None
      track['loss'] = track['loss'] + warmup_coeff * torch.abs(self.gamma_ood) * (track['ood_rep_loss'] if not track['ood_rep_loss'].isnan() else 0.0)

    return track

  @staticmethod
  def from_config(config: dict):
    """ Create a method from a configuration dictionary """
    config = ParticleMethods.from_config(config)
    
    if 'model_kernel_ood' in config:
      config['model_kernel_ood'] = build_model_kernel(config['model_kernel_ood'])
    
    return config

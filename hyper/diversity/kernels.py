""" Kernel functions for calculating gram matrices on batched data """
from torch import Tensor
import torch
import torch.nn as nn
import math
import random
import numpy as np


# specify the additional variables
# that could be tracked or used in the kernel
# see model_kernels.py for usage/to keep track of learnable parameters
KERN_PARAMS = {
  'linear': {},
  'log': {},
  'cossim': {
    'eps': 1e-6
  },
  'hyperenergy': {
    'arc_eps': 0.005,
    'eps': 1e-6,
    'mhe_s': 2
  },
  'rbf': {
    'param': 1.0
  },
  'laplacian': {
    'param': 1.0
  }
}


class EMA(nn.Module):
  def __init__(self, decay: float, initial: Tensor=None):
    """ Exponential Moving Average class
    
    Used in testing to track median bandwidth for RBF kernels/smooth calculated parameters
    """
    super().__init__()
    self.decay = decay

    self.ten = initial
    if initial is None:
      shadow = None
    else:
      shadow = initial.clone().detach()

    self.register_buffer('shadow', shadow, persistent=True)

  @torch.no_grad()
  def update(self):
    if not self.training:
        # print("EMA update should only be called during training", file=stderr, flush=True)
        return

    # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    self.shadow.sub_((1. - self.decay) * (self.shadow - self.ten))
  
  def get(self):
    return self.shadow
  

  @torch.no_grad()
  def forward(self, inputs: Tensor, update: bool=True) -> Tensor:
    if self.training and update:
      if self.ten is None:  # first time do not update MA
        self.ten = inputs
        self.shadow = inputs.clone().detach()
      else:
        self.ten = inputs
        self.update()  # update moving average
    return self.shadow  # return the shadow variable


def detach_diagonal(gram: torch.Tensor, readd: bool=True) -> torch.Tensor:
  """ Detaches diagonal elements of a batched matrix

  Args:
    gram (torch.Tensor): [B, N, P]
    readd (bool): readds the diagonal elements back to the matrix

  Returns:
    torch.Tensor: [B, N, P] with diagonal along each B_i detached
  """
  B, N = gram.shape[:2]
  with torch.no_grad():
    # copy diagonal
    diags = gram.diagonal(offset=0, dim1=1, dim2=-1)
    diag_embed = torch.diag_embed(diags, offset=0, dim1=1, dim2=-1).detach()
    
    # create zeroing matrix
    zeroer = torch.ones(N, N, device=gram.device, dtype=gram.dtype)
    zeroer.fill_diagonal_(0.0)

    # repeat them across B
    zeroer = zeroer.repeat(B, 1, 1)

  # zero out the diagonal then add detached diagonal
  if readd:
    gram = gram.multiply(zeroer) + diag_embed
  else:
    gram = gram.multiply(zeroer)
  return gram


def linear_gram(X: torch.Tensor, Y: torch.Tensor=None, detach_diag: bool=True, readd: bool=True, *args, **kwargs) -> torch.Tensor:
  """ Creates batched linear gram matrix k(xi, yi) = x^Ty

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readdd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  if Y is None:
    Y = X

  gram = torch.bmm(X.contiguous(), Y.transpose(1, 2).contiguous())

  if detach_diag:
    gram = detach_diagonal(gram, readd=readd)

  return gram


def log_gram(X: torch.Tensor, Y: torch.Tensor=None, detach_diag: bool=True, readd: bool=True, *args, **kwargs) -> torch.Tensor:
  """ Creates batched linear gram matrix k(xi, yi) = x^Ty

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readdd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  return torch.log(torch.abs(linear_gram(X, Y, detach_diag, readd, *args, **kwargs)) + 1e-6)


def cossim_gram(X: torch.Tensor, Y: torch.Tensor=None, detach_diag: bool=True, readd: bool=True, *args, **kwargs) -> torch.Tensor:
  """ Creates batched cosine similarity gram matrix k(xi, yi) = x^Ty / ||x|| ||y||

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readdd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  if Y is None:
    Y = X

  gram = torch.bmm(X.contiguous(), Y.transpose(1, 2).contiguous())

  # normalize by the norms
  norms = torch.sqrt(torch.bmm(X.square().sum(dim=2, keepdim=True), Y.square().sum(dim=2, keepdim=True).transpose(1, 2)))
  gram = gram / norms

  if detach_diag:
    gram = detach_diagonal(gram, readd=readd)

  return gram


def hyperenergy_gram(X: torch.Tensor, Y: torch.Tensor=None, detach_diag: bool=True, readd: bool=True, *args, **kwargs) -> torch.Tensor:
  """ Creates batched hyperspherical energy gram matrix
  
  Args:
    X (torch.Tensor): expected shape of [B, N, P]
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readd (bool): readds the diagonal elements back to the matrix
  """

  # get cossim values
  cosim = cossim_gram(X, Y, detach_diag=False, readd=True)

  # get the energy
  arc_eps = 1.0 + kwargs.get('arc_eps', 0.005)
  eps = kwargs.get('eps', 1e-4)
  mhe_s = kwargs.get('mhe_s', 2)
  if not isinstance(arc_eps, float) and arc_eps.ndim > 0:
    arc_eps = arc_eps.view(arc_eps.shape[0], 1, 1)
    eps = eps.view(arc_eps.shape[0], 1, 1)
    
  arcs = torch.arccos(cosim / arc_eps)
  energy = (1.0 + eps) / (torch.pow(arcs, mhe_s) + eps)
  return energy


def rbf_gram(X: torch.Tensor, Y: torch.Tensor=None, param: float=1.0, median: bool=True, detach_diag: bool=True, readd: bool=True, grad_var=None, median_ma: EMA=None, median_update: bool=True, detach_right: bool=False, *args, **kwargs):
  """ Creates batched rbf gram matrix

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  # fix to single batch item if not batched
  if X.ndim == 2:
    X = X.unsqueeze(0)
    Y = Y.unsqueeze(0) if Y is not None else None
    single = True
  else:
    single = False
  
  # make contiguous
  X = X.contiguous()
  if Y is None:
    Y = X
  else:
    Y = Y.contiguous()
  
  if detach_right:
    Y = Y.detach()

  # pairwise distance
  XX = torch.bmm(X, X.transpose(1, 2))
  XY = torch.bmm(X, Y.transpose(1, 2))
  YY = torch.bmm(Y, Y.transpose(1, 2))
  dists = -2 * XY + XX.diagonal(offset=0, dim1=1, dim2=2).unsqueeze(2) + YY.diagonal(offset=0, dim1=1, dim2=2).unsqueeze(1)
  
  # detach in a non-inplace way by zeroing out diag then adding it back
  if detach_diag:
    dists = detach_diagonal(dists, readd=readd)
  
  # see cuda.cpp for description of this method:
  B = X.shape[0]

  # construct scaling terms (default is just param)
  if isinstance(param, float):
    params = param * torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
  else:
    if param.ndim > 0 and param.shape[0] > 1:
      params = param.view(param.shape[0], 1, 1)
    else:
      params = param.squeeze() * torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
  
  # use median heuristic for bandwidth
  if median:
    with torch.no_grad():
      weight = 2.0 * math.log(X.size(1) + 1)
      if median_ma is not None:
        # use interval median updates
        if median_update:
          # pick a random model to approximate
          rnd = random.randint(0, B-1)
          med = median_ma(dists[rnd].median().detach() / weight, update=True)
        else:
          med = median_ma.get()  # get current bandwidth
        
        for i in range(B):
          params[i] = param * med  # heuristic shared across batch
      else:
        # calculate median for all models
        for i in range(B):
          # calculate heuristic/parameter
          params[i] = param * dists[i].median().detach() / weight  # param is a scaler of median heuristic
  else:
    # non-median heuristic/normal sigma
    params = torch.square(params)
  
  # now fix distance/create grams
  grams = (-dists / (2.0 * params)).exp()
  
  # fix single batch item gram
  if single:
    grams = grams.squeeze(0)
  
  # calculate gradients
  if grad_var is not None:
    return grams, torch.autograd.grad(grams.sum(), grad_var, retain_graph=True)[0]
  
  # return RBF gram matrix
  return grams


def laplacian_gram(X: torch.Tensor, Y: torch.Tensor=None, param: float=1.0, median: bool=True, detach_diag: bool=True, readd: bool=True, *args, **kwargs):
  """ Creates batched laplacian gram matrix 

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  raise NotImplementedError('Currently laplacian kernel is not complete')
  
  if Y is None:
    Y = X.contiguous()
  else:
    Y = Y.contiguous()

  # pairwise distance
  X = X.contiguous()
  dists = torch.cdist(X, Y, p=2)
  
  # detach in a non-inplace way by zeroing out diag then adding it back
  if detach_diag:
    dists = detach_diagonal(dists, readd=readd)
  
  # see cuda.cpp for description of this method:
  B = X.shape[0]

  # construct scaling terms (default is just param)
  params = param * torch.ones(B, 1, 1, device=X.device, dtype=X.dtype)
  
  if median:
    for i in range(B):
      # calculate heuristic/parameter
      with torch.no_grad():
        params[i] = param * 2.0 * dists[i].median()  # param is a scaler of median heuristic
  
  # now fix distance/create grams
  grams = torch.exp((-dists).divide(params.detach()))
  
  # return laplacian gram matrix
  return grams


def center_gram(gram: torch.Tensor, ma: EMA=None, update_ma: bool=True) -> torch.Tensor:
  """ Centers gram matrix row-wise

  Args:
      gram (torch.Tensor): [B, N, P] batched gram matrix to center features for

  Returns:
      torch.Tensor: centered gram matrix
  """
  if ma is None:
    return gram - gram.mean(dim=2, keepdim=True)
  
  # mean = ma(gram.mean(dim=2).mean(0), update=update_ma)  # now [B, N] consider this B updates to MA
  # print(gram.shape, gram.mean(dim=2).mean(0).shape, mean.shape)
  
  # if means misaligned then get global mean not feature mean
  # if mean.shape[0] != gram.shape[1]:
  #   mean = mean.mean(0).view(1, -1, 1)
  
  mean = ma(gram.mean((1, 2), keepdim=True), update=update_ma)
  
  # if mean across model BS is not the same as the gram BS then take the mean of the means
  # should only happen when using a method/testing a variable amount of models
  if mean.shape[0] != gram.shape[0]:
    mean = mean.mean(0).view(1, -1, 1)
  
  return gram - mean# .view(1, -1, 1)


MAPS = {
  'linear': linear_gram,
  'log': log_gram,
  'cossim': cossim_gram,
  'hyperenergy': hyperenergy_gram,
  'rbf': rbf_gram,
  'laplacian': laplacian_gram
}


def learned_gram(X: torch.Tensor, Y: torch.Tensor=None, detach_diag: bool=True, readd: bool=True, *args, **kwargs) -> torch.Tensor:
  """ Creates batched linear gram matrix k(xi, yi) = x^Ty

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    detach_diag (bool): detaches diagonal elements from backward
    readdd (bool): readds the diagonal elements back to the matrix
    
  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  if Y is None:
    same = True
    Y = X
  else:
    same = False

  # apply model on X and Y
  model = kwargs['model']
  
  # shared feature model
  B, N, P = X.shape
  L = Y.shape[1]
  
  # if using shared feature model or not
  if isinstance(model, (tuple, list)):
    X = torch.stack([m(X[i]) for i, m in enumerate(model)])
  else:
    # kernel shared across models
    BN, BL = B * N, B * L
    X = model(X.reshape(BN, P)).reshape(B, N, -1)
  
  if same:  # do not recompute given same features
    Y = X
  else:
    if isinstance(model, (tuple, list)):
      Y = torch.stack([m(Y[i]) for i, m in enumerate(model)])
    else:
      # kernel shared across models
      BN, BL = B * N, B * L
      Y = model(X.reshape(BN, P)).reshape(B, N, -1)

  # apply the applied kernel now
  n_kernel = kwargs['learned_kernel']
  method = MAPS[n_kernel]
  pass_kwargs = kwargs
  pass_kwargs['detach_diag'] = detach_diag
  pass_kwargs['readd'] = readd
  return method(X, Y, **kwargs)

def batched_gram(X: torch.Tensor, Y: torch.Tensor=None, kernel: str='linear', detach_diag: bool=True, readd: bool=True, center: bool=True, ma: EMA=None, update_ma: bool=True, *args, **kwargs):
  """ Creates batched gram matrix

  Args:
    X (torch.Tensor): expected shape of [B, N, P] 
    Y (torch.Tensor): expected shape of [B, L, P]. Optional defaults to X.
    kernel (str): the type of kernel. Default is linear
    detach_diag (bool): detaches diagonal elements from backward
    readd (bool): readds the diagonal elements back to the matrix
    center (bool): center features row-wise of the gram matrix
    *args, **kwargs: all other args are passed to specific gram function

  Returns:
    torch.Tensor: [B, N, L] or [B, N, N] if Y=None
  """
  method = None
  if kernel == 'learned':
    method = learned_gram
  else:
    method = MAPS[kernel]

  # construct gram matrix
  gram = method(X, Y, detach_diag=detach_diag, readd=readd, *args, **kwargs).contiguous()

  # possibly center features
  if center:
    return center_gram(gram, ma=ma, update_ma=update_ma)
  
  return gram


def vectorize_grams(gram: torch.Tensor) -> torch.Tensor:
  """ Converts the batched feature matrices into a new family of feature matrices that are vectorized with unit norms

  Args:
    gram (torch.Tensor): [B, N, P] gram matrix format between samples N and P

  Returns:
    torch.Tensor: [B, N*P] new vectorized variant (also contiguous) of the grame matrix
  """
  return gram.reshape(gram.shape[0], gram.shape[1]*gram.shape[2])

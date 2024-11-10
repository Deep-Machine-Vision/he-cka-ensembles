from functools import partial
from argparse import Namespace
from typing import Generator, Iterable, Iterator, Optional, Tuple, Union
from torch import TensorType
from tqdm import tqdm
from torch import DeviceObjType
import torch.optim as optim
import numpy as np
import argparse
import math
from bisect import bisect_right, bisect_left
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import numpy as np
import torch
import copy


AVAILABLE_SCHEDULERS = {}

def register_scheduler(name: str, override=None):
  """ Decorator to register a model """
  if override is not None:
    AVAILABLE_SCHEDULERS[name] = override
  else:
    def decorator(cls):
      AVAILABLE_SCHEDULERS[name] = cls
      return cls
    return decorator


def build_scheduler(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Metric configurations not found in configs. Must define a metric and contain the name of the metric as a key in that dictionary')
  
  cls = AVAILABLE_SCHEDULERS[name]
  return partial(cls, **config)



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer, warmup, T_max, min_lr_factor=0):
    self.warmup = warmup
    self.max_num_iters = T_max
    self.min_lr = min_lr_factor
    super().__init__(optimizer)

  def get_lr(self):
    lr_factor = self.get_lr_factor(epoch=self.last_epoch)
    return [base_lr * lr_factor for base_lr in self.base_lrs]

  def get_lr_factor(self, epoch):
    if epoch == 0:
      epoch = 1
    lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
    if epoch <= self.warmup and self.warmup > 0:
        lr_factor *= epoch * 1.0 / self.warmup
    return lr_factor


class CyclicCosAnnealingLR(_LRScheduler):
  r""" Modified from https://raw.githubusercontent.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch/master/cyclicLR.py
  
  Implements reset on milestones inspired from CosineAnnealingLR pytorch
  
  Set the learning rate of each parameter group using a cosine annealing
  schedule, where :math:`\eta_{max}` is set to the initial lr and
  :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
  .. math::
      \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
      \cos(\frac{T_{cur}}{T_{max}}\pi))
  When last_epoch > last set milestone, lr is automatically set to \eta_{min}
  It has been proposed in
  `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
  implements the cosine annealing part of SGDR, and not the restarts.
  Args:
      optimizer (Optimizer): Wrapped optimizer.
      milestones (list of ints): List of epoch indices. Must be increasing.
      decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
      gamma (float): factor by which to decay the max learning rate at each decay milestone
      eta_min (float): Minimum learning rate. Default: 1e-6
      last_epoch (int): The index of last epoch. Default: -1.
      
      
  .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
      https://arxiv.org/abs/1608.03983
  """

  def __init__(self, optimizer, milestones, decay_milestones='same', gamma=0.5, eta_min=1e-6, T_max=-1, warmup=0):
    if T_max < 0:
      raise ValueError('need a non-negative last_epoch, got {}'.format(T_max))
    
    # just linearly divide epochs into milestones
    if isinstance(milestones, int):
      milestones = np.linspace(0, T_max, milestones + 1, dtype=int).tolist()
    
    # default to milestones
    if decay_milestones is not None and decay_milestones == 'same':
      decay_milestones = milestones
    
    if not list(milestones) == sorted(milestones):
      raise ValueError('Milestones should be a list of'
                        ' increasing integers. Got {}', milestones)
    
    self.eta_min = eta_min
    self.milestones = milestones
    self.milestones2 = decay_milestones
    self.warmup = warmup
    
    self.gamma = gamma
    super(CyclicCosAnnealingLR, self).__init__(optimizer, T_max)
      
  def get_lr(self):
    # warmup period
    if self.last_epoch < self.warmup:
      return [self.eta_min + (base_lr * (self.last_epoch / self.warmup)) for base_lr in self.base_lrs]
    last_epoch = self.last_epoch#  + self.warmup
    
    if self.last_epoch >= self.milestones[-1]:
      return [self.eta_min for base_lr in self.base_lrs]

    idx = bisect_right(self.milestones,self.last_epoch)
    
    left_barrier = 0 if idx==0 else self.milestones[idx-1]
    right_barrier = self.milestones[idx]

    width = right_barrier - left_barrier
    curr_pos = last_epoch - left_barrier 
    
    if self.milestones2:
      return [self.eta_min + ( base_lr* self.gamma ** bisect_right(self.milestones2, last_epoch)- self.eta_min) *
              (1 + math.cos(math.pi * curr_pos/ width)) / 2
              for base_lr in self.base_lrs]
    else:
      return [self.eta_min + (base_lr - self.eta_min) *
          (1 + math.cos(math.pi * curr_pos/ width)) / 2
          for base_lr in self.base_lrs]


# register the specified schedulers
register_scheduler('cosine_annealing', torch.optim.lr_scheduler.CosineAnnealingLR)
register_scheduler('warmup_cosine_annealing', CosineWarmupScheduler)
register_scheduler('cyclic_cosine_annealing', CyclicCosAnnealingLR)

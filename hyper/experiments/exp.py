""" Contains common overview functions for experiments """

import traceback
# from hyper.experiments.models.hyper import build_generator
from hyper.data import load_dataset
from collections import OrderedDict
from hyper.experiments.metrics import build_metric
from hyper.diversity.losses import build_loss
from hyper.experiments.training import HyperTrainer
from hyper.diversity.methods import build_method
from hyper.generators.base import build_generator
from hyper.experiments.schedulers import build_scheduler
import hyper.target  # do not remove! required to register modules before building them
import importlib.machinery
import importlib.util
from functools import partial
import torch.autograd as autograd
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import time
import math
import copy
import os
import json
import yaml
import matplotlib.pyplot as plt


def null_fn(*args, **kwargs):
  """ Null function """
  pass


### basic functions to initially build experiments
# this builds all basic ones. Others could be loaded through the extras python file
def build_exp_requirements(exp_type: str, sub_type: str=None, bs: int=1, configs: dict=None, ddp: bool=False, load_seed=None):
  """ Builds the requirements for a basic experiment here """
  if configs is None:
    configs = {}
  
  # based on experiment type
  if exp_type == 'toy':
    data = load_dataset('toy', bs=bs, type=sub_type, ddp=ddp, **configs)
  elif exp_type == 'tinyimagenet':
    data = load_dataset('tinyimagenet', bs=bs, drop_last=True, ddp=ddp, load_seed=load_seed)  # keep batch size consistent
  elif exp_type == 'cifar' or exp_type == 'cifar100':
    configs['cifar100'] = 'cifar100' == exp_type
    if ddp:
      raise RuntimeError('@TODO add cifar exp to rewritten codebase')
    # create dataset for corrupted data if expected
    data = load_dataset(f'cifar', bs=bs, drop_last=False, load_seed=load_seed, **configs)
  elif exp_type == 'mnist':
    data = load_dataset('mnist', bs=bs, dirty=True, ddp=ddp, load_seed=load_seed,  **configs)
  else:
    raise NotImplementedError(f'The experiment type {exp_type} has not been added to build_exp_requirements')
  
  return {
    'data': data,
    'bs': bs
  }


def build_from_dict(configs, log=null_fn, base_path=None, out_path=None, ddp=False, trainer_kwargs=None):
  if trainer_kwargs is not None and 'device_id' in trainer_kwargs:
    log_device_id = ' on device %s' % str(trainer_kwargs['device_id'])
  else:
    log_device_id = ''
  print(f'Loading configuration{log_device_id}:', configs)
  configs = copy.deepcopy(configs)
  
  # is there an extra functions file to load from
  # load python file with current directory in path
  if 'extras_file' in configs:
    if base_path is None:
      raise RuntimeError('Must provide base path to load extras file')
    extras = configs['extras_file']
    based = os.path.abspath(os.path.dirname(base_path))
    full_path = os.path.join(based, extras)
    mod_name = os.path.splitext(os.path.basename(full_path))[0]
    loader = importlib.machinery.SourceFileLoader(mod_name, full_path)
    spec = importlib.util.spec_from_loader(mod_name, loader)
    loaded_extras = importlib.util.module_from_spec(spec)
    loader.exec_module(loaded_extras)
    
    # functions should now be available and let's swap out ext. with the loaded functions
    def swapper(cur):
      if cur is None:
        return None
      if isinstance(cur, dict):
        return {k: swapper(v) for k, v in cur.items()}
      if isinstance(cur, (list, tuple, set)):
        return [swapper(v) for v in cur]
      if isinstance(cur, str) and cur.startswith('ext.'):  # loading an external function/class
        # if called with arguments
        # assume it's a function/pass arguments
        if '(' in cur:
          parts = cur.split('(')
          func = getattr(loaded_extras, parts[0][4:])
          args = []
          post = parts[1].split(')')[0]
          if len(post) > 0:
            args = [eval(a) for a in post.split(',')]
          return func(*args)
        return getattr(loaded_extras, cur[4:])  # otherwise just return the attribute

      return cur
    
    # swap out functions
    configs = swapper(configs)
  
  # handle special configuration for sweeps like codesize/latentsize
  def replace_config(_map, match, value):
    for k in _map.keys():
      if _map[k] == match:
        _map[k] = value
      elif isinstance(_map[k], (dict, OrderedDict)):
        replace_config(_map[k], match, value)
  
  if 'latent_size' in configs:
    replace_config(configs, '__latent_size__', int(configs['latent_size']))
  if 'code_size' in configs:
    replace_config(configs, '__code_size__', int(configs['code_size']))
  
  # construct the method from the configs
  method = build_method(configs['method'])
  
  
  # remove weights from hyper if specified (as this is handled by the trainer loader)
  if isinstance(configs['hyper'], (OrderedDict, dict)):
    hyper_configs = copy.deepcopy(configs['hyper'])
    if 'weights' in hyper_configs:
      del hyper_configs['weights']
    hyper = build_generator(hyper_configs)
  else:
    hyper = configs['hyper']  # already a module
  
  # build metrics
  if 'metrics' in configs['trainer']:
    configs['trainer']['metrics'] = [build_metric(metric) for metric in configs['trainer']['metrics']]
  
  # build the dataset requirements
  if 'requirements' not in configs:
    raise RuntimeError('Requirements must be specified in the configuration')
  
  # define experiment by name or leave as is (ie use extras file configuration)
  if isinstance(configs['requirements'], dict) and 'exp_name' in configs['requirements']:
    exp_name = configs['requirements']['exp_name']
    sub_type = configs['requirements'].get('sub_type')
    load_seed = configs['requirements'].get('seed')
    batch_size = int(configs['requirements'].get('batch_size', 100))
    other_configs = configs['requirements'].get('loader_args', {})
    require = build_exp_requirements(exp_name, sub_type, batch_size, other_configs, ddp=ddp, load_seed=load_seed)
  else:  # assume loaded from extras
    require = configs['requirements']
  
  if 'output' in configs:
    out_path = configs['output']
  
  # build the rollout loss if applicable
  if 'learn_hyper_loss' in configs['trainer']:
    configs['trainer']['learn_hyper_loss'] = build_loss(configs['trainer']['learn_hyper_loss'])
  
  # build the scheduler if needed. Note the scheduler could already be loaded from externals file
  if 'scheduler' in configs['trainer'] and isinstance(configs['trainer']['scheduler'], (OrderedDict, dict)):
    configs['trainer']['scheduler'] = build_scheduler(configs['trainer']['scheduler'])
  
  # include other arguments
  extra_kwargs = {}
  if trainer_kwargs is not None:
    extra_kwargs = trainer_kwargs
  
  trainer = HyperTrainer(
    data=require['data'],
    hyper=hyper,
    method=method,
    log=log,
    output_dir=out_path,
    **configs['trainer'],
    **extra_kwargs
  )
  return trainer


# merge configs
def merge_config(a, b, l_extra=None, base_path=None):
  if isinstance(a, dict) and isinstance(b, dict):
    if 'extras_file' in a and not 'extras_file' in b:
      l_extra = base_path
    
    for k, v in b.items():
      if k in a:
        a[k], l_extra = merge_config(a[k], v, l_extra, base_path)
      else:
        a[k] = v
    return a, l_extra
  return b, l_extra


def build_from_file(file, override_config: dict=None, log=null_fn, load_raw: bool = False, out_path=None, ddp=False, trainer_kwargs=None):
  """ Builds an experiment from a file """
  if not os.path.isfile(file):
    raise FileNotFoundError(f'Could not find file {file}')
  
  # make a copy of override config to avoid modifying the original
  if override_config is not None:
    override_config = copy.deepcopy(override_config)
  
  # load from json
  if file.endswith('.json'):
    with open(file, 'r') as f:
      configs = json.load(f)
  elif file.endswith('.yaml') or file.endswith('.yml'):
    with open(file, 'r') as f:
      configs = yaml.safe_load(f)
  else:
    raise RuntimeError(f'Unsupported file type {file}')
  
  # handle any base dependencies
  load_extras_path = file
  if 'base' in configs:
    base = configs['base']
    base_path = os.path.join(os.path.dirname(file), base)
    del configs['base']
    base_configs = build_from_file(base_path, log=log, load_raw=True, out_path=out_path)
    configs, load_extras_path = merge_config(base_configs, configs, load_extras_path, base_path)
  
  # return raw configs only
  if load_raw:
    return configs
  
  # handle overrides
  if override_config is not None:
    configs = copy.deepcopy(configs)
    configs = merge_config(configs, override_config)[0]
    
  # print('LOADED', configs)
  # exit(0)
  
  return configs, build_from_dict(configs, log=log, base_path=load_extras_path, out_path=out_path, ddp=ddp, trainer_kwargs=trainer_kwargs)


""" Contains common overview functions for experiments """

import traceback
# from hyper.experiments.models.hyper import build_generator
from hyper.data import load_dataset
from collections import OrderedDict
from hyper.experiments.metrics import build_metric
from hyper.net.spectral import spectral_norm_fc, spectral_norm_conv
from hyper.diversity import batched_gram, pairwise_cossim, unit_vector_rows
from hyper.diversity.losses import build_loss
from hyper.util import CosineWarmupScheduler, CyclicCosAnnealingLR
from hyper.util.collections import flatten_keys, unflatten_keys
from hyper.diversity.kernels import EMA, batched_gram, rbf_gram
from hyper.diversity.ssge import SpectralSteinEstimator
from hyper.experiments.training import HyperTrainer
from hyper.diversity.methods import build_method
from hyper.generators.base import build_generator
import hyper.target  # required to register modules
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
def build_exp_requirements(exp_type: str, sub_type: str=None, bs: int=1, configs: dict=None, ddp=False):
  """ Builds the requirements for a basic experiment here """
  if configs is None:
    configs = {}
  
  # based on experiment type
  if exp_type == 'toy':
    data = load_dataset('toy', bs=bs, type=sub_type, ddp=ddp, **configs)
  elif exp_type == 'tinyimagenet':
    data = load_dataset('tinyimagenet', bs=bs, drop_last=True, ddp=ddp)  # keep batch size consistent
  elif exp_type == 'cifar' or exp_type == 'cifar100':
    configs['cifar100'] = 'cifar100' == exp_type
    if ddp:
      raise RuntimeError('@TODO add cifar exp to rewritten codebase')
    # create dataset for corrupted data if expected
    data = load_dataset(f'cifar', bs=bs, drop_last=False, **configs)
  elif exp_type == 'mnist':
    data = load_dataset('mnist', bs=bs, dirty=True, ddp=ddp, **configs)
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
  hyper = build_generator(configs['hyper'])
  
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
    batch_size = int(configs['requirements'].get('batch_size', 100))
    other_configs = configs['requirements'].get('loader_args', {})
    require = build_exp_requirements(exp_name, sub_type, batch_size, other_configs, ddp=ddp)
  else:  # assume loaded from extras
    require = configs['requirements']
  
  if 'output' in configs:
    out_path = configs['output']
  
  # build the rollout loss if applicable
  if 'learn_hyper_loss' in configs['trainer']:
    configs['trainer']['learn_hyper_loss'] = build_loss(configs['trainer']['learn_hyper_loss'])
  
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


def run_training(requirements, configs, log=null_fn):
  """ Runs typical training loop for an experiment """
  hyper, data, target, loss_fn = requirements['hyper'], requirements['data'], requirements['target'], requirements['loss']
  metric_fn = requirements.get('metrics', null_fn)
  epoch_end = requirements.get('epoch_end', null_fn)
  ood_sample = requirements.get('ood_sample', null_fn)
  ood_loss_fn = requirements.get('ood_loss', null_fn)
  bs = requirements['bs']
  
  model_bs = configs['model_bs']  # model batch size

  # rpobably incremental sampling so get largest (assuming ordered)
  if isinstance(model_bs, list):
      model_bs = model_bs[-1]

  weight_file = configs.get('file', 'weights.pt')  # default do not save
  load_weight = configs.get('load', True)
  save_weight = configs.get('save', True)
  run_open = configs.get('run_open', False)
  svgd = configs.get('svgd', False)
  method = configs.get('method', 'ws')
  feat_kernel = configs.get('feat_kernel', 'linear')
  feat_learned_kernel = configs.get('learned_feat_kernel', 'linear')
  feat_kernel_last = configs.get('feat_kernel_last', feat_kernel)  # default to same as first
  kernel = configs.get('kernel', 'he')
  open_K = configs.get('open_K')
  
  moving_avgs = []
  median_moving_avgs = []
  
  prior_variance = 1.0
  prior_state = None
  # ood_exp_entropy = math.log(2) - 0.1

  # cache parameter shapes
  param_shapes = flatten_keys(hyper.get_parameter_from_shapes())
  target_parameters = list(hyper.target_parameters())
  gamma_ood_init = configs.get('gamma_ood', 0.0)
  target_weight_decay = configs.get('target_weight_decay', 0.0)
  configs['_gamma_ood'] = gamma_ood_init
  # gamma_delta = 0.0

  
  
  loaded = False
  load_new = False
  if weight_file is not None and load_weight:
    if os.path.isfile(weight_file):
      print('Attempting to load previous weights from ', weight_file)
      try:
        save_data = torch.load(weight_file)
        # new loader
        if 'hyper' in save_data:
          hyper.load_state_dict(save_data['hyper'], strict=False)
          load_new = True
        else:  # old loader/old weight files
          hyper.load_state_dict(save_data, strict=False)
          load_new = False
        loaded = True
      except RuntimeError as err:
        print(f'Failed to load weights {str(err)}')
    else:
      print('Failed to find weight file', weight_file, 'ignoring...')
  
  train_loader, test_loader = data
  num_train, num_test = len(train_loader), len(test_loader)
  
  # we need to presolve min hyperspherical energy
  # to make normalized scale
  if not configs.get('use_pcka', False) and not configs.get('he_raw', False):
    print('Precomputing min/max hyperspherical energy')
    # ensure batched gram calculation is sim to before
    fake_grams = torch.empty(model_bs, bs, bs, requires_grad=True, dtype=torch.float32, device='cuda')
    torch.nn.init.normal_(fake_grams)
      
    # maximize energy for each method
    # now minimize hyperspherical energy
    fake_optim = torch.optim.Adam([fake_grams], lr=0.05)
    fake_sched = torch.optim.lr_scheduler.StepLR(fake_optim, step_size=10, gamma=0.1)

    for i in range(100):
      fake_optim.zero_grad()

      with torch.no_grad():
        fake_grams.data = unit_vector_rows(fake_grams.data, eps=0.0)

      if svgd:
        energy = mean_hyperspherical_energy(fake_grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=False)  # True)
      else:
        energy = mean_hyperspherical_energy(fake_grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True))
      energy = energy.mean()
      energy.backward()
      fake_optim.step()
      fake_sched.step()
    min_he_energy = energy.min().item()
    
    # now calculate "max" via a projected gaussian far away
    # simulating grams really close together
    with torch.no_grad():
      torch.nn.init.normal_(fake_grams, mean=1000.0, std=0.1)
      if svgd:
        max_energy = mean_hyperspherical_energy(fake_grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=False)  # True)
      else:
        max_energy = mean_hyperspherical_energy(fake_grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True))
      max_he_energy = max_energy.max().item()  # approximate "max"
    
    print('Completed hyperspherical energy max and min calc')
    print('FINAL ITER', i, 'MIN ENERGY', min_he_energy, 'MAX ENERGY', max_he_energy)

  lr = configs['lr']
  params = []
  params_ood = []

  # sample a batch
  for b, (X, Y) in enumerate(train_loader):
    X = X.cuda()
    Y = Y.cuda()
    break

  # sample a batch of parameters
  sparam = hyper.sample_params(model_bs, device=X.device)  # sample init codes (should change name... @TODO)
  mparams = hyper.forward_params(sparam)  # feed through param generator/hypernetwork
  pred, feat, feat_ood, _, orig_feats = forward(X, mparams, ood_N=0, include_orig_feats=True)
  orig_feats_flat = flatten_keys(orig_feats)
  
  first = feat[0]
  with torch.no_grad():
    start_value = configs.get('gamma_start_value', 1.0)
    if configs.get('gamma_increase_rate', None) is None:
      weights = torch.linspace(1.0, 10.0, len(feat), device=first.device, dtype=first.dtype)
    else:
      weights = torch.linspace(1.0, configs.get('gamma_increase_rate', 1.0) * len(feat), len(feat), device=first.device, dtype=first.dtype)
    
    # set initial value weight
    weights[0] = start_value
      
    if 'gamma_second_value' in configs and configs['gamma_second_value'] is not None:
      weights[1] = configs['gamma_second_value']
      
    # last layer has first layer's CKA weight
    
    # set uniform weighting
    if configs.get('unif_layer_weight', False):
      weights[:] = 1.0
      
    # set last weight if applicable
    if 'gamma_end_value' in configs and configs['gamma_end_value'] is None:
      configs['gamma_end_value'] = len(feat)
    weights[-1] = configs.get('gamma_end_value', (10.0 if (configs.get('gamma_increase_rate', None) is None) else len(feat)))
    
    # normalize weights
    # weights /= torch.linalg.vector_norm(weights, ord=2)
    # weights /= weights.sum()
    
  # handle learnable smoothing parameter
  # very critical for non-destructive diversity
  # positive examples being inlier and negative being synthetic OOD
  ood_params = []
  ood_hyper_params = []
  if configs.get('learn_diversity', False):
    if configs.get('ood_sample', 0) == 0:
      raise RuntimeError('For learnable eps/diversity params you need ood samples for negative signal')

    # now we know how many matrices to expect let's build params
    # that could be learned
    empty = lambda init, shape=(): torch.nn.Parameter(init * torch.ones(shape, device='cuda'), requires_grad=True)
    grad_param = empty(0.0 if configs.get('grad_penalty') is None else configs.get('grad_penalty'))
    if configs.get('grad_penalty') is not None:
      ood_hyper_params.append(grad_param)
      params_ood.append({
        'params': grad_param,
        'initial_lr': lr,
        'lr':lr
      })
    
    for ind, (features, orig_feature) in enumerate(zip(feat, orig_feats_flat.values())):
      # load from file instead of here
      # as file has updated values
      ood_options = configs
      if loaded and load_new:
        ood_options = save_data.get('ood_params', [])
        if ind < len(ood_options):
          ood_options = ood_options[ind]  # load layer information from file
        else:
          ood_options = configs  # default use the configuration by user
      
      shared_shape = (features.shape[0]) if not configs.get('learned_shared', True) else ()
      eps_param = empty(ood_options.get('eps', 1e-4), shape=())
      arc_param = empty(ood_options.get('arc_eps', 1e-4), shape=())
      mhe_param = empty(ood_options.get('mhe_s', 2.0), shape=())
      rbf_param = empty(ood_options.get('rbf_sigma', 1.0), shape=shared_shape)  # shared or separate amongst models
      
      # learn separate values
      if configs.get('separate_ood_features', True):
        ood_eps_param = empty(ood_options.get('eps', 1e-4), shape=())
        ood_arc_param = empty(ood_options.get('arc_eps', 1e-4), shape=())
        ood_mhe_param = empty(ood_options.get('mhe_s', 2.0), shape=())
        ood_rbf_param = empty(ood_options.get('rbf_sigma', 1.0), shape=shared_shape)  # shared or separate amongst models
      else:
        ood_eps_param = eps_param
        ood_arc_param = arc_param
        ood_mhe_param = mhe_param
        ood_rbf_param = rbf_param
      
      # only used for learned kernel with hyperspherical energy featurespace kernel
      learned_kernel_eps_param = empty(ood_options.get('leps', ood_options.get('eps', 1e-4)), shape=shared_shape)
      learned_kernel_arc_param = empty(ood_options.get('larc_eps', ood_options.get('arc_eps', 1e-4)), shape=shared_shape)
      learned_kernel_mhe_param = empty(ood_options.get('lmhe_s', ood_options.get('mhe_s', 2.0)), shape=shared_shape)
      if configs.get('separate_ood_learned_kernel', True):
        ood_learned_kernel_eps_param = empty(ood_options.get('l_ood_eps', ood_options.get('leps', ood_options.get('eps', 1e-4))), shape=shared_shape)
        ood_learned_kernel_arc_param = empty(ood_options.get('l_ood_arc_eps', ood_options.get('larc_eps', ood_options.get('arc_eps', 1e-4))), shape=shared_shape)
        ood_learned_kernel_mhe_param = empty(ood_options.get('l_ood_mhe_s', ood_options.get('lmhe_s', ood_options.get('mhe_s', 2.0))), shape=shared_shape)
      else:
        ood_learned_kernel_eps_param = learned_kernel_eps_param
        ood_learned_kernel_arc_param = learned_kernel_arc_param
        ood_learned_kernel_mhe_param = learned_kernel_mhe_param
      
      with torch.no_grad():
        if loaded and load_new:  # load weights from file if applicable
          weights_i = ood_options.get(f'weight', weights[ind])
          weights_ood_i = ood_options.get(f'weight_ood', weights[ind])
        else:
          weights_i = weights[ind]
          weights_ood_i = weights[ind]
        
      # specify weighting of each layer
      weight_param = empty(weights_i.clone())
      weight_ood_param = empty(weights_ood_i.clone())
      
      # create a learned kernel (two layered + proj)
      add_params = []
      if feat_kernel == 'learned' or feat_kernel_last == 'learned':
        def make_learned_kernel(is_ood=False):
          inp = features.shape[-1]
          outp = configs.get('learned_projection', 20)  # some arbitrary dim
          outp_conv = configs.get('learned_projection_conv', 5)
          dim0 = max(inp, outp)
          dim1 = max(int(inp / 1.5), outp)
          dim2 = max(int(inp / 2), outp)
          
          def make_lin(_in, _out):
            linear = nn.Linear(_in, _out)
            linear = spectral_norm_fc(linear, 2.0, n_power_iterations=1)
            return linear
          
          def make_conv(inc, inh, inw, *args, **kwargs):
            conv = nn.Conv2d(inc, *args, **kwargs)
            with torch.no_grad():
              torch.nn.init.kaiming_uniform_(conv.weight)
            conv = spectral_norm_conv(conv, 2.0, (int(inc), int(inh), int(inw)), n_power_iterations=1)
            return conv
          
          # if Conv space then use conv otherwise linear by default
          prefix_layers = [
            make_lin(inp, dim0),
            nn.LeakyReLU(),
            make_lin(dim0, dim1),
            nn.LeakyReLU(),
            make_lin(dim1, dim2),
            nn.LeakyReLU(),
            make_lin(dim2, outp)
          ]
          if configs.get('learned_conv_projection', True):
            if orig_feature is not None:
              if orig_feature.ndim == 5:  # conv layer
                print('Building Conv2D based kernel layer on', orig_feature.shape)
                inpc = orig_feature.shape[2]
                inh, inw = orig_feature.shape[3], orig_feature.shape[4]
                mins = min(inh, inw)
                
                cdim0 = max(inpc, outp_conv)
                cdim1 = max(int(inpc / 2), outp_conv)
                l2 = max(inh / 2 if mins > 8 else inh, 1)  # @TODO support non-square dims
                l3 = max(l2 / 2 if mins > 16 else l2, 1)
                pooled = min(6, l3)
                inp = outp_conv*pooled*pooled
                dim0 = max(inp, outp)
                dim1 = max(int(inp / 1.5), outp)
                dim2 = max(int(inp / 2), outp)
                # dim2 = max(int(inp / 2), outp)
                prefix_layers = [
                  nn.Unflatten(1, orig_feature.shape[2:]),  # reshape back with spatial dims
                  make_conv(inpc, inh, inw, cdim0, 5, 2 if mins > 8 else 1, padding=2),
                  nn.LeakyReLU(),
                  # nn.MaxPool2d(2),
                  make_conv(cdim0, l2, l2, cdim1, 5, 2 if mins > 16 else 1, padding=2),
                  nn.LeakyReLU(),
                  make_conv(cdim1, l3, l3, outp_conv, 5, 2 if mins > 32 else 1, padding=2),
                  nn.AdaptiveAvgPool2d((pooled, pooled)),
                  nn.Flatten(),
                  make_lin(inp, dim0),
                  nn.LeakyReLU(),
                  make_lin(dim0, dim1),
                  nn.LeakyReLU(),
                  make_lin(dim1, outp)
                ]
                
                # project down to some min size 
                # min_space = configs.get('learned_projection_min_space', 8)
                # while min(inh, inw) > min_space:
                #   prefix_layers.extend([
                #     nn.Conv2d(inc, int(inc * 1.75), 3),
                #     nn.LeakyReLU(),
                #     nn.MaxPool2d(2)
                #   ])
                #   inc = int(inc * 1.75)
                #   inh = int(inh / 2)
                #   inw = int(inw / 2)
                
                # global average pool and apply linear on conv
                # prefix_layers.extend([
                #   nn.AdaptiveAvgPool2d((1, 1)),
                #   nn.Flatten()
                # ]),
                # dim0 = inc
                # dim1 = max(int(inc / 1.5), outp)
                # dim2 = max(int(inc / 2), outp)
          
          learned_kernel = nn.Sequential(
            *prefix_layers,
            # (nn.ReLU() if feat_learned_kernel == 'linear' else nn.Identity())
          ).to(features.device)
          
          # load weights from file
          if loaded and load_new:
            key = 'ood_kernel' if is_ood else 'kernel'
            if key in ood_options:
              learned_kernel.load_state_dict(ood_options[key])
          
          return learned_kernel
        
        
        # shared kernel between models
        # or different kernels
        if configs.get('learned_shared', True):
          learned_kernel = make_learned_kernel()
          learned_ood_kernel = learned_kernel
          add_params.extend(list(learned_kernel.parameters()))
        else:
          model_bs = features.shape[0]
          learned_kernel = [make_learned_kernel() for _ in range(model_bs)]
          learned_ood_kernel = learned_kernel
          for k in learned_kernel:
            add_params.extend(list(k.parameters()))
        
        if configs.get('separate_ood_learned_kernel', True):
          if configs.get('learned_shared', True):
            learned_ood_kernel = make_learned_kernel(is_ood=True)
            
            # include learned parameters for hyperenergy based kernel        
            add_params.extend(list(learned_ood_kernel.parameters()))
          else:
            learned_ood_kernel = [make_learned_kernel() for _ in range(model_bs)]
            for k in learned_ood_kernel:
              add_params.extend(list(k.parameters()))
      else:
        learned_kernel = None
        learned_ood_kernel = None
      
      # gamma_param = empty(configs['gamma'])
      # gamma_ood_param = empty(configs['gamma_ood'])
      ood_params.append({
        'eps': eps_param,
        'arc_eps': arc_param,
        'mhe_s': mhe_param,
        'rbf_sigma': rbf_param,
        'ood_eps': ood_eps_param,
        'ood_arc_eps': ood_arc_param,
        'ood_mhe_s': ood_mhe_param,
        'ood_rbf_sigma': ood_rbf_param,
        'leps': learned_kernel_eps_param,
        'larc_eps': learned_kernel_arc_param,
        'lmhe_s': learned_kernel_mhe_param,
        'l_ood_eps': ood_learned_kernel_eps_param,
        'l_ood_arc_eps': ood_learned_kernel_arc_param,
        'l_ood_mhe_s': ood_learned_kernel_mhe_param,
        'weight': weight_param,
        'weight_ood': weight_ood_param,
        'kernel': learned_kernel,
        'ood_kernel': learned_ood_kernel
        
        # 'gamma': gamma_param,
        # 'gamma_ood': gamma_ood_param
      })
      
      add_params += [eps_param, arc_param, mhe_param, learned_kernel_eps_param, learned_kernel_arc_param, learned_kernel_mhe_param, weight_param, weight_ood_param] 
      for p in add_params:
        ood_hyper_params.append(p)
        params_ood.append({
          'params': p,
          'initial_lr': lr,
          'lr': lr * configs.get('ood_lr_mult', 5.0)
        })

  # include wd to relevant params
  hyper_params = list(hyper.parameters())
  for name, param in hyper.named_parameters():
    weight_decay = configs.get('weight_decay', 0.0)
    
    # target specific parameter
    if 'target' in name and configs.get('agc', True):
      weight_decay = target_weight_decay
      
    # reduce learning rate for affine weights/remove weight decay
    if name.endswith('affine_weight') or name.endswith('affine_bias'):
      weight_decay = 0.0
      # lr /= 2
    
    # reduce learning rate/remove weight decay for skip gain residual connections
    if 'skip_gain' in name:
      weight_decay = 0.0
      # lr /= 2
    
    # no weight decay for bias
    if name.endswith('bias'):
      weight_decay = 0.0
    
    # add the group of parametes to optimizer
    params.append({
      'params': param,
      'initial_lr': lr,
      'lr': lr,
      'weight_decay': weight_decay,
      # 'momentum': 0.9
      'decoupled_weight_decay': True  # like in AdamW
    })
  
  if len(params_ood) == 0:
    params_ood = [torch.tensor(0.0, requires_grad=True)]
  
  if configs.get('optim', 'adamw') == 'adamw':
    optim = torch.optim.AdamW(params)
    optim_ood = torch.optim.AdamW(params_ood)
  elif configs.get('optim', 'adamw') == 'nadam':
    optim = torch.optim.NAdam(params)  # torch.optim.AdamW(params)
    optim_ood = torch.optim.NAdam(params_ood)
  elif configs.get('optim', 'adamw') == 'sgd':
    optim = torch.optim.SGD(params, momentum=0.9)
    optim_ood = torch.optim.SGD(params_ood, momentum=0.9)
  else:
    raise ValueError(f'Invalid optimizer {configs.get("optim", None)}')
  
  if loaded and load_new:
    
    optim.load_state_dict(save_data['optim'])
    optim_ood.load_state_dict(save_data['optim_ood'])
  
  # for k, v in hyper.named_parameters():
  #   print(k)
  # optim = SGD_AGC(
  #   hyper.named_parameters(),
  #   lr=configs['lr'],
  #   weight_decay=configs.get('weight_decay', 0.0),
  #   nesterov=True,
  #   momentum=0.9,
  #   clipping=0.1
  # )
  if 'warmup' in configs and configs['warmup'] is not None:
    warmup = CosineWarmupScheduler(
      optim,
      warmup=configs['warmup'],
      max_iters=configs['epochs'] if configs.get('max_iter') is None else configs.get('max_iter'),
    )
    # warmup = CyclicCosAnnealingLR(
    #   optim,
    #   milestones=6,
    #   decay_milestones='same',  # same as milestones
    #   gamma=0.7,
    #   eta_min=1e-9,
    #   last_epoch=configs['epochs'],
    #   warmup=configs['warmup']
    # )
    # warmup = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #   optim,
    #   last_epoch=configs['epochs'],
    #   T_0=configs['warmup'],
    #   eta_min=1e-11,
    #   T_mult=1
    # )
    # warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
    #   optim,
    #   T_max=configs['epochs'],
    #   eta_min=1e-11,
    #   # verbose=True
    # )
    
    # uncomment to test plot lr
    # lrs = []
    # for ep in range(configs['epochs']):
    #   warmup.step(ep)
    #   lrs.append(warmup.get_lr())
    # plt.plot(lrs)
    # plt.savefig('lr.png')
    # exit(0)
  else:
    warmup = None
  
  # multistep LR reduction
  # mstep_lr = torch.optim.lr_scheduler.MultiStepLR(
  #   optim,
  #   milestones=[int(configs['epochs'] / 2), int(configs['epochs'] * 3 / 4)],
  #   gamma=0.1,
  #   last_epoch=configs['epochs'],
  #   verbose=True
  # )
  
  if 'lr_patience' in configs:
    reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optim,
      mode='min',
      factor=0.1,
      patience=configs['lr_patience'],
      verbose=True,
      cooldown=5,
      min_lr=1e-8,
    )
    ood_reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optim_ood,
      mode='min',
      factor=0.1,
      patience=configs['lr_patience'],
      verbose=True,
      cooldown=5,
      min_lr=1e-8,
    )
    
    if loaded and load_new:
      reduce_lr_on_plateau.load_state_dict(save_data['reduce_lr_on_plateau'])
      ood_reduce_lr_on_plateau.load_state_dict(save_data['ood_reduce_lr_on_plateau'])
  else:
    reduce_lr_on_plateau = None
    ood_reduce_lr_on_plateau = None
  
  
  def calc_div(ep, batch_ind, feature_matrices, feature_ood_matrices, parameters, ood_loss):
    """ Calculate diversity if applied """
    
    d_loss = 0.0
    d_ood_loss = 0.0
    ckas = []
    ckas_ood = []
    
    if not configs.get('learn_diversity', False):
      # warmup to gamma over this number of epochs
      gamma = configs['gamma']
      
      # if ood_entropy is None:
      gamma_ood = gamma_ood_init
    else:
      # weights are learned through weight and weight_ood
      gamma = 1.0
      gamma_ood = 1.0
    
    # filter out nones
    if len(feature_matrices) == 0:
      print('WARNING empty feature calc')
      return 0.0, 0.0, 0.0, [], []

    ''' Calculate hyperspherical energy between channels/features of each model
    params = flatten_keys(parameters)
    total = 0.0
    for pname, pvalue in params.items():
      for i in range(pvalue.shape[0]):
        param = pvalue[i]
        # d_loss += 0.5 * mean_hyperspherical_energy(param.reshape(param.shape[0], -1), s=1.0, half_space=False, eps=1e-3, arc_eps=1e-3) / pvalue.shape[0]
      total += 1.0
    # d_loss = d_loss / total
    '''
    l2_loss = 0.0  # removed and added weight decay to target gradients
    '''
    params = flatten_keys(parameters)
    total = 0.0
    for pname, pvalue in params.items():
      if 'affine' in pname or 'bias' in pname:  # ignore shared affine/norm params
        continue
      # l2 norm
      l2_loss += torch.square(pvalue).view(pvalue.shape[0], -1).sum(dim=1).mean()
      # for i in range(pvalue.shape[0]):
      #   param = pvalue[i]
      # d_loss += 0.5 * mean_hyperspherical_energy(param.reshape(param.shape[0], -1), s=1.0, half_space=False, eps=1e-3, arc_eps=1e-3) / pvalue.shape[0]
      total += 1.0
    l2_loss = 1e-5 * (l2_loss / total)
    '''
     
    # print('FEAT', len(feature_matrices), 'OOD FEAT', len(feature_ood_matrices))
    # linearly increase weight to later layers
    
    first = feature_matrices[0]
    # if using function space methods we don't calc internal feats
    if method == 'fs':
      return torch.tensor(0.0, device=first.device), torch.tensor(0.0, device=first.device), torch.tensor(0.0, device=first.device), [], []
      
    # weights = [0.6, 2.0, 3.0, 4.0, 6.0, 8.0]
    # norm = sum([s**2 for s in weights])**(0.5)
    # weights = [w/norm for w in weights]

    # create centering matrix/detacher
    # batch_size = feature_matrices[0].shape[1]
    # model_bs = feature_matrices[0].shape[0]
    # device = feature_matrices[0].device
    # dtype = feature_matrices[0].dtype
    # with torch.no_grad():
    #   H = torch.full(size=(batch_size, batch_size), fill_value=-1.0/batch_size, device=device, dtype=dtype)
    #   H.fill_diagonal_(1.0 - (1.0 / batch_size))

    #   # creates matrix with all 1.0s except on diagonal
    #   # @TODO make this faster by just zeroing out diagonal grad in backward pass
    #   zeroer = torch.ones(batch_size, batch_size, device=device, dtype=dtype)
    #   zeroer.fill_diagonal_(0.0)
    #   zeroer = zeroer.repeat(model_bs, 1, 1)

    # in SVGD we construct kernel matrix
    if svgd:
      sv_K = torch.zeros((model_bs, model_bs), device=first.device, dtype=first.dtype, requires_grad=True)
    else:
      sv_K = None
    
    # print('Feat matrices', len(feature_matrices), len(feature_ood_matrices))
    
    # create moving averages if empty
    if len(moving_avgs) == 0:
      for _ in feature_matrices:
        if configs.get('ma_decay') is not None:
          moving_avg = EMA(decay=configs.get('ma_decay'))
        else:
          moving_avg = None
        moving_avgs.append(moving_avg)
    if len(median_moving_avgs) == 0:
      for _ in feature_matrices:
        if configs.get('median_ma_decay') is not None:  # only used for RBF kernel
          median_moving_avg = EMA(decay=configs.get('median_ma_decay'))
        else:
          median_moving_avg = None
        median_moving_avgs.append(median_moving_avg)

    # sample all models or just a subset
    diverse_sample = configs.get('diverse_sample', None)
    diverse_sample_fixed = configs.get('diverse_sample_fixed', False)  # if True then use same set of models
    if diverse_sample is not None:
      get_model_sample = lambda: torch.randperm(diverse_sample, device=first.device, requires_grad=False)
      if diverse_sample_fixed:
        m_sampled = get_model_sample()  # use same set of models
        def model_sampling(feat, sv=None):
          if sv is None:
            return feat[m_sampled, :, :], None
          return feat[m_sampled, :, :], m_sampled
      else:
        def model_sampling(feat, sv=None):
          if sv is None:
            return feat[get_model_sample(), :, :], None
          sample = get_model_sample()
          return feat[sample, :, :], sample

    for ind, (features, moving_avg, median_moving_avg) in enumerate(zip(feature_matrices, moving_avgs, median_moving_avgs)):
      if diverse_sample is not None:
        if svgd:
          raise NotImplementedError('Cannot use SVGD with model sampling')
        features, sv_ind = model_sampling(features, sv_K)
      
      # reduce feature variance between models
      if configs.get('reduce_var', False):
        mean = moving_avg.get() if moving_avg.get() is not None else features.mean()
        d_loss += 0.00001 * torch.sqrt(torch.sum(torch.square(features - mean.view(1, 1, 1))))
      
      if configs.get('learn_diversity', False):
        l_kernel = ood_params[ind]['kernel']
        l_eps = torch.abs(ood_params[ind]['leps']) + 1e-6
        l_arc = torch.abs(ood_params[ind]['larc_eps']) + 1e-9
        l_mhe_s = torch.abs(ood_params[ind]['lmhe_s'])
        l_rbf_s = torch.abs(ood_params[ind]['rbf_sigma'])
        kwargs = {
          'model': l_kernel,
          'learned_kernel': feat_learned_kernel,
          'eps': l_eps,
          'median': True,  # do not use RBF median
          'param': 1.0,  # l_rbf_s,
          'arc_eps': l_arc,
          'mhe_s': l_mhe_s,
        }
      else:
        kwargs = {}
        
      update_median = batch_ind % configs.get('median_ma_interval', 1) == 0
      grams = batched_gram(
        features, None,  # features,
        kernel=feat_kernel if ind < len(feature_matrices) - 1 else feat_kernel_last,
        detach_diag=configs.get('detach_diag', True),
        readd=configs.get('readd', True),
        center=configs.get('center', True),
        ma=moving_avg, update_ma=True, median_ma=median_moving_avg, median_update=update_median,
        **kwargs
      )
      cka_vals = pairwise_cossim(grams)
      ckas.append(cka_vals.mean())
      # if configs.get('use_pcka', False):
      #   # useful to tests against mhe or cka
      #   mhe = cka_vals.mean()
      # else:
      #   mhe = mean_hyperspherical_energy(grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('eps', 1e-4), offset=configs.get('mhe_offset', 0.0), eps=configs.get('eps', 1e-4))
      # d_loss += weights[ind] * mhe
      
      # normalize by number of models
      # given we do pairwise 
      n_models = first.shape[0]
      
      # @TODO remove this normalization test later. Just for reference
      # does not work given non-linear nature of hyperspherical energy :(
      # upper_right = float(((n_models - 1) * n_models) / 2.0)
      norm_model = 1.0 / (n_models ** 2.0)
      # arc_smooth = 1.0 * norm_model * configs.get('arc_eps', 1e-4)
      # eps_smooth = 1.0 * n_models * configs.get('eps', 1e-3)
      if configs.get('learn_diversity', False):
        eps_smooth = torch.abs(ood_params[ind]['eps']).detach() + 1e-6
        arc_smooth = torch.abs(ood_params[ind]['arc_eps']) + 1e-9
        weight_smooth = torch.abs(ood_params[ind]['weight'])
        mhe_s = torch.abs(ood_params[ind]['mhe_s'])
        print('Layer', ind, 'EPS VAL', eps_smooth.item(), 'ARC VAL', arc_smooth.item(), 'WEIGHT VAL', weight_smooth.item(), 'MHE S', mhe_s.item())
        # gamma = torch.abs(ood_params[ind]['gamma'])
        # gamma_ood = torch.abs(ood_params[ind]['gamma_ood'])
        # print('EPS VAL', eps_smooth.item(), arc_smooth.item(), weight_smooth.item())
      else:
        eps_smooth = configs.get('eps', 1e-4)
        arc_smooth = configs.get('arc_eps', 1e-4)
        mhe_s = configs.get('mhe_s', 2.0)
        weight_smooth = weights[ind]
      
      if svgd:
        if configs.get('use_pcka', False):
          sv_K = sv_K + gamma * weight_smooth * norm_model * torch.abs(pairwise_cossim(grams, reduction=None, detach_right=True))
        elif configs.get('he_raw', False):
          sv_K = sv_K + gamma * weight_smooth * hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=False)  # True)
        else:
          he_vals = hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=False)
          he_vals = torch.abs(he_vals - min_he_energy + 0.1) / (max_he_energy - min_he_energy)  # normalize to 0 and 1
          # print(he_vals)
          sv_K = sv_K + gamma * weight_smooth * he_vals
      else:
        if configs.get('use_pcka', False):
          # useful to tests against mhe or cka
          mhe = cka_vals.mean()
        elif configs.get('he_raw', False):
          mhe = hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True))
        else:
          he_vals = hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none')
          mhe = ((he_vals - min_he_energy) / (max_he_energy - min_he_energy)).mean()  # normalize to 0 and 1
        d_loss += weight_smooth * mhe
      
      # lin_grams = torch.bmm(features, torch.transpose(features, 1, 2))

      # # let's create detached diagonal matrix multiplier
      # diags = lin_grams.diagonal(offset=0, dim1=1, dim2=-1).detach()
      # diag_embed = torch.diag_embed(diags, offset=0, dim1=1, dim2=-1).detach()
      
      # # now fix the gram matrix
      # lin_grams = lin_grams.multiply(zeroer) + diag_embed # zero out diagonals and add detached version

      # # now center gram matrices and vectorize gram matrices
      # cgrams = torch.matmul(lin_grams, H).view(model_bs, batch_size * batch_size)

      # # compute gram frobenius norms (ie euclide of vectorized form)
      # cgram_norms = torch.linalg.vector_norm(cgrams, ord=2, dim=1, keepdim=True)

      # # create unit norm gram matrices
      # ncgrams = cgrams.divide(cgram_norms + 1e-6)

      # # compute pairwise dot products (just upper triangle of inner prod)
      # gram_pd = torch.matmul(ncgrams, ncgrams.t())
      # rows, cols = torch.triu_indices(model_bs, model_bs, offset=1)
      # gram_pd = gram_pd[rows, cols]  # now just a tensor of the isolated pairwise

      # # calculate 1/arccos of these pairs, get the mean and that's our loss
      # # d_loss += (1.0 / (torch.arccos(gram_pd) + 1e-5)).mean()
      # grams = batched_gram(features, features, kernel='linear', detach_diag=True, center=True)
      # d_loss += (1.0 / len(feature_matrices)) * mean_hyperspherical_energy(grams, half_space=False, s=0.5)

    # print(torch.isfinite(d_loss), d_loss)
    # print([float(torch.sum(p).cpu().item()) for p in hyper.parameters()])

    # apply the same for ood if applicable
    if configs.get('separate_ood_features', True) and ((not configs.get('learn_diversity', False)) or (configs.get('learn_diversity', False) and configs.get('learned_use_ood_diversity', True))):
      for ind, (features, moving_avg) in enumerate(zip(feature_ood_matrices, moving_avgs)):
        if features is None:
          continue
        
        if diverse_sample is not None:
          if svgd:
            raise NotImplementedError('Cannot use SVGD with model sampling')
          features, sv_ind = model_sampling(features, sv_K)
        
        if configs.get('learn_diversity', False):
          l_kernel = ood_params[ind]['ood_kernel']
          l_eps = torch.abs(ood_params[ind]['l_ood_eps']) + 1e-6
          l_arc = torch.abs(ood_params[ind]['l_ood_arc_eps']) + 1e-9
          l_mhe_s = torch.abs(ood_params[ind]['l_ood_mhe_s'])
          l_rbf_s = torch.abs(ood_params[ind]['rbf_sigma'])
          kwargs = {
            'model': l_kernel,
            'learned_kernel': feat_learned_kernel,
            'eps': l_eps,
            'median': True,
            'param': 1.0,  # l_rbf_s,
            'arc_eps': l_arc,
            'mhe_s': l_mhe_s
          }
        else:
          kwargs = {}
        
        grams = batched_gram(
          features, None,  # features,
          kernel=feat_kernel if ind < len(feature_matrices) - 1 else feat_kernel_last,
          detach_diag=configs.get('detach_diag', True), readd=configs.get('readd', True),
          center=configs.get('center', True), ma=moving_avg, update_ma=True,
          **kwargs
        )
        cka_vals = pairwise_cossim(grams)
        ckas_ood.append(cka_vals.mean())
        # if configs.get('use_pcka', False):
        #   # useful to tests against mhe or cka
        #   mhe = cka_vals.mean()
        # else:
        #   mhe = mean_hyperspherical_energy(grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('eps', 1e-4), eps=configs.get('eps', 1e-4), offset=configs.get('mhe_offset', 0.0))
        # d_ood_loss += weights[ind] * mhe
        
        if configs.get('learn_diversity', False):
          eps_smooth = torch.abs(ood_params[ind]['ood_eps']).detach() + 1e-6
          arc_smooth = torch.abs(ood_params[ind]['ood_arc_eps']) + 1e-9
          weight_smooth = torch.abs(ood_params[ind]['weight_ood'])
          mhe_s = torch.abs(ood_params[ind]['ood_mhe_s'])
          # gamma_ood = torch.abs(ood_params[ind]['gamma_ood'])
          # print('EPS VAL', eps_smooth.item(), weight_smooth.item())
        else:
          eps_smooth = configs.get('eps', 1e-4)
          arc_smooth = configs.get('arc_smooth', 1e-4)
          mhe_s = configs.get('mhe_s', 2.0)
          weight_smooth = weights[ind]
        
        if svgd:
          if configs.get('use_pcka', False):
            sv_K = sv_K + gamma_ood * weight_smooth * torch.abs(pairwise_cossim(grams, reduction=None, detach_right=True))
          elif configs.get('he_raw', False):
            sv_K = sv_K + gamma_ood * weight_smooth * norm_model * mean_hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=True)  # True)
          else:
            he_vals = mean_hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=False)
            he_vals = torch.abs(he_vals - min_he_energy + 0.1) / (max_he_energy - min_he_energy)  # normalize to 0 and 1
            # print(he_vals)
            sv_K = sv_K + gamma_ood * weight_smooth * he_vals
          # else:
          #   sv_K = sv_K + gamma * weights[ind] * mean_hyperspherical_energy(grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True), reduction='none', remove_diag=False, detach_right=True, abs_vals=True)
        else:
          if configs.get('use_pcka', False):
            # useful to tests against mhe or cka
            mhe = cka_vals.mean()
          elif configs.get('he_raw', False):
            mhe = hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True))
          else:
            he_vals = hyperspherical_energy(grams, half_space=False, s=mhe_s, arc_eps=arc_smooth, eps=eps_smooth, use_exp=configs.get('use_exp', True), reduction='none')
            mhe = ((he_vals - min_he_energy) / (max_he_energy - min_he_energy)).mean()  # normalize to 0 and 1
          # else:
          #   mhe = mean_hyperspherical_energy(grams, half_space=False, s=configs.get('mhe_s', 1.0), arc_eps=configs.get('arc_eps', 1e-4), eps=configs.get('eps', 1e-4), use_exp=configs.get('use_exp', True))
          d_ood_loss += weight_smooth * mhe
    
    # else:
    #   gamma_ood = configs['_gamma_ood']
    #   # exp_entropy = torch.mean(ood_entropy).cpu().item()
    #   # diff = (ood_exp_entropy - exp_entropy)
    #   # gamma_ood = configs['_gamma_ood'] + (gamma_delta * diff)
    #   # configs['_gamma_ood'] = gamma_ood
      
    #   # if configs.get('show_other', True):
    #   #   print('GAMMA OOD', gamma_ood, 'OOD ENTROPY', exp_entropy)
    #   pass
    
    if configs.get('warmup_cka', 0.0) > 0:
      warm = float(min((1e-8 + ep) / float(configs['warmup_cka']), 1.0))
      gamma *= warm
      gamma_ood *= warm
    
    # fix ood to tensor
    if isinstance(d_ood_loss, float):
      # d_ood_loss = torch.tensor(0.0, device=d_loss.device)
      d_ood_loss = torch.tensor(0.0, device=first.device)
    
    # return gamma * d_loss, gamma_ood * d_ood_loss, l2_loss, ckas, ckas_ood
    if svgd:
      d_loss = sv_K  # replace with mat
    else:
      d_loss = gamma * d_loss
    
    return d_loss, gamma_ood * d_ood_loss, l2_loss, ckas, ckas_ood


  ood_N = configs.get('ood_sample', 0)
  print(f'Using {ood_N} OOD samples')
  use_ood = ood_N > 0
  aggregate_track = {}
  total_train_time = 0.0
  start_epoch = 0
  best_test_loss = 1e+10
  applied_alternate = True
  proportion_adverserial = configs.get('prop_adv', 0.2)  # 20% of examples are adversarial
  next_ood = None
  adv_sample = None
  adv_sample_Y = None
  
  if loaded and load_new:
    start_epoch = save_data.get('epoch', 0)
  
  if 'start_epoch' in configs:
    print('Overriding start epoch as', configs['start_epoch'])
    # override
    start_epoch = configs['start_epoch']
  
  for ep in range(start_epoch, configs['epochs']):
    print(f'Running epoch {ep + 1}/{configs["epochs"]}')

    # run training loop
    tracking_train = []
    hyper.train()
    for moving_avg in moving_avgs:
      if moving_avg is not None:
        moving_avg.train()
    tq_train = tqdm(train_loader, desc='Train', total=num_train, colour='green')
    train_time_start = time.time()
    for b, (X, Y) in enumerate(tq_train):
      X_o = X.cuda()
      X_o.requires_grad_(True)  # for adversarial OOD
      Y_o = Y.cuda()

      num_adv = int(proportion_adverserial * X_o.shape[0])
      if adv_sample is not None and num_adv > 0:
        # ensure batch dim stays the same shape
        # other parts of exp need batch size to stay consistent
        if num_adv > adv_sample.shape[0]:
          num_adv = adv_sample.shape[0]
        
        # remove some old from batch and replace with adversarial examples from previous batch
        X_a = torch.concat([
          X_o[:-num_adv],
          adv_sample[:num_adv]
        ], dim=0)
        Y = torch.concat([
          Y_o[:-num_adv],
          adv_sample_Y[:num_adv]
        ], dim=0)
      else:
        X_a = X_o  # no adversarial examples yet. Wait until at least 1 batch goes through
        Y = Y_o

      # combine with OOD
      if use_ood:
        if next_ood is None:
          X = torch.concat([
            X_a,
            ood_sample(ood_N, X_a)
          ], dim=0)
          
          # print('OOD X', X.shape, 'X_o', X_o.shape, 'X_a', X_a.shape, ood_sample(ood_N, X_a).shape, adv_sample.shape)
        else:
          X = torch.concat([
            X_a,
            next_ood.detach()
          ], dim=0)
          ood_N = next_ood.shape[0]  # ensure ood_N is right size for prediction/loss. next_ood is from previous batch
      else:
        X = X_a
      
      # sample a batch of parameters
      sparam = hyper.sample_params(model_bs, device=X.device)  # sample init codes (should change name... @TODO)
      params = hyper.forward_params(sparam)  # feed through param generator/hypernetwork
      
      # count number and quit
      # sf = flatten_keys(params)
      # tot = 0
      # for k, v in sf.items():
      #   print(k, v.shape)
      #   tot += v[0].numel()
      # print('TOTAL', tot)
      # exit(0)
      pred, feat, feat_ood, _ = forward(X, params, ood_N=ood_N)
      pred_all = pred
      if ood_N is not None and ood_N > 0:
        pred_ood = pred[:, -ood_N:]
        pred = pred[:, :-ood_N]
      
      # initialize grad prior with generated params on first time
      if svgd and prior_state is None:
        device = X.device
        prior_state = None
        fparams = flatten_keys(params)
        tparams = 0
        for key, p in fparams.items():
          if p.shape[0] != model_bs:
            print('Key', key, 'param', p.shape, 'model bs', model_bs)
            raise RuntimeError('Wrong param size assumption made. Please adjust exp.py')
          num_param = int(p[0].numel())
          tparams += num_param
        prior_state = torch.distributions.normal.Normal(torch.zeros(tparams).to(device), torch.ones(tparams).to(device) * prior_variance)
        
        # @NOTE
        # prior state for all target parameters
        # make bold assumption about target parameter being scalers right now
        # cannot easily impl a gradient prior to target parameters without a full rewrite
        # on next rewrite consider this to be an option :(
        ssge = SpectralSteinEstimator(0.01, None, 'rbf', device=device)
      
      if configs.get('diversity', True):
        if ood_N > 0:
          ood_loss = ood_loss_fn(model_bs, pred_ood).mean()
        else:
          ood_loss = None
        d_loss, d_ood_loss, l2_loss, ckas, ckas_ood = calc_div(ep, b, feat, feat_ood, params, ood_loss)  # calc_div(ep + ((b / num_train) % 1.0), feat)
        run_diversity = True
      else:
        d_loss, d_ood_loss = torch.tensor(0.0, device=X.device), torch.tensor(0.0, device=X.device)
        ckas = []
        ckas_ood = []
        l2_loss = 0.0
        ood_loss = None
        run_diversity = False
              
      # adversarial 
      optim.zero_grad()
      optim_ood.zero_grad()
      loss = loss_fn(model_bs, X, Y, pred)
      # print('PRE LOSS!', loss)
      
      # if applying a gradient penalty
      if configs.get('grad_penalty', None) is not None:
        # Y_t = Y.repeat(model_bs)  # for all models
        # ce_loss = F.cross_entropy(pred.mean(0), Y, reduction='none')
        
        # grads = torch.autograd.grad(F.log_softmax(pred, dim=-1).sum(), X, create_graph=True, retain_graph=True)[0]
        grads = torch.autograd.grad(pred.sum(), X, create_graph=True, retain_graph=True)[0]
        grads = grads.view(X.shape[0], -1)  # [BS, flat]
        
        # NOTE: learnable grad penalty was a miss :(
        # if configs.get('learn_diversity', False):
        #   grad_penalty = torch.abs(grad_param)
        # else:
        grad_penalty = configs.get('grad_penalty')
        
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.abs(torch.sqrt(torch.sum(torch.square(grads), dim=1) + 1e-12) - 1.0)
        
        # add to loss
        loss += grad_penalty * torch.where(gradients_norm < 0.1, torch.square(gradients_norm), gradients_norm).mean()
      
      # particle gradients for svgd
      if svgd:
        flat_par_params, par_empty_params = flatten_keys(params, include_empty=True) #  do not include shared target parameters
        par_params = list(flat_par_params.values())
        # par_nlp = torch.zeros(model_bs).to(par_params[0].device)
      
      # target network parameters to accumulate grad into
      # include shared parameters (those not controlled by the hypernetwork) in the target definition
      flat_params_accumulate = list(flatten_keys(params).values()) + target_parameters
      
      # only populate target network parameters. Hypernetwork will be backproped through later
      if configs.get('agc', True) or svgd:
        pred.retain_grad()
        
        # function space add prior
        if 'fkde' in method or 'fsge' in method or 'fssge' in method:
          lflat = torch.concat([p.view(model_bs, -1) for p in par_params], dim=1)
          lprior = prior_state.log_prob(lflat).sum()
          loss = loss - lprior
        
        loss.backward(retain_graph=True, inputs=flat_params_accumulate + [X_o])

        # copy score function gradients
        score_func = pred.grad.clone().detach()
        # score_func_d = -torch.stack([F.nll_loss(F.log_softmax(pred[p]), Y) for p in range(model_bs)])
        # score_func = autograd.grad(score_func_d.sum(), pred)[0]
      else:
        # print('RETAIN')
        loss.backward(retain_graph=configs.get('diversity', True), inputs=hyper_params)

      # create adversarial example for next batch
      if num_adv > 0:
        with torch.no_grad():
          num_adv_get = num_adv + 5  # safe get at least a few
          adv_sample = X_o[:num_adv_get] + np.random.uniform(0.05, configs.get('adv_eps', 0.25)) * torch.sign(X_o.grad[:num_adv_get])
          adv_sample_Y = Y_o[:num_adv_get]

      if run_diversity or svgd:
        # save driving force grad
        if svgd:
          # if len(target_parameters) > 0:
          #   raise NotImplementedError('SVGD not implemented for shared parameters')
          
          par_params_grad = [p.grad.clone().detach() for p in par_params]
          par_params_grad_flat = torch.concat(par_params_grad, dim=1) 
          for p in par_params:
            p.grad.zero_()
          
          # use weight space
          if method == 'ws' or method == 'kde-wgd' or method == 'sge-wgd':
            flats = [p.view(model_bs, -1) for p in par_params]
            pset = torch.concat(flats, dim=1)
            
            # this is sv_K
            if kernel != 'rbf':  # this includes learned kernel!
              sv_Ks = d_loss  # computed in calc div
              
              # calculate gradient wrt he/pcka kernel
              sv_Ks.sum().backward(inputs=par_params, retain_graph=True, create_graph=feat_kernel == 'learned')
              
              # get flat grad
              sv_Kgrad = torch.concat([p.grad.view(model_bs, -1) for p in par_params], dim=1)
            else:
              sv_Ks, sv_Kgrad = rbf_gram(pset, pset.detach(), grad_var=pset, detach_diag=False)
            # update gradients with SVGD driving kernel grads
            # can use the learned parameters and kernel for SVGD as well
            if configs.get('learn_diversity', False):
              raise NotImplementedError('Currently learned kernel for SVGD does not work correctly!')
              # to "learn" new parameters
              # we must project models to new space with current grad
              # calculated via diversity loss then estimate how that affects inlier/outlier detection
              # and backprop through projected model back into original for new hyper estimates
              # loss.backward(inputs=hyper_params, retain_graph=T)
              
              if method == 'ws':  # typical SVGD term
                # typical SVGD (adjust ll) term
                grad = torch.matmul(sv_Ks.detach(), -par_params_grad_flat) - sv_Kgrad
              else:
                if 'kde' in method:
                  grad = -par_params_grad_flat*model_bs - sv_Kgrad / sv_Ks.sum(1, keepdim=True)
                elif 'sge' in method:
                  eta = 0.001
                  K_ = sv_Ks+eta*torch.eye(model_bs).to(sv_Ks.device)
                  grad = -par_params_grad_flat - torch.linalg.solve(K_, sv_Kgrad)
                else:
                  raise NotImplementedError('method not implemented')
              
              # move model via step
              move_params = OrderedDict()
              offset = 0
              step_size = 1.0
              for (k, p), g, f in zip(flat_par_params.items(), par_params_grad, flats):
                move_params[k] = p + step_size*grad[:, offset:offset + f.shape[1]]
                offset += f.shape[1]
              
              # forward with new model
              # now include params with no values
              move_params.update(par_empty_params)
              params_proj = unflatten_keys(move_params)
              
              # forward through new params
              pred_proj, _, _, _ = forward(X, params_proj, ood_N=ood_N)
              pred_all_proj = pred_proj
              if ood_N is not None and ood_N > 0:
                pred_ood_proj = pred_proj[:, -ood_N:]
                pred_proj = pred_proj[:, :-ood_N]
              
              # calculate loss on projected model
              loss_learn = (loss_fn(model_bs, X, Y, pred_proj).mean() + configs.get('learn_ood_weight', 1.0) * ood_loss_fn(model_bs, pred_ood_proj).mean()) / 2.0
              loss_learn.backward(inputs=ood_hyper_params, retain_graph=True)
            
              # apply normal backward through all particles after learned parameters
              offset = 0
              for p, g, f in zip(par_params, par_params_grad, flats):
                p.backward(inputs=hyper_params, gradient=-grad[:, offset:offset + f.shape[1]].view_as(p.grad))
                offset += f.shape[1]
            else:
              # normal SVGD no gradients for learned kernel needed
              with torch.no_grad():
                if method == 'ws':  # typical SVGD term
                  # typical SVGD (adjust ll) term
                  print(sv_Ks.shape, par_params_grad_flat.shape)
                  grad = torch.matmul(sv_Ks.detach(), -par_params_grad_flat) - sv_Kgrad
                else:
                  if 'kde' in method:
                    grad = -par_params_grad_flat*model_bs - sv_Kgrad / sv_Ks.sum(1, keepdim=True)
                  elif 'sge' in method:
                    eta = 0.001
                    K_ = sv_Ks+eta*torch.eye(model_bs).to(sv_Ks.device)
                    grad = -par_params_grad_flat - torch.linalg.solve(K_, sv_Kgrad)
                  else:
                    raise NotImplementedError('method not implemented')
                
                # backward flat grad through particles
                offset = 0
                for p, g, f in zip(par_params, par_params_grad, flats):
                  p.backward(gradient=-grad[:, offset:offset + f.shape[1]].view_as(p.grad))
                  offset += f.shape[1]
          elif method == 'fs' or method == 'fkde-wgd' or method == 'fsge-wgd' or method == 'fssge-wgd':  # function space kernel
            # if kernel == 'he':
            #   raise NotImplementedError('HE requires weight space input not function space input')

            # calculate repuslive
            pred_k = pred.reshape(pred.shape[0], -1)
            
            # single rbf between particles
            # kern = batched_gram(pred_k.unsqueeze(0), pred_k.unsqueeze(0).detach(), kernel=kernel, detach_diag=False, center=False)[0]
            
            # # calculate repuslive gradients in function space
            # grad_kern = autograd.grad(kern.sum(), pred_k)[0]
            if 'ssge' not in method:
              if kernel == 'he':
                raise NotImplementedError('Cannot impl this')
              #   sv_Ks = d_loss.sum()
              #   sv_Ks.backward(inputs=flat_params_accumulate, retain_graph=True)
              # else:  # rbf kernel otherwise
              else:
                kern, grad_kern = rbf_gram(pset, pset.detach(), grad_var=pset, detach_diag=False)
            # gradient functional prior (normal prior)
            rparam = hyper.sample_random_params(model_bs, device=X.device)
            rparams = hyper.forward_params(rparam, device=X.device)  # feed through param generator/hypernetwork
            rpred, rfeat, rfeat_ood, _ = forward(X, rparams, ood_N=ood_N)
            
            rpred_all = pred
            if ood_N is not None and ood_N > 0:
              rpred_ood = rpred[:, -ood_N:]
              rpred = rpred[:, :-ood_N]
            
            # compute gradient prior
            grad_prior = ssge.compute_score_gradients(pred.reshape(model_bs, -1), rpred.reshape(model_bs, -1))
            
            # driving force 
            # print(kern.shape, score_func.shape, grad_prior.shape)
            if 'kde' in method or 'sge' in method or 'ssge' in method:
              drive = score_func.reshape(model_bs, -1) - grad_prior/model_bs
            else:
              drive = kern.matmul(score_func.reshape(model_bs, -1) - grad_prior/model_bs)  #  -rprior_grad.reshape(model_bs, -1) + grad_prior)  # SVGD otherwise
            
            if 'kde' in method:
              grad_kern = grad_kern / kern.sum(1, keepdim=True)
            elif 'sge' in method:
              eta = 0.01
              K_ = kern+eta*torch.eye(model_bs).to(kern.device)
              grad_kern = torch.linalg.solve(K_, grad_kern)
            
            # gradient
            grad = (drive - grad_kern).reshape(pred.shape)

            # calculate jacob to weight space and run backward
            pred.backward(gradient=grad)
          # elif 'wgd' in method:
          #   # adapted from: https://github.com/ratschlab/repulsive_ensembles/blob/master/methods/WGD.py
          #   # apply diversity kernel
          #   # with torch.no_grad():
          #   if 'kde' in method or 'sge' in method:
          #     flats = [p.view(model_bs, -1) for p in par_params]
          #     pset = torch.concat(flats, dim=1)
              
          #     if kernel == 'he':
          #       sv_Ks = d_loss.sum()
          #       sv_Ks.backward(inputs=flat_params_accumulate, retain_graph=True)
                
          #       # construct gradient of kernel
          #       sv_Kgrad = torch.concat([p.grad.view(model_bs, -1) for p in par_params], dim=1)
          #     else:
          #       # with torch.no_grad():
          #       # sv_Ks, sv_Kgrad = rbf_fn(pset, pset.detach())
          #       sv_Ks, sv_Kgrad = RBF()(pset, pset.detach(), grad_var=pset)
          #       # sv_Kgrad = sv_Kgrad.sum(0)
              
          #     with torch.no_grad():
          #       if 'kde' in method:
          #         sv_Kgrad = sv_Kgrad # / sv_Ks.sum(1, keepdim=True)
          #       elif 'sge' in method:
          #         eta = 0.01
          #         K_ = sv_Ks+eta*torch.eye(model_bs).to(sv_Ks.device)
          #         sv_Kgrad = torch.linalg.solve(K_, sv_Kgrad)
          #       else:
          #         raise NotImplementedError('method not implemented')
                
          #       # update particle params with right grad
          #       offset = 0
          #       for f, p, g in zip(flats, par_params, par_params_grad):
          #         grad = g - sv_Kgrad[:, offset:offset + f.shape[1]].view_as(p.grad)
          #         p.backward(gradient=grad)
          #         offset += f.shape[1]
          # else:
          #   raise NotImplementedError('method not implemeneted')
          else:
            raise NotImplementedError('Invalid method applied')
          
            # UNORM POST
            # ll = -loss*self.num_train / self.config.pred_dist_std ** 2

            # if particles is None:
            #     particles = self.ensemble.particles

            # if self.add_prior:
            #     log_prob = torch.add(self.prior.log_prob(particles).sum(1), ll)
            # else:
            #     log_prob = ll
          
          # back if not using agc on weight space methods. Function space handle independently
          # if method != 'fs' and method != 'fkde-wgd' and method != 'fsge-wgd' and method != 'fssge-wgd':
          #   if not configs.get('agc', True):
          #     for p in par_params:
          #       p.backward(gradient=p.grad)
        else:
          loss = loss + (d_loss if not d_loss.isnan() else 0.0) + (d_ood_loss if not d_ood_loss.isnan() else 0.0) + l2_loss
          # print('OOD', ood_loss)
          if configs.get('agc', True):
            if configs.get('learn_diversity', False):
              raise RuntimeError('Cannot use adaptive gradients with learned diversity paramters')
            
            loss.backward(inputs=flat_params_accumulate, retain_graph=True)
          else:
            if configs.get('learn_diversity', False):
              # to "learn" new parameters
              # we must project models to new space with current grad
              # calculated via diversity loss then estimate how that affects inlier/outlier detection
              # and backprop through projected model back into original for new hyper estimates
              # loss.backward(inputs=hyper_params, retain_graph=T)
              
              step = ep + b
              do_param = step % configs.get('learn_diversity_every', 1) == 0
              
              # rollout num
              rollouts = configs.get('learn_diversity_rollouts', 1)
              if rollouts > 1:
                raise NotImplementedError('Rollouts above 1 not implemented yet')
              
              # batch sampler for rollouts
              # rollout_iter = iter(train_loader) @TODO
              
              while do_param and rollouts > 0:
                # get dir (we save for later anyways)
                flat_params, empty_params = flatten_keys(params, include_empty=True)
                loss.backward(inputs=flat_params.values(), create_graph=True, retain_graph=True)
                
                # flatt
                step_size = configs.get('learned_kernel_rollout_step_size', 0.1)
                for key, p in flat_params.items():
                  if p.grad is not None:
                    flat_params[key] = p - (step_size*p.grad)
                  else:
                    print('WARNING: Unused parameter!', key)
                
                # now include params with no values
                flat_params.update(empty_params)
                params_proj = unflatten_keys(flat_params)
                
                # pull next batch @TODO FINISH MULTIPLE ROLLOUTS
                # try:
                #   rX, rY = next(rollout_iter)
                # except StopIteration:
                #   rollout_iter = iter(train_loader)
                #   rX, rY = next(rollout_iter)
                # rX = rX.cuda()
                # rY = rY.cuda()
                
                # forward through new params
                pred_proj, _, _, _ = forward(X, params_proj, ood_N=ood_N)
                pred_all_proj = pred_proj
                if ood_N is not None and ood_N > 0:
                  pred_ood_proj = pred_proj[:, -ood_N:]
                  pred_proj = pred_proj[:, :-ood_N]
                rollouts -= 1
            
            # at the end of the rollouts perform apply the added ood beta loss
            # then apply the gradients to the hypernetwork/ensemble
            if do_param:
              # do we alternate loss?
              apply_hyper = True
              apply_normal = True
              if configs.get('learn_diversity_alternate', False):  # alternate between optimizing objectives
                apply_hyper = applied_alternate
                apply_normal = not applied_alternate
                applied_alternate = not applied_alternate
                
              # calculate loss on projected model
              if apply_hyper:
                loss_learn = (loss_fn(model_bs, X, Y, pred_proj).mean() + configs.get('learn_ood_weight', 1.0) * ood_loss_fn(model_bs, pred_ood_proj).mean()) / 2.0
                loss_learn.backward(inputs=ood_hyper_params, retain_graph=True)
              
              # incldue ood loss after loss learn
              if apply_normal:
                if ood_loss is not None and configs.get('ood_beta', 1.0) != 0.0:
                  loss = loss + (configs.get('ood_beta', 1.0) * ood_loss.mean())
                loss.backward(inputs=hyper_params)
            
            # normal backward
            if not do_param:
              if ood_loss is not None and configs.get('ood_beta', 1.0) != 0.0:
                loss = loss + (configs.get('ood_beta', 1.0) * ood_loss.mean())
              loss.backward(inputs=hyper_params)  # normal backward
      
      # clip gradients
      try:
        if configs.get('agc', True): # apply adaptive gradient clipping
          # on target network parameters
          flat_params = flatten_keys(params)
          batched_dim = next(iter(flat_params.values())).shape[0]
          eps = torch.tensor(1e-4, device=X.device)
          clipping = configs.get('clip', 0.01)
          
          # construct full concatenated param vector to apply grad to
          # doing this so I don't do multiple .backward() calls for hypernetwork but rather just through this concatenated
          # theta vector
          full_params = torch.cat([p.ravel() for p in flat_params.values() if p.grad is not None], dim=0)
          full_grad = []
          
          # first apply AGC on target network parameters
          # we have to take the theta vectors from hypernetwork and unflatten them
          # into their paramter groups (ex: weight and bias) and apply AGC with unitwise norms
          # to each. Note the batched version just does the same as normal AGC but with batched theta
          with torch.no_grad():
            for name, target_flat_param in flat_params.items():
              if target_flat_param.grad is None:
                continue
              
              # unpack flattend view of parameters into original shapes
              target_params_shapes = param_shapes[name]
              
              target_flat_grad = target_flat_param.grad.detach()
              target_flat_param.grad = None  # remove orig grad
              
              # @TODO apply this more generally
              if not ('fc.self' in name):
                for param_name, target_shape, from_index, to_index in target_params_shapes:
                  # get section of parameter to calculate unitwise norms on
                  target_param = target_flat_param[:, from_index:to_index].view(batched_dim, *target_shape)
                  target_grad = target_flat_grad[:, from_index:to_index].view(batched_dim, *target_shape)

                  # calculate parameter and gradient norms / max allowed norm
                  param_norm = torch.max(batched_unitwise_norm(target_param), eps)
                  grad_norm = torch.max(batched_unitwise_norm(target_grad), eps)
                  max_norm = (param_norm * clipping)
                  
                  # scale gradients
                  clipped = (target_grad * (max_norm / grad_norm))
                  
                  # update flattened parameters gradients with clipped version
                  target_grad = torch.where(
                    (grad_norm > max_norm),  # if current grad norm exceeds max
                    clipped,                 # apply clipped version
                    target_grad              # o.t keep original
                  )
                  
                  # apply weight decay here if applicable
                  if target_weight_decay != 0.0 and 'bias' not in param_name:
                    target_grad = target_grad + (target_weight_decay * target_param)
                  
                  # update flattened view
                  target_flat_grad[:, from_index:to_index] = target_grad.view(batched_dim, -1)
                
              # add to full gradient list
              full_grad.append(target_flat_grad.ravel())
              
          # continue pushing gradients through hypernetwork
          full_params.backward(
            gradient=torch.concat(full_grad, dim=0)
          )
        
          # now apply AGC on the hypernetwork
          with torch.no_grad():
            for name, param in hyper.named_parameters():
              
              # ignore ensemble parameters since those are handled already above
              # @TODO make this not ugly
              if param.grad is None or 'self.fixed_parameters' in name or 'gamma' in name or 'bias' in name:
                continue
              
              param_norm = torch.max(unitwise_norm(param), eps)
              grad_norm = torch.max(unitwise_norm(param.grad.detach()), eps)
              max_norm = (param_norm * clipping)
              clipped = (param.grad * (max_norm / grad_norm))
              param.grad.data.copy_(torch.where(
                (grad_norm > max_norm),  # if current grad norm exceeds max
                clipped,                 # apply clipped version
                param.grad               # o.t keep original
              ))
        else:
          # use typical clipping
          if configs.get('clip') is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(hyper.parameters(), configs.get('clip', NORM_CLIP), norm_type=2, error_if_nonfinite=True)
        
        # take step on success
        optim.step()
        optim_ood.step()
        
        # apply schedulers
        if warmup is not None:
          warmup.step(ep + (b / num_train))
      except RuntimeError as err:
        traceback.print_exc()
        print(f'Clip error! {str(err)}')

      # do logging
      with torch.no_grad():
        _to_log = metric_fn(model_bs, X, Y, pred)
        to_log = {}
        if _to_log is not None:
          for k, v in _to_log.items():
            to_log[f'{k}/train'] = v
        
        to_log.update({
          'loss/train': loss.item()
        })
        for ind, c in enumerate(ckas):
          to_log[f'test/cka_{ind + 1}/train'] = c.item()
        for ind, c in enumerate(ckas_ood):
          to_log[f'cka_ood_{ind + 1}/train'] = c.item()
        log(to_log)
        tracking_train.append(to_log)

        # update tqdm desc
        others = ', '.join([('%s: %.3f' % (k, float(v))) for k, v in to_log.items()])
        # description = others + '\n'  # f'L {loss.item()} ' +
        if configs.get('show_other', True):
          print(others)
        tq_train.set_description(f'L {loss.item()}')
    train_time_end = time.time() - train_time_start
    total_train_time += train_time_end
    print(f'Finished training in {train_time_end}s. Total train time {total_train_time}s')
    
    # run testing loop
    tracking_test = []
    hyper.eval()
    for moving_avg in moving_avgs:
      if moving_avg is not None:
        moving_avg.eval()
    t_loss = 0.0
    with torch.no_grad():
      agg_test = []
      tq_test = tqdm(test_loader, desc='Test', total=num_test, colour='blue')
      for b, (X, Y) in enumerate(tq_test):
        X = X.cuda()
        Y = Y.cuda()
        
        # sample some parameters
        sparam = hyper.sample_params(model_bs, device=X.device)
        params = hyper.forward_params(sparam)
        pred, feat, feat_ood, params = forward(X, params, mbs=model_bs, ood_N=0)
        # pred_all = pred
        # if ood_N is not None and ood_N > 0:
        #   pred_ood = pred[:, -ood_N:]
        #   pred = pred[:, :-ood_N]
        d_loss, d_ood_loss, l2_loss, ckas, ckas_ood = calc_div(ep, b, feat, feat_ood, params, ood_loss)
        # pred = pred.reshape(pred.shape[0]*pred.shape[1], -1)
        # loss = loss_fn(model_bs, X, Y, pred) + (d_loss if not d_loss.isnan() else 0.0) + (d_ood_loss if not d_ood_loss.isnan() else 0.0) + l2_loss
        if svgd:
          # print(pred.shape, Y.shape, X.shape)
          loss = loss_fn(model_bs, X, Y, pred) + l2_loss
        else:
          loss = loss_fn(model_bs, X, Y, pred) + (d_loss if not d_loss.isnan() else 0.0) + (d_ood_loss if not d_ood_loss.isnan() else 0.0) + l2_loss
        t_loss += loss.item() / num_test

        # do logging
        _to_log = metric_fn(model_bs, X, Y, pred)
        to_log = {}
        if _to_log is not None:
          for k, v in _to_log.items():
            to_log[f'{k}/val'] = v
        
        to_log.update({
          'loss/val': loss.item()
        })
        for ind, c in enumerate(ckas):
          to_log[f'cka_{ind + 1}/val'] = c.item()
        for ind, c in enumerate(ckas_ood):
          to_log[f'cka_ood_{ind + 1}/val'] = c.item()
        agg_test.append(to_log)

        # update tqdm desc
        others = ', '.join([('%s: %.3f' % (k, float(v))) for k, v in to_log.items()])
        # description = others + '\n'  # f'L {loss.item()} ' +
        if configs.get('show_other', True):
          print(others)

        # update tqdm desc
        description = f'L {loss.item()}'
        tq_test.set_description(description)
    
    # checkpointing
    if t_loss < best_test_loss:
      print('New best loss', t_loss)
      
      if weight_file is not None and save_weight:
        print('Saving weights to', weight_file)
        if ood_params is not None and len(ood_params) > 0:
          try:
            save_ood = copy.deepcopy(ood_params)
            for i in range(len(save_ood)):
              save_ood[i]['kernel'] = ood_params[i]['kernel'].state_dict()
              save_ood[i]['ood_kernel'] = ood_params[i]['ood_kernel'].state_dict()
          except Exception as err:
            print('ERROR: Failed to save ood params!', err)
            save_ood = {}
        else:
          save_ood = {}
        to_save = {
          'epoch': ep,
          'hyper': hyper.state_dict(),
          'ood_params': save_ood,
          'optim': optim.state_dict(),
          'optim_ood': optim_ood.state_dict()
        }
        
        if reduce_lr_on_plateau is not None:
          to_save.update({
            'reduce_lr_on_plateau': reduce_lr_on_plateau.state_dict(),
            'ood_reduce_lr_on_plateau': ood_reduce_lr_on_plateau.state_dict()
          })
        
        torch.save(
          to_save,
          weight_file
        )
      best_test_loss = t_loss
    
    # update tracking to test
    new_log = {}
    for k, v in to_log.items():
      if isinstance(v, (torch.Tensor, float, int)):
        new_log[k] = torch.mean(torch.tensor([h[k] for h in agg_test])).cpu()
      new_log[k] = v
    log(new_log)
    tracking_test.append(to_log)

    # run epoch end code
    epoch_end(hyper, ep, tracking_train, tracking_test, aggregate_track, train_time_end, total_train_time, log)
    
    # mstep_lr.step(ep)
    if reduce_lr_on_plateau is not None:
      reduce_lr_on_plateau.step(t_loss)
      ood_reduce_lr_on_plateau.step(t_loss)
    
  if weight_file is not None and save_weight:
    print('Saving final weights to', weight_file)
    try:
      save_ood = copy.deepcopy(ood_params)
      for i in range(len(save_ood)):
        save_ood[i]['kernel'] = ood_params[i]['kernel'].state_dict()
        save_ood[i]['ood_kernel'] = ood_params[i]['ood_kernel'].state_dict()
    except Exception as err:
      print('ERROR: Failed to save ood params!', err)
      save_ood = {}
    to_save = {
      'epoch': ep,
      'hyper': hyper.state_dict(),
      'ood_params': save_ood,
      'optim': optim.state_dict(),
      'optim_ood': optim_ood.state_dict()
    }
    
    if reduce_lr_on_plateau is not None:
      to_save.update({
        'reduce_lr_on_plateau': reduce_lr_on_plateau.state_dict(),
        'ood_reduce_lr_on_plateau': ood_reduce_lr_on_plateau.state_dict()
      })
      
    torch.save(to_save,
      weight_file
    )

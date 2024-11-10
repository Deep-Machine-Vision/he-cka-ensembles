""" Functions useful for training """
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union, Callable
import random
import time
import copy
import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from hyper.util.collections import flatten_keys, unflatten_keys
from hyper.diversity.methods import ParticleMethods
from hyper.generators.base import LayerCodeModelGenerator
from hyper.experiments.metrics import Metric, Callback
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


class HyperTrainer(nn.Module):
  def __init__(self,
      data: Tuple[object, object],
      hyper: LayerCodeModelGenerator,
      method: ParticleMethods, 
      metrics: nn.ModuleList=None,
      callbacks: nn.ModuleList=None,
      output_dir: str=None,
      optim: str='adamw',
      optim_args: Dict[str, Any]=None,
      ood_N: int=0,
      scheduler: int=None,
      schedule_epoch: bool=False,
      max_iter: int=None,
      clip: float=None,
      wd_affine: bool=False,
      lr_patience: int=None,
      learn_hyper: bool=False,
      learn_hyper_rollout: int=1,
      learn_hyper_loss=None,
      adversarial: Union[float, int]=0,  # not implemented
      log: Callable=print, 
      device: str='cuda',
      device_id=None,
      ddp_reload_every: int=None,
      **configs: Dict[str, Any],
    ) -> None:
    """ Initialize the trainer object

    Args:
        data (Tuple[object, object]): tuple of train and test data loaders
        hyper (LayerCodeModelGenerator): The hyper model generator
        method (ParticleMethods): The particle method to use. Such as SVGD, CKA, or HE
        metrics (nn.ModuleList, optional): List of metrics to use. Defaults to None.
        callbacks (nn.ModuleList, optional): List of callbacks to use. Defaults to None.
        output_dir (str, optional): Output directory to save logs and weights. Defaults to None.
        optim (str, optional): The optimizer to use. Defaults to 'adamw'.
        ood_N (int, optional): Number of out of distribution samples. Defaults to 0.
        scheduler (int, optional): The lr scheduler to use. Defaults to None.
        schedule_epoch (bool). If lr schedule is defined then call scheduler every epoch (True) or every step (False)
        max_iter (int, optional): Maximum number of iterations or epochs for lr scheduler. Notation is weird check the code for more details. Defaults to None ie number of epochs.
        clip (float, optional): Gradient norm clipping value. Defaults to None.
        wd_affine (bool): Apply weight decay to affine and bias values. Default is False.
        lr_patience (int, optional): Number of epochs to wait before reducing learning rate. Defaults to None.
        learn_hyper (bool, optional): Whether to learn the hyper model parameters using a rollout. Defaults to False.
        learn_hyper_rollout (int, optional): Number of rollouts to learn the hyper model. Defaults to 0.
        learn_hyper_loss ([type], optional): Loss function to use for learning the particle method parameters. Defaults to None.
        adversarial (int, optional): Number of adversarial samples. Defaults to 0.
        adversarial_loader (object, optional): Adversarial data loader. Defaults to None.
        configs (Dict[str, Any], optional): Other configurations. Defaults to {}.
        ddp_reload_every (int, optional): When set to an integer will create a barrier and force reloading of weights every n epochs to ensure all models are insync.
        log (Callable, optional): Logging function that accepts a dictionary of key vals. Defaults to null_fn.
    """
    super(HyperTrainer, self).__init__()
    
    if len(data) == 3:
      self.train_loader, self.test_loader, self.ood_loader = data
    else:
      self.train_loader, self.test_loader = data
      self.ood_loader = None
    self.num_train, self.num_test = len(self.train_loader), len(self.test_loader)
    self.hyper = hyper
    self.optim_name = optim
    self.optim_args = optim_args
    self.particle_method = method
    self.model_bs = method.num
    self.output_dir = output_dir
    self.ood_N = ood_N
    self.scheduler_func = scheduler
    self.schedule_epoch = schedule_epoch
    self.clip = clip
    self.wd_affine = wd_affine
    self.lr_patience = lr_patience
    self.learn_hyper = learn_hyper
    self.learn_hyper_rollout = learn_hyper_rollout
    self.learn_hyper_loss = learn_hyper_loss
    self.configs = configs
    self.metrics = [] if metrics is None else metrics
    self.callbacks = [] if callbacks is None else callbacks
    self.max_iter = max_iter
    self.device = device
    self.device_id = device_id if device_id is not None else device
    self.ddp_reload_every = ddp_reload_every
    self.log = log
    self.ddp = False
    
    if self.learn_hyper and self.learn_hyper_rollout > 1:
      raise NotImplementedError("Rollouts above 1 not yet implemented")
    
    # create output if not exists
    if output_dir is not None and not os.path.exists(output_dir) and (self.device_id is None or self.device_id == 'cuda' or self.device_id == 0 or self.device_id == 'cuda:0'):
      os.makedirs(output_dir)
    
    if adversarial > 0:
      raise NotImplementedError("Adversarial training not yet implemented")
    self.prop_adversarial = adversarial

    # tracked training variables
    self.adversarial_batch = None
    self.num_adv = 0
    self.rank = 0
    self.ood_epoch = 0
    self.ood_iter = None
    self.optim_hyper = None
    self.optim_method = None
    self.lr_scheduler = None
    self.rlop = None
    self.rlop_method = None
 
  def output_file(self, name: str):
    """ Get the output file """
    if self.output_dir is None:
      return name
    return os.path.join(self.output_dir, name)
 
  def append_ood(self, X):
    """ Appends the ood samples to the input if applicable

    Args:
        X: input batch to the model
    """
    if self.ood_loader is None:
      return X  # no ood loader
    
    # init ood loader
    if self.ood_iter is None:
      # ood sampler can be out of sync with id sampler
      # requires keeping track of an ood epoch variable
      if self.ood_loader is not None and self.ddp:
        self.ood_loader.sampler.set_epoch(self.ood_epoch)
      
      self.ood_iter = iter(self.ood_loader) if self.ood_loader is not None else None
 
    # if ood sampler provide ood batch
    try:
      item = next(self.ood_iter)
    except StopIteration:  # end of ood loader
      self.ood_epoch += 1
      if self.ood_loader is not None and self.ddp:
        self.ood_loader.sampler.set_epoch(self.ood_epoch)  # update to new epoch number
      self.ood_iter = iter(self.ood_loader)
      item = next(self.ood_iter)
    
    if isinstance(item, (list, tuple)):
      item = item[0]
    X_ood = item.to(self.device_id)
    
    # ensure ood_N is satisfied
    assert X_ood.shape[0] == self.ood_N, 'Incorrect OOD samples provided. Expected %d, got %d' % (self.ood_N, X_ood.shape[0])

    # append ood samples to input
    X = torch.cat([X, X_ood], dim=0)
    return X
 
  def train_single_epoch(self, epoch: int, epochs: int):
    """ Train the model for a single epoch """
    if self.rank == 0:
      print(f'Running train on epoch {epoch}/{epochs}')
    
    # gets the shared target parameters
    module = self.hyper.module if self.ddp else self.hyper
    shared_parameters = list(module.target_parameters())
    all_parameters = list(self.hyper.parameters()) + shared_parameters
    self.hyper.train()
    self.particle_method.train()
    
    if self.ddp:
      self.train_loader.sampler.set_epoch(epoch - 1)
    
    if self.rank == 0:    
      tq_train = tqdm(self.train_loader, desc='Train', total=self.num_train, colour='green')
    else:
      tq_train = self.train_loader  # other ranks just load/do not print
    train_time_start = time.time()
    
    # prepare rollout iterators
    # if self.learn_hyper and self.learn_hyper_rollout > 0:
    #   train_rollouts = iter(self.train_loader)
    
    # reset matric trackers
    with torch.no_grad():
      for metric in self.metrics:
        metric.reset()
    
    # loop through batches
    for b, (X, Y) in enumerate(tq_train):
      X = X.to(self.device_id)
      Y = Y.to(self.device_id)      
      X = self.append_ood(X)
      tot_step = ((epoch-1)*self.num_train) + b
        
      # handle the method start
      self.particle_method.batch_start(self, X, Y)
      
      # sample model_bs models and run through hypernetwork
      track = self.hyper(
        params=self.model_bs,
        x=X,
        sample_params=True,  # when True then params argument is now the number of models we want to sample not sample codes
        ret_params=True,  # include the generated hypernetwork parameters into the tracking dictionary
        feature_split=True,   # split inlier and outlier samples and create tracking dictionary
        
        # all other arguments are fed through hyper.forward_split(...)
        ood_N=self.ood_N,
        skip_empty=True,
        split_pred_only=not self.configs.get('separate_ood_features', True)
      )
      params = track['params']
      p2 = OrderedDict()
      for k, v in flatten_keys(params).items():
        if isinstance(v, torch.Tensor):
          p2[k] = v.retain_grad()
        p2[k] =v 
      _, em = flatten_keys(params, include_empty=True)
      em.update(p2)
      params = unflatten_keys(em)
      
      # push through method and get loss/forward data
      method_track = self.particle_method.forward(
        trainer=self,
        params=params,
        shared_params=shared_parameters,
        track=track,
        x=X, gt=Y, step=tot_step,
        ignore_ood_loss=self.learn_hyper  # we apply this later during rollouts
      )
      
      # keep track of params for rollouts if applicable
      if self.learn_hyper and self.learn_hyper_rollout > 0:
        flat_params, empty_params = flatten_keys(params, include_empty=True)
      
      # calculate current loss and backward
      # possibly ignoring the ood loss
      loss, logs = self.particle_method.backward(
        trainer=self,
        params=params,
        shared_params=shared_parameters,
        track=method_track,
        x=X, gt=Y,
        retain_graph=self.learn_hyper,
        create_graph=self.learn_hyper,
        inputs=flat_params.values() if self.learn_hyper else all_parameters,  # possibly only update the param grads
      )
      
      # handle learned rollouts if applicable
      if self.learn_hyper and self.learn_hyper_rollout > 0:
        if self.ddp:
          raise RuntimeError('Learned method parameters/rollouts do not support DDP yet')
        
        step_size = self.configs.get('rollout_step_size', 0.1)
        for key, p in flat_params.items():
          if p.grad is not None:
            flat_params[key] = p - (step_size*p.grad)
          else:
            print('WARNING: Unused parameter!', key)

        # forward through the target networks
        flat_params.update(empty_params)
        params_proj = unflatten_keys(flat_params)
        proj_track = self.hyper.forward_split(
          params_proj, X,
          ood_N=self.ood_N,
          skip_empty=True,
          split_pred_only=not self.configs.get('separate_ood_features', True)
        )
        
        # calculate the loss for the projection
        proj_loss = self.learn_hyper_loss(self.model_bs, X, proj_track, Y)
        proj_loss.backward(inputs=self.params_method, retain_graph=True)
        
        # now we update all the parameters
        # however we include any ood losses now
        if self.particle_method.ood_loss is not None:
          proj_method_track = self.particle_method.forward_ood(
            trainer=self,
            params=flat_params,
            shared_params=shared_parameters,
            track=proj_track,
            x=X, gt=Y
          )
          method_track['loss'] += proj_method_track['loss']
        
        # now push all the way back including hypernetwork/ensemble parameters
        self.particle_method.backward(
          trainer=self,
          params=flat_params,
          shared_params=shared_parameters,
          track=method_track,
          x=X, gt=Y,
          retain_graph=False,
          create_graph=False,
          inputs=all_parameters  # all except the rollout/learnable hyperparameters
        )
      
      # handle optim stepping/etc
      self.particle_method.batch_end(self, X, Y)
      
      # make sure synced to this point
      # if self.ddp:
      #   dist.barrier()
    
      # apply schedulers with smoothing
      if not self.schedule_epoch:  # we apply scheduler at step level
        if self.lr_scheduler is not None:
          self.lr_scheduler.step()

      # do logging
      with torch.no_grad():
        # construct logging from metrics
        _to_log = {}
        _log_term = {}
        for metric in self.metrics:
          logs = metric.log_train(
            trainer=self, model_bs=self.model_bs, params=params, X=X, Y=Y, track=track
          )
          
          # add tqdm/terminal specific logs
          if 'term' in logs:
            _log_term.update(logs.pop('term'))
          
          # other logs/for wandb and such
          _to_log.update(
            logs
          )
        
        # append train tag
        to_log = {}
        if _to_log is not None:
          for k, v in _to_log.items():
            to_log[f'{k}/train'] = v
        
        # add loss if applicable
        if loss is not None:
          to_log['loss/train'] = float(loss.item())
  
        # update tqdm desc and wandb update on rank 0 only
        if self.rank == 0:
          self.log(to_log)
          
          if len(_log_term) > 0:
            others = ' | ' + ', '.join([('%s: %.3f' % (k, float(v))) for k, v in _log_term.items()])
          else:
            others = ''
          tq_train.set_description(f'L {loss.item():.4f}{others}')
        # break
        
    train_time_end = time.time() - train_time_start
    self.total_train_time += train_time_end
    
    # otherwise apply scheduler at epoch level
    if self.schedule_epoch:
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    
    if self.rank == 0:
      print(f'Finished training in {train_time_end}s. Total train time {self.total_train_time}s')
  
  def val_single_epoch(self, epoch: int, epochs: int):
    """ Test the model for a single epoch """
    if self.rank == 0:
      print(f'Running val on epoch {epoch}/{epochs}')
    
    # gets the shared target parameters
    module = self.hyper.module if self.ddp else self.hyper
    shared_parameters = list(module.target_parameters())
    self.hyper.eval()
    self.particle_method.eval()
    
    if self.ddp:
      self.test_loader.sampler.set_epoch(epoch - 1)
    
    t_loss = 0.0
    agg_test = []
    if self.rank == 0:
      tq_test = tqdm(self.test_loader, desc='Test', total=self.num_test, colour='blue')
    else:
      tq_test = self.test_loader
    
    # reset matric trackers
    with torch.no_grad():
      for metric in self.metrics:
        metric.reset()
    
    test_time_start = time.time()
    for b, (X, Y) in enumerate(tq_test):
      X = X.to(self.device_id)
      Y = Y.to(self.device_id)
      X = self.append_ood(X)
      
      # sample model_bs parameters from hypernetwork and feed through target network
      track = self.hyper(
        params=self.model_bs,
        x=X,
        sample_params=True,  # when True then params argument is now the number of models we want to sample not sample codes
        ret_params=True,  # include the generated hypernetwork parameters into the tracking dictionary
        feature_split=True,   # split inlier and outlier samples and create tracking dictionary
        
        # all other arguments are fed through hyper.forward_split(...)
        ood_N=self.ood_N,
        skip_empty=True,
        split_pred_only=not self.configs.get('separate_ood_features', True)
      )
      params = track['params']
      
      # push through method and get loss/forward data
      track = self.particle_method.forward(
        trainer=self,
        params=params,
        shared_params=shared_parameters,
        track=track,
        x=X, gt=Y, step=None  # ignore
      )
      
      # calculate loss and test stats
      loss, logs = self.particle_method.test(
        trainer=self,
        params=params,
        shared_params=shared_parameters,
        track=track,
        x=X, gt=Y,
      )
      
      t_loss += loss.item() / self.num_test
      
      # do logging
      with torch.no_grad():
        # construct logging from metrics
        _to_log = {}
        _log_term = {}
        for metric in self.metrics:
          logs = metric.log_val(
            trainer=self, model_bs=self.model_bs, params=params, X=X, Y=Y, track=track
          )
          
          # add tqdm/terminal specific logs
          if 'term' in logs:
            _log_term.update(logs.pop('term'))
          
          # other logs/for wandb and such
          _to_log.update(
            logs
          )
        
        # append train tag
        to_log = {}
        if _to_log is not None:
          for k, v in _to_log.items():
            to_log[f'{k}/val'] = v
        
        # add loss if applicable
        if loss is not None:
          to_log['loss/val'] = float(loss.item())
        agg_test.append(to_log)
        
        if self.rank == 0:
          if len(_log_term) > 0:
            others = ' | ' + ', '.join([('%s: %.3f' % (k, float(v))) for k, v in _log_term.items()])
          else:
            others = ''
          tq_test.set_description(f'L {loss.item():.4f}{others}')
        # break
    test_time_end = time.time() - test_time_start
    self.total_test_time += test_time_end

    # update tracking to test with the average value
    new_log = {}
    for k, v in to_log.items():
      if isinstance(v, (torch.Tensor, float, int)):
        new_log[k] = torch.mean(torch.tensor([h[k] for h in agg_test])).cpu()
      new_log[k] = v
    
    # only rank 0 reports logs
    if self.rank == 0:
      self.log(new_log)
      # tracking_test.append(to_log)
      print(f'Finished testing in {test_time_end}s. Total test time {self.total_test_time}s')
    
    # handle callbacks
    for callback in self.callbacks:
      trainer = self
      callback.epoch_end(epoch, trainer, self.model_bs, X, Y, track)
    return t_loss

  def prepare_optim(self, ddp_rank=None):
    """ Prepare the optimizers """
    params = []
    self.params_method = list(filter(lambda x: x.requires_grad, self.particle_method.parameters()))  # filter by learnable
    
    if ddp_rank is not None and len(self.params_method) > 0:
      raise RuntimeError('Currently learnable method parameters are not supported with distributed training! Currently %d parameters have been specified' % len(self.params_method))
      
    # define arguments
    if self.optim_args is None:
      self.optim_args = {}
    optim_args = copy.deepcopy(self.optim_args)
    all_weight_decay = optim_args.pop('weight_decay', 0.0)

    # add default values if not defined
    if self.optim_name in ['adam', 'nadam', 'adamw']:
      if 'decoupled_weight_decay' not in optim_args:
        optim_args['decoupled_weight_decay'] = True
    
    # selective weight decay
    for name, param in self.hyper.named_parameters():
      weight_decay = all_weight_decay

      # remove weight decay on affine parameters unless otherwise specified
      if not self.wd_affine:
        # reduce learning rate for affine weights/remove weight decay
        if name.endswith('affine_weight') or name.endswith('affine_bias') or 'affine.self' in name:  # see norm.py about affine
          weight_decay = 0.0
        
        # reduce learning rate/remove weight decay for skip gain residual connections
        if 'skip_gain' in name:
          weight_decay = 0.0

        # no weight decay for bias
        if name.endswith('bias'):
          weight_decay = 0.0

      # add group of parameters to optimizer
      params.append({
        'params': param,
        'lr': optim_args['lr'],
        'initial_lr': optim_args['lr'],
        'weight_decay': float(weight_decay),
        **optim_args
      })
    
    try:
      optim = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'nadam': torch.optim.NAdam,
        'sgd': torch.optim.SGD,
      }[self.optim_name]
    except KeyError:
      raise ValueError(f'Optimizer {self.optim_name} not found')

    # create optimizers
    self.optim_hyper = optim(params)
    
    # dummy var given sometimes this is empty
    if len(self.params_method) == 0:
      self.params_method = [torch.tensor(0.0, requires_grad=True)]
    self.optim_method = optim(self.params_method)

  def prepare_schedulers(self, epochs: int):
    """ Prepare the schedulers """
    # cosine annealing warmup
    if self.scheduler_func is not None:
      if self.schedule_epoch:
        T_max = epochs if self.max_iter is None else self.max_iter  # max epoch iterations
      else:
        T_max = int((epochs*self.num_train) if self.max_iter is None else (self.max_iter*self.num_train))
      self.lr_scheduler = self.scheduler_func(self.optim_hyper, T_max=T_max)
    else:
      self.lr_scheduler = None
    
    # lr patience/reduce lr on plateau
    if self.lr_patience is not None and self.lr_patience > 0:
      self.rlop = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optim_hyper,
        mode='min',
        factor=0.1,
        patience=self.lr_patience,
        verbose=True,
        cooldown=5,
        min_lr=1e-8,
      )
      self.rlop_method = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optim_hyper,
        mode='min',
        factor=0.1,
        patience=self.lr_patience,
        verbose=True,
        cooldown=5,
        min_lr=1e-8,
      )
    else:
      self.rlop = None
      self.rlop_method = None

  def state_dict(self, **extra):
    """ Get the state dict """
    return {
      'hyper': self.hyper.module.state_dict() if self.ddp else self.hyper.state_dict(),
      'method': self.particle_method.state_dict(),
      'optim': self.optim_hyper.state_dict(),
      'optim_method': self.optim_method.state_dict(),
      'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
      'rlop': self.rlop.state_dict() if self.rlop is not None else None,
      'rlop_method': self.rlop_method.state_dict() if self.rlop_method is not None else None,
      **extra
    }

  def load_state_dict(self, state: dict):
    """ Load the state dict """
    self.hyper.load_state_dict(state['hyper'])
    
    if self.particle_method is not None:
      self.particle_method.load_state_dict(state['method'])
    
    if self.optim_hyper is not None:
      self.optim_hyper.load_state_dict(state['optim'])
    
    if self.optim_method is not None:
      self.optim_method.load_state_dict(state['optim_method'])
    
    # handle loading old files with different name. @TODO remove later on
    if state.get('warmup') is not None and self.lr_scheduler is not None:
      self.lr_scheduler.load_state_dict(state['warmup'])
    
    if state.get('lr_scheduler') is not None and self.lr_scheduler is not None:
      self.lr_scheduler.load_state_dict(state['lr_scheduler'])
    
    if state.get('rlop') is not None and self.rlop is not None:
      self.rlop.load_state_dict(state['rlop'])
    
    if state.get('rlop_method') is not None and self.rlop_method is not None:
      self.rlop_method.load_state_dict(state['rlop_method'])

  def load_checkpoint(self, weight_file: str):
    """ Load a checkpoint """
    if os.path.isfile(weight_file):
      print('Attempting to load previous weights from ', weight_file)
      try:
        if self.ddp:
          self.save_data = torch.load(
            weight_file,
            map_location={  # copy rank0/cuda tensors to current device
              'cuda:0': f'cuda:{self.rank}',
              'cuda': f'cuda:{self.rank}'
            }
          )
        else:
          self.save_data = torch.load(weight_file)
        self.load_state_dict(self.save_data)
      except RuntimeError as err:
        print(f'Failed to load weights {str(err)}')
    else:
      print('Failed to find weight file', weight_file, 'ignoring...')

  def train(self, epochs: int=None, start_epoch: int=None, load_from_checkpoint: str=None, save_checkpoint: str=None, save_every: int=None, seed: int=None, ddp_rank: int=None, start_weights: str=None):
    """ Train the model for the specified number of epochs
    
    Args:
      epochs (int, optional): Number of epochs to train for. Defaults to None.
      start_epoch (int, optional): The starting epoch. Defaults to None.
      load_from_checkpoint (str, optional): The file to load from. Defaults to None.
      save_checkpoint (bool, optional): The file to save to (relative to output folder). Do not include .pt in the name. Defaults to None.
      save_every (int, optional): Save every n epochs. Defaults to None (ie disable).
      seed (int, optional): The seed for reproducibility. Defaults to None. If None then it is not set
      ddp_rank: sets mode as distributed data parallel and specifies the rank
      start_weights (str, optional): if specified load the weight file as the starting weights of the hypernetwork/ensemble. This assumes load from checkpoint is None.
    """
    
    # init seeds
    if seed is not None:
      random.seed(seed)
      torch.manual_seed(seed)
      np.random.seed(seed)
    
    # prepare DDP if applicable
    if ddp_rank is not None:
      if torch.cuda.device_count() == 0:
        raise RuntimeError('No cuda devices found!')
      self.ddp = True
      self.rank = ddp_rank
      self.hyper = self.hyper.to(self.device_id)
      self.hyper = DDP(self.hyper, device_ids=[self.device_id])
    else:
      self.ddp = False
    
    # setup optimizers
    self.prepare_optim(ddp_rank=ddp_rank)

    if epochs is None:
      epochs = self.configs.get('epochs', 100)

    # setup schedulers
    self.prepare_schedulers(epochs)
    
    # override default settings
    if start_epoch is None:
      start_epoch = self.configs.get('start_epoch', 0)
    
    if load_from_checkpoint is not None and start_weights is not None:
      raise ValueError('Can only specify loading from a checkpoint (ie resume training) or start weights (ie initialize hypernetwork/ens with specified weights) not both at the same time')
    
    # if we are loading from a checkpoint
    # load the saved data
    if load_from_checkpoint is not None:
      l_file = self.output_file(load_from_checkpoint)  # check output folder first
      if not os.path.exists(l_file):
        l_file = load_from_checkpoint
        if not os.path.exists(load_from_checkpoint):
          raise FileNotFoundError(f'Failed to find checkpoint file {load_from_checkpoint}')
      self.load_checkpoint(l_file)

    if start_weights is not None:
      if not os.path.exists(start_weights):
        start_weights = os.path.join(os.path.dirname(__file__), '..', '..', start_weights)
        if not os.path.exists(start_weights):
          raise FileNotFoundError(f'Failed to find checkpoint (start weights) file {start_weights}')
      state = torch.load(start_weights)
      if 'hyper' in state:  # checkpoint file or standalone
        state = state['hyper']
      self.hyper.load_state_dict(state)
      print(f'Loaded initial weights from {start_weights}')

    # loop through epochs
    self.total_train_time = 0.0
    self.total_test_time = 0.0
    best_loss = float('inf')
    for epoch in range(start_epoch, epochs):
      epoch += 1
      self.train_single_epoch(epoch, epochs)
      t_loss = self.val_single_epoch(epoch, epochs)

      if t_loss < best_loss:
        best_loss = t_loss
        if save_checkpoint is not None:
          if self.rank == 0:
            print('Saving new best checkpoint...')
            torch.save(self.state_dict(epoch=epoch), self.output_file(save_checkpoint) + '-best.pt')
      
      # @TODO use nccl/other method of distributing tensors
      # right now just saving state and reloading works "fine"
      if self.ddp and self.ddp_reload_every is not None and epochs % self.ddp_reload_every == 0:
        print('Resyncing DDP...')
        dist.barrier()
        if self.rank == 0:
          torch.save(self.state_dict(epoch=epoch), self.output_file(f'ddp-temp.pt'))  # have rank 0 save its state
        dist.barrier()
        
        if self.rank > 0:
          self.load_checkpoint(self.output_file(f'ddp-temp.pt'))  # reload state on all other ranks
        print('Resynced!')
      
      # save every n epochs
      if save_every is not None and epoch % save_every == 0:
        if self.rank == 0:
          print(f'Saving checkpoint...')
          torch.save(self.state_dict(epoch=epoch), self.output_file(f'{save_checkpoint}-{epoch}.pt'))

      # reduce lr on plateau
      if self.rlop is not None:
        self.rlop.step(t_loss)
        self.rlop_method.step(t_loss)
      
      # otherwise hanging on train step?
      if self.ddp:
        dist.barrier()
      
    # save final model
    if save_every is not None:
      if self.rank == 0:
        print('Saving final checkpoint...')
        torch.save(self.state_dict(epoch=epoch), self.output_file(f'{save_checkpoint}-{epoch}.pt'))

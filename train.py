import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import os
import random
import multiprocessing
import configargparse
import copy

import torch.distributed as dist
import torch.multiprocessing as mp

from hyper.util.collections import unflatten_keys
from hyper.experiments.exp import build_from_file

# specify wandb project
WANDB_PROJECT = 'he-cka-ensembles'

# specify number of workers for data loaders
os.environ['MAX_WORKERS'] = '0'  # str(min(int(multiprocessing.cpu_count() / 2), 10))


# arg parse
p = configargparse.ArgParser()
p.add('config', type=str, help='The training configuration file to use. Note that this is relative to the configs folder')
p.add('-o', '--out', type=str, default=None, help='Output folder to save to. Default is in output + relative to the config file')
p.add('-wb', '--wandb', action='store_true', default=False, help='Use wandb for logging')
p.add('-sw', '--sweep', type=str, default=None, help='Start as an agent and accept config overrides from wandb')  # to use a wandb agent
p.add('-r', '--runs', type=int, default=1, help='Number of runs to average over')
p.add('-l', '--load', type=str, default=None, help='Load a model from a checkpoint. Relative to output or absolute path')
p.add('-s', '--save', type=str, default='model', help='Save a model to a checkpoint. Relative to output or absolute path')
p.add('-se', '--save_every', type=int, default=None, help='Save a model every n epochs. Default load from config')
p.add('-g', '--gpus', type=int, default=None, help='Use DDP training with N gpus. Default is None or one GPU and no DDP')
p.add('--seed', type=int, default=5, help='Seed for reproducibility')
args = p.parse_args()

# output in the experiments folder
if args.out is None:
  config_relative = os.path.splitext(os.path.relpath(args.config, os.path.dirname(__file__)))[0]
  
  # common case is configs is the start of the path
  if config_relative.startswith('configs/'):
    config_relative = config_relative.split('configs/')[-1]
  
  # default out is relative config path
  OUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs', config_relative)
else:
    OUT_FOLDER = args.out


# specify distributed training
if args.gpus is not None:
  ddp = True
  ddp_size = args.gpus
  
  if args.runs > 1:
    # please see NOTE stated here https://pytorch.org/docs/stable/distributed.html
    # Reinitialization: destroy_process_group can also be used to destroy individual process groups. 
    # issue potentially with reinitialization of runs. Currently not well supported
    raise RuntimeError('Multiple runs with DDP not supported due to reinitialization see notes in https://pytorch.org/docs/stable/distributed.html')  
else:
  ddp = False
  ddp_size = 1

  # if using wandb start run and logs
  # login here and do not worry about DDP
  if args.wandb:
    import wandb
    wandb.login()


# for sweeps we only allow single runs
if args.wandb and args.sweep is not None and args.runs > 1:
  raise ValueError('Cannot specify multiple runs and a sweep at the same time')


# specify main training function
def train_main(run=0, rank=None, ddp=False):
  """ Handles training using HyperTraininer with local training or DDP support """
  is_rank0 = rank is None or rank == 0
  
  if ddp:  # single run ddp do wandb login on rank 0
    # if using wandb start run and logs
    if args.wandb and is_rank0:
      import wandb
      wandb.login()
    device_id = (0 if rank is None else rank) % torch.cuda.device_count()
  else:
    device_id = 'cuda'
    if args.wandb:
      import wandb
  
  def log(*pargs, **kwargs):
    if args.wandb and is_rank0:
      wandb.log(*pargs, **kwargs)  # log only on main rank
  
  out_folder = OUT_FOLDER + f'-run{run}'
  configs, trainer = build_from_file(args.config, log=log, out_path=out_folder, ddp=ddp, trainer_kwargs={'device_id': device_id})
  
  # if in normal local cuda (not multi device) move all params over
  if not ddp:
    trainer = trainer.cuda()

  # start run on wandb if args specified and rank 0
  if args.wandb and args.sweep is None and is_rank0:
    print('Initializing new wandb run...')
    wandb.init(
      project=WANDB_PROJECT,
      config=configs,
      name=args.config.replace('.yml', '').replace('/', '_').replace('\\', '_').replace('configs_', ''),
    )
  
  # load save every from config by default
  if args.save_every is None:
    if 'save_every' in configs['trainer']:
      args.save_every = configs['trainer']['save_every']
  
  # start the training on this device
  trainer.train(
    load_from_checkpoint=args.load,
    save_checkpoint=args.save,
    save_every=args.save_every,
    seed=None,
    ddp_rank=rank
  )


def run_ddp_training(rank, world_size):
  """ Main function to handle training via DDP """
  if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'
  if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '14356'
  
  # start process group
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  
  # run main training
  try:
    train_main(run=0, rank=rank, ddp=True)
  finally:
    # destroy process group
    dist.destroy_process_group()
    print('Cleaned up process group...')

# if sweep let's collect the override config
# we do not save/have an output in that case
if __name__ == '__main__':
  if args.wandb and args.sweep is not None:
    if ddp:
      raise RuntimeError('Sweep currently not supported with DDP')

    def log(*pargs, **kwargs):
      if args.wandb:
        wandb.log(*pargs, **kwargs)

    def agent_wrap():
      with wandb.init() as run:
        print('CONFIG', wandb.config)
        
        # unflatten into expected structure
        extra = copy.deepcopy(unflatten_keys(wandb.config))

        for num in range(args.runs):
          print('Starting run', num)
          
          # rebuild the trainer each time with new config
          _, trainer = build_from_file(
            file=args.config,  # base config
            override_config=extra,
            log=log,
            out_path=None
          )
          trainer = trainer.cuda()
          trainer.train(
            load_from_checkpoint=None, save_checkpoint=None,
            save_every=None, seed=None
          )
          print('Run', num, 'done')
        print('All runs done')
        run.finish()
    
    # prepare sweep agent
    wandb.agent(
      sweep_id=args.sweep,
      function=agent_wrap,
      project=WANDB_PROJECT
    )
  else: # run normally
    # run for n runs
    for run in range(args.runs):
      print('Starting run', run)
      
      # set seed for reproducibility
      random.seed(args.seed + run)
      torch.manual_seed(args.seed + run)
      np.random.seed(args.seed + run)
      
      # rebuild the trainer each time/restart ddp if applicable
      if ddp:
        mp.spawn(
          run_ddp_training,
          args=(ddp_size,),
          nprocs=ddp_size,
          join=True
        )
      else:
        train_main(run=run, rank=None, ddp=False)
      
      print('Run', run, 'done')
    print('All runs done')



  # EXP_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
  # CONFIG = 'configs/toy/fourclass/fsvgd/rbf.yml'

  # # default out is relative config path
  # OUT_FOLDER = os.path.join(EXP_FOLDER, CONFIG.split('configs/')[-1].split('.yml')[0])

  # trainer = build_from_file(CONFIG, out_path=OUT_FOLDER).cuda()
  # trainer.train(seed=None)
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import os
import random
import json
import math
import configargparse
import copy

from hyper.util.collections import unflatten_keys
from hyper.experiments.exp import build_from_file, null_fn
from hyper.experiments.testing.mnist import run_mnist_tests


# specify wandb project
WANDB_PROJECT = 'hyper-models'

# specify number of workers for data loaders
os.environ['MAX_WORKERS'] = '0'


EVAL_SCRIPTS = {
  'mnist': run_mnist_tests
}


# arg parse
p = configargparse.ArgParser()
p.add('config', type=str, help='The training configuration file to use. Note that this is relative to the configs folder')
p.add('experiment', type=str, choices=['mnist', 'cifar10', 'cifar100'], help='The set of evaluations to run')
p.add('file', type=str, default='model-50.pt', help='Name of the weight file located in the output folder')
p.add('-f', '--folder', type=str, default=None, help='Base to output folder to load project from. Must contain be same relative pos to configs')
p.add('-r', '--runs', type=int, help='The number of runs to process')
p.add('-mbs', '--model_bs', type=int, default=None, help='Specify the model batch size. Must be specified for hypernetwork based models, optional for ensemble ones, with default being ensemble size.')
p.add('--cka', action='store_true', default=False, help='Plot CKA values as well')
p.add('--seed', type=int, default=5, help='Seed for reproducibility')
args = p.parse_args()


if args.experiment not in EVAL_SCRIPTS:
  raise ValueError('Invalid experiment option. Check eval.py code for more details')

# output in the experiments folder
if args.folder is None:
  config_relative = os.path.splitext(os.path.relpath(args.config, os.path.dirname(__file__)))[0]
  
  # common case is configs is the start of the path
  if config_relative.startswith('configs/'):
    config_relative = config_relative.split('configs/')[-1]
  
  # default out is relative config path
  OUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs', config_relative)
else:
  OUT_FOLDER = args.folder


# load the trainer
configs, trainer = build_from_file(args.config, log=null_fn, out_path=None)
trainer = trainer.cuda()

# get the default batch size from config or override using arguments
if 'hyper' in configs and 'ensemble_size' in configs['hyper']:
  DEFAULT_MODEL_BS = configs['hyper']['ensemble_size']
else:
  DEFAULT_MODEL_BS = None
if args.model_bs is None:
  args.model_bs = DEFAULT_MODEL_BS  # default use the configuration ensemble size
else:
  args.model_bs = int(args.model_bs)

if args.model_bs is None:
  raise ValueError('Must specify a model batch size. Ensemble size could not be determined from configuration')


def eval_with_bs(model_bs: int):
  results = []
  
  # run for n runs
  for run in range(args.runs):
    print('Starting run', run+1)
    
    # set seed for reproducibility
    random.seed(args.seed + run)
    torch.manual_seed(args.seed + run)
    np.random.seed(args.seed + run)
    
    # load the checkpoint
    out_folder = OUT_FOLDER + f'-run{run}'
    trainer.load_checkpoint(os.path.join(out_folder, args.file))
    
    # run the eval function
    output = EVAL_SCRIPTS[args.experiment](
      hyper=trainer.hyper,
      model_bs=model_bs,
      device='cuda',
      out_path=out_folder,
      plot_cka=args.cka
    )
    results.append(output)
    print('Run', run, 'done')
  print('All runs done')
  
  print('Saving results')
  # process all runs
  # unzip dicts
  unzipped = {}
  for k in results[0].keys():
    unzipped[k] = [r[k] for r in results]
  
  # mean, std, and standard error values
  mean_res = {}
  std_res = {}
  std_err_res = {}
  for k, v in unzipped.items():
    mean_res[k] = np.mean(v).item()
    std_res[k] = np.std(v).item()
    std_err_res[k] = std_res[k] / math.sqrt(len(v))  # standard error
  
  # save results
  with open(os.path.join(os.path.dirname(OUT_FOLDER), f'{os.path.basename(OUT_FOLDER)}-{os.path.splitext(args.file)[0]}-{args.runs}runs-mbs{args.model_bs}-results.json'), 'w') as f:
    json.dump({
      'means': mean_res,
      'stds': std_res,
      'std_errs': std_err_res
      }, f, indent=4
    )
  
eval_with_bs(args.model_bs)
print('Evaluation finished')

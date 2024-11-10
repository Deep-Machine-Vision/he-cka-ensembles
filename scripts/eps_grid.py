# @TODO david update this to use same as paper values (on work machine)

from hyper.experiments.metrics import Callback
from hyper.data import generate_classification_data

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']='\\usepackage{amsmath}'

import torch
import numpy as np
import random
import cv2

WANDB = False
ENTROPY = False
PLOT = True


# output in the experiments folder
EXP_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
CONFIG = 'configs/toy/fourclass/svgd/he-cossim.yml'
OUT_FOLDER = os.path.join(EXP_FOLDER, CONFIG.split('configs/')[-1].split('.yml')[0], 'eps_grid')


arc_eps = [0.01, 0.015, 0.02, 0.025, 0.03]
eps = [0.00025]

PLOT_SIZE = 10.0
PLOT_POINTS = generate_classification_data(n_samples=320)[0]


class EPSGridCallback(Callback):
  def __init__(self, name):
    super(EPSGridCallback, self).__init__()
    self.name = name
  
  def epoch_end(self, epoch: int, trainer, model_bs, X, Y, forward_data=None, *args, **kwargs):
    """ Logs the metrics """
    if epoch % 50 == 0:
      # plot_entropy_grid(trainer.hyper, PLOT_POINTS, PLOT_SIZE, trainer.output_file(f'toy-grid-{epoch}.png'))
      points = next(iter(trainer.train_loader))[0]
      print('SAVING TO', trainer.output_file(f'grid-{self.name}.png'))
      plot_entropy_grid(trainer.hyper, points, PLOT_SIZE, trainer.output_file(f'grid-{self.name}.png'))
    

if PLOT:
  print('Plotting training results')
  
  def load_img(arc, eps):
    path = f'outputs/toy/fourclass/svgd/he-cossim/eps_grid/grid-arc{a}-eps{e}.png'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[110:422, 100:414]  # crop
  
  fig, axs = plt.subplots(len(eps), len(arc_eps), figsize=(12, 3), dpi=200)
  # for i, e in enumerate(eps):
  e = eps[0]
  for j, a in enumerate(arc_eps):
    img = load_img(a, e)
    axs[j].imshow(img)
    axs[j].axis('off')
    axs[j].set_title('$\\epsilon_{\\text{arc}}=%.3f$' % (a), fontsize=26)
  plt.tight_layout()
  fig.savefig('arc-eps-grid.png')
  
else:
  print('Training')
  for a in arc_eps:
    for e in eps:
      print('ARC_EPS', a, 'EPS', e)
      
      from hyper.experiments.metrics import Callback
      from hyper.util.vis import plot_entropy_grid
      from hyper.data import generate_classification_data
      from hyper.experiments.exp import build_from_file
      random.seed(5)
      torch.manual_seed(5)
      np.random.seed(5)

      # specify override config
      override = {
        'method': {
          'model_kernel': {
            'layer_kernel': {
              'arc_eps': a,
              'eps': e
            }
          }
        },
        'trainer': {
          'callbacks': [EPSGridCallback(f'arc{a}-eps{e}')]
        }
      }

      # train
      trainer = build_from_file(CONFIG, override_config=override, out_path=OUT_FOLDER)[1].cuda()
      trainer.train(seed=None)
      
      # restart trainer
      del trainer
      import gc
      gc.collect()
      torch.cuda.empty_cache()
""" Other useful functions for the fourclass toy dataset.

This is loaded as an extras file in the fourclass toy dataset.
"""
import os

import torch

from hyper.data import generate_classification_data
from hyper.experiments.training import HyperTrainer
from hyper.experiments.metrics import Callback
from hyper.util.vis import plot_entropy_grid
import sklearn.datasets as skdatasets


PLOT_SIZE = 10.0
PLOT_POINTS = generate_classification_data(n_samples=320)[0]

class FourClassCallback(Callback):
  def __init__(self):
    super(FourClassCallback, self).__init__()
  
  def epoch_end(self, epoch: int, trainer: HyperTrainer, model_bs, X, Y, forward_data=None, *args, **kwargs):
    """ Logs the metrics """
    if epoch % 50 == 0:
      # plot_entropy_grid(trainer.hyper, PLOT_POINTS, PLOT_SIZE, trainer.output_file(f'toy-grid-{epoch}.png'))
      points = next(iter(trainer.train_loader))[0]
      plot_entropy_grid(trainer.hyper, points, PLOT_SIZE, trainer.output_file(f'toy-grid-{epoch}.png'))
    
def test():
  print('OKAY')
  

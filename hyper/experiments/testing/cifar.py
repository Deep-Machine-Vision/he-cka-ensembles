""" Handles plotting results for mnist """
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import OrderedDict
from typing import Dict

from hyper.data.fast_cifar import get_test_loader, get_test100_loader
from hyper.data import load_svnh_test
from hyper.diversity.uncertainty import get_eval_stats_ensemble
from hyper.diversity import uncertainty
from hyper.experiments.testing.mnist import entropy, Evaluation, null_log


import traceback
from dataclasses import dataclass
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


BATCH_SIZE = 128


def run_cifar10_tests(hyper, model_bs: int, device: str='cuda', out_path: str='.', plot_cka: bool=False):
  """ Runs all tests for MNSIT
  
  Args:
    hyper: The hypernetwork
    model_bs: The number of models in the ensemble
    device: The device to run on
    out_path: The output path to save to
    plot_cka: Whether to plot the CKA values
    
  Returns:
    The results of the tests as a dictionary
  """

  if isinstance(model_bs, list):
      incr_samp = model_bs
      model_bs = model_bs[-1]
  else:
      incr_samp = [model_bs]
  
  # preload the parameters to not regenerate them every batch
  sparam = hyper.sample_params(model_bs, device=device)  # sample init for params
  params = hyper.forward_params(sparam)  # feed through parameter generator

  results = {}
  for incr_s in incr_samp:
    print('Sampling incremental', incr_s)

    params_s = OrderedDict()
    def sub_grab(p, p_s):
      for k in p.keys():
        if isinstance(p[k], OrderedDict):
          p_s[k] = OrderedDict()
          sub_grab(p[k], p_s[k])
        else:
          p_s[k] = p[k][:incr_s]

    # sample first n params
    sub_grab(params, params_s)

    # evaluate the model
    ood_test_loader = load_svnh_test(
      bs=BATCH_SIZE, drop_last=False
    )
    id_test_loader = get_test_loader(
      batch_size=BATCH_SIZE
    )
    
    if plot_cka:
      print('Plotting CKA values')
      hyper.unbiased_cka(id_test_loader, device=device, save_path=os.path.join(out_path, 'cka_mnist.png'))

    try:
      nlls, nlls_dang, accuracy, ece, ece_30, auroc_mi, auprc_mi, auroc_pe, auprc_pe, auroc_conf, auprc_conf = get_eval_stats_ensemble(
        hyper=hyper,
        params=params_s,
        test_loader=id_test_loader,
        ood_test_loader=ood_test_loader,
        device='cuda'
      )
      print(f'nll: {nlls}, nll_dang: {nlls_dang}, acc: {accuracy}, ece: {ece}, ece_30: {ece_30}, auroc_mi: {auroc_mi}, auprc_mi: {auprc_mi}, auroc_pe: {auroc_pe}, auprc_pe: {auprc_pe}')
      results.update({
        f'{incr_s}_nll': nlls.cpu().item(),
        f'{incr_s}_nll_dang': nlls_dang.cpu().item(),
        f'{incr_s}_acc': 100.0 * accuracy,
        f'{incr_s}_ece': 100.0 * ece,
        f'{incr_s}_auroc_mi': 100.0 * auroc_mi,
        f'{incr_s}_auprc_mi': 100.0 * auprc_mi,
        f'{incr_s}_auroc_pe': 100.0 * auroc_pe,
        f'{incr_s}_auprc_pe': 100.0 * auprc_pe
      })
    except ValueError as e:
      traceback.print_exc()
      print('Failed to evaluate model', str(e))
  
  return results

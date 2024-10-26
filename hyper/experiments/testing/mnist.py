""" Handles plotting results for mnist """
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from collections import OrderedDict
from typing import Dict

from hyper.data.ambiguous_mnist import get_loaders as amnist_loader
from hyper.data.dirty_mnist import get_test_loader as dmnist_loader, get_train_valid_loader as dmnist_train_loader
from hyper.data.fashion_mnist import get_loaders as fmnist_loader
from hyper.data import load_mnist
from hyper.data.fast_mnist import get_test_loader as mnist_test_loader
from hyper.diversity.uncertainty import get_eval_stats_ensemble
from hyper.diversity import uncertainty


import traceback
from dataclasses import dataclass
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


BATCH_SIZE = 128


def null_log(*args, **kwargs):
  pass


def entropy(p):
  nats = -p * torch.log(p)
  nats[torch.isnan(nats)] = 0.0
  entropy = torch.sum(nats, dim=-1)
  return entropy


@dataclass
class Evaluation:
  predictions: torch.Tensor
  batch_predictions: torch.Tensor
  features: torch.Tensor
  entropies: torch.Tensor
  labels: torch.Tensor
  confidence: torch.Tensor
  accuracy: float


@torch.no_grad()
def evaluate(hyper, params, test_loader, mbs, mean_after=True, save_to=None, threshold=1.4, lt=True):
  """ As described in https://blackhc.github.io/ddu_dirty_mnist/dirtymnist.html#Evaluating-DirtyMNIST-and-AMNIST """
  hyper.eval()

  labels = []
  predictions = []
  batch_predictions = []
  confidence = []
  features = []
  correct = 0.0
  total = 0.0
  hyper.eval()
  im_index = 0
  for X, Y in tqdm(test_loader, total=len(test_loader)):
    X = X.cuda()
    Y = Y.cuda()
    
    # we don't care about features here
    # feed through ensemble/hypernetwork
    feat, pred = hyper.forward(params, X)
    # print(flatten_keys(feat)['4'].keys())
    # batch_features = feat['4']['self']  # get the features before shared

    # get average prediction
    if mean_after:
      avg_pred = F.softmax(pred, dim=-1)
      batch_pred = avg_pred.mean(dim=0)  # average across batched models
    else:
      avg_pred = pred.mean(dim=0)
      batch_pred = F.softmax(avg_pred, dim=-1)
    
    # save the images that go above/below entropy threshold
    if save_to is not None:
      entrop_inds = np.nonzero((entropy(pred) < threshold) if lt else (entropy(pred) > threshold)).flatten()
      if len(entrop_inds) > 0:
        for i in entrop_inds:
          im = torch.clamp(255.0 * ((X[i] * 0.3081) + 0.1307), 0, 255).cpu().view(28, 28).to(torch.uint8).numpy()
          cv2.imwrite(os.path.join(save_to, f'{im_index}.png'), im)
          print(f'Saved {im_index}')
          im_index += 1
    
    # get accuracy
    correct += (torch.argmax(batch_pred, dim=-1) == Y).sum().item()
    total += Y.shape[0]
    predictions.append(batch_pred)
    batch_predictions.append(pred)
    # features.append(batch_features)
    confidence.append(torch.max(batch_pred, dim=1)[0])
    labels.append(Y)
  
  labels = torch.concat(labels).cpu()
  predictions = torch.concat(predictions).cpu()
  batch_predictions = torch.concat(batch_predictions, dim=1).cpu()
  features = None  # torch.concat(features, dim=1).cpu()
  entropies = entropy(predictions).cpu()
  confidence = torch.concat(confidence).cpu()
  accuracy = 100.0 * (correct / total) if total > 0.0 else -1.0
  return Evaluation(predictions, batch_predictions, features, entropies, labels, confidence, accuracy)


@torch.no_grad()
def run_evals(hyper, params, model_bs, dsets=['dmnist', 'amnist', 'mnist', 'fmnist']) -> Dict[str, Evaluation]:
  if isinstance(hyper, (OrderedDict, dict)):
    print('Loading from experiment requirements')
    hyper = hyper['hyper']
  
  if isinstance(model_bs, (OrderedDict, dict)):
    model_bs = model_bs['model_bs']
  
  returns = {}
  if 'dmnist' in dsets:
    # dirty mnist
    print('Running dirty...')
    dmnist_test_loader = dmnist_loader(
      batch_size=BATCH_SIZE
    )
    dmnist_test_evaluation = evaluate(hyper, params, dmnist_test_loader, model_bs)
    print(f'Accuracy {dmnist_test_evaluation.accuracy}%')
    returns['dmnist_loader'] = dmnist_test_loader
    returns['dmnist'] = dmnist_test_evaluation
  
  if 'amnist' in dsets:
    # ambiguous mnist
    print('Running ambiguous')
    amnist_test_loader = amnist_loader(
      train=False,
      batch_size=BATCH_SIZE
    )
    amnist_test_evaluation = evaluate(hyper, params, amnist_test_loader, model_bs)# , save_to='debug/amnist', lt=False)
    print(f'Accuracy {amnist_test_evaluation.accuracy}%')
    returns['amnist_loader'] = amnist_test_loader
    returns['amnist'] = amnist_test_evaluation
  
  if 'mnist' in dsets:
    # regular mnist
    print('Running regular...')
    # _, mnist_test_loader = data  # load from build_exp_requirements
    _, mnist_test_loader = load_mnist(
      bs=BATCH_SIZE,
      dirty=False
    )
    mnist_test_evaluation = evaluate(hyper, params, mnist_test_loader, model_bs)
    print(f'Accuracy {mnist_test_evaluation.accuracy}%')
    returns['mnist_loader'] = mnist_test_loader
    returns['mnist'] = mnist_test_evaluation
  
  if 'fmnist' in dsets:
    # fashion mnist
    print('Running fashion')
    fmnist_test_loader = fmnist_loader(
      batch_size=BATCH_SIZE,
      train=False
    )
    fmnist_test_evaluation = evaluate(hyper, params, fmnist_test_loader, model_bs) # , save_to='debug/fmnist', lt=True)
    print(f'Accuracy {fmnist_test_evaluation.accuracy}%')
    returns['fmnist_loader'] = fmnist_test_loader
    returns['fmnist'] = fmnist_test_evaluation
  return returns



def plot_density(dirty, fashion, binrange, separate_ID=False, save_fig='fig.png'):
  clrs = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22','#17becf']
  sns.set_style('whitegrid')

  plt.figure(figsize=(2.5,2.5/1.6))
  plt.tight_layout()

  range = dict(bins=30, binrange=binrange, element="step", fill=True, alpha=0.7)
  kw_separate_id = dict(hue="category", multiple="stack",
              hue_order=[1, 0],
              palette=[sns.color_palette()[4], sns.color_palette()[0]]) if separate_ID else dict(color=sns.color_palette()[0])

  sns.histplot(data=dirty.numpy(), **kw_separate_id,
              stat='probability', kde=False, **range, label="dummy", legend=False) # 'Dirty-MNIST (In-distribution)')
  sns.histplot(fashion.numpy(), color=sns.color_palette()[1],
              stat='probability', kde=False, **range, label="dummy", legend=False) #, label='Fashion-MNIST (OoD)')

  plt.xlabel('Log Density', fontsize=12)
  plt.ylabel('Fraction', fontsize=12)
  plt.savefig(save_fig)
  plt.close()


def run_mnist_tests(hyper, model_bs: int, device: str='cuda', out_path: str='.', plot_cka: bool=False):
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

  incr = False
  if isinstance(model_bs, list):
      incr = True
      incr_samp = model_bs
      model_bs = model_bs[-1]
  else:
      incr_samp = [model_bs]
  
  # preload the parameters to not regenerate them every batch
  sparam = hyper.sample_params(model_bs, device=device)  # sample init for params
  
  # if isinstance(sparam, list):
  #   mbs = sparam[0].shape[0]
  # else:
  #   mbs = sparam.shape[0]
  params = hyper.forward_params(sparam)  # feed through parameter generator

  returns = run_evals(hyper, params, model_bs)  # run on all datasets
  dmnist_test_evaluation = returns['dmnist']
  amnist_test_evaluation = returns['amnist']
  mnist_test_evaluation = returns['mnist']
  fmnist_test_evaluation = returns['fmnist']

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

    # print('Running test on dirty mnist and fashion mnist')
    # # evaluate the model
    ood_test_loader = fmnist_loader(
      batch_size=BATCH_SIZE,
      train=False
    )
    id_test_loader = dmnist_loader(
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
  
  sns.set_style('white')
  sns.set_style('ticks')
  sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 2.5})
  
  # borrowed and modifed from DDU's implementation
  common_kwargs = dict(stat='probability', kde=False, bins=12, binrange=[0,2.4], legend=False, element="step", alpha=0.7)
  id_kwargs = dict(color=sns.color_palette()[0])

  fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5,3.5/1.6), gridspec_kw={'height_ratios': [1, 3]}, constrained_layout=True)
  fig.subplots_adjust(hspace=0.30)  # adjust space between axes

  first = False
  for ax in axes:
    sns.histplot(mnist_test_evaluation.entropies.numpy(), **id_kwargs,
                **common_kwargs, ax=ax, label="MNIST" if first else None)
    sns.histplot(amnist_test_evaluation.entropies.numpy() , color=sns.color_palette()[2],
                **common_kwargs, ax=ax, label="AMNIST" if first else None)
    sns.histplot(fmnist_test_evaluation.entropies.numpy(), color=sns.color_palette()[1],
                **common_kwargs, ax=ax, label="FMNIST" if first else None)
    first = True

  axes[0].set_ylim(0.7, 1.0)  # outliers only
  axes[0].set_xticks([])
  axes[0].set_xticks([], minor=True)
  axes[1].set_ylim(0, .35)  # most of the data

  axes[0].spines['bottom'].set_visible(False)
  axes[1].spines['top'].set_visible(False)
  axes[0].set_ylabel("")
  axes[1].set_ylabel("Fraction", fontsize=14)
  axes[1].set_xlabel("Entropy", fontsize=14)

  axes[1].yaxis.set_label_coords(0.075, 0.55, fig.transFigure)

  d = .5  # proportion of vertical to horizontal extent of the slanted line
  kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
  axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
  axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)

  fig.subplots_adjust(bottom=0.25, left=0.175, right=0.835)
  axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=False, ncol=5, fontsize=11)
  fig.set_size_inches(4.0,4.0/1.6)
  axes[1].yaxis.set_label_coords(-.125, .675)
  
  # fig.tight_layout()
  fig.savefig(os.path.join(out_path, 'mnist-entropies.png'))
  plt.close(fig)
  
  return results

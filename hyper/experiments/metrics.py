""" Class to handle logging specific metrics from the method """
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import copy

from hyper.diversity import pairwise_cossim


AVAILABLE_METRICS = {}

def register_metric(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_METRICS[name] = cls
    return cls
  return decorator


def build_metric(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Metric configurations not found in configs. Must define a metric and contain the name of the metric as a key in that dictionary')
  
  cls = AVAILABLE_METRICS[name]
  return cls(**config)


class Metric(object):
  def __init__(self) -> None:
    pass
  
  def log(self, trainer, model_bs, X, Y, track, *args, **kwargs):
    """ Logs the metrics """
    return {}
  
  def log_val(self, *args, **kwargs):
    """ Logs the metrics """
    return self.log(*args, **kwargs)
    
  def log_train(self, *args, **kwargs):
    """ Logs the metrics """
    return self.log(*args, **kwargs)


@register_metric("ckas")
class CKASMetric(Metric):
  def __init__(self, track_kernel: str='ind') -> None:
    """ Handle tracking intermediate CKAS values by either indistribution or out of distribution kernel.

    NOTE: @TODO David if you specify None for track_kernel it will recalculate from scratch using linear CKA, but why recompute if we are already tracking that tensor.

    Args:
        kernel (str, optional): 'ind' for tracking in distribution kernel, and 'ood' for tracking ood kernel. Defaults to 'ind'.
    """
    self.kernel = track_kernel
  
  @torch.no_grad()
  def log(self, train_test, trainer, model_bs, X, Y, track, *args, **kwargs):
    """ Logs the metrics """
    
    if self.kernel is None:
      # todo: add linear cka calc
      raise NotImplementedError('CKAS with None kernel is not implemented yet.')
    
    # resolve ind or ood kernel ckas
    try:
      track = track[f'{self.kernel}_kernel']
    except KeyError as err:
      print(f'Could not report CKAS! {self.kernel} kernel was not found in the layer track.')
      return {}
    
    # see if we have the ckas
    if 'layer_track' not in track:
      print('Could not report CKAS! Gram matrices were not found in the layer track.')
      return {}
  
    to_log = {}
    for ind, c in enumerate(track['layer_track']):
      if 'gram' not in c:
        continue
      
      to_log[f'cka_{self.kernel}_{ind + 1}/{train_test}'] = pairwise_cossim(c['gram']).mean().item()
    
    return to_log
  
  def log_val(self, *args, **kwargs):
    """ Logs the metrics """
    return self.log('val', *args, **kwargs)
    
  def log_train(self, *args, **kwargs):
    """ Logs the metrics """
    return self.log('train', *args, **kwargs)


@register_metric("accuracy")
class AccuracyMetric(Metric):
  def __init__(self, classes: int, term_track: bool=True) -> None:
    self.acc = Accuracy(task="multiclass", num_classes=classes)
    self.term_track = term_track
    
  @torch.no_grad()
  def log(self, trainer, model_bs, X, Y, track, *args, **kwargs):
    """ Logs the metrics """
    Y_p = track['pred_ind']
    
    # Y = Y_t.repeat(n)  # for all models
    preds = torch.argmax(Y_p.reshape(Y_p.shape[0]*Y_p.shape[1], -1), dim=1)
    
    # construct accuracy metric object on this device
    if self.acc.device != preds.device:
      self.acc = self.acc.to(preds.device)
    
    # get std of acc
    _, bs = Y.repeat(model_bs).reshape(model_bs, -1).shape
    accs = []
    for i in range(model_bs):
      accs.append(self.acc(preds[i*bs:(i+1)*bs], Y).cpu().item())
    
    res = {
      'accuracy': self.acc(torch.argmax(F.softmax(Y_p, dim=-1).mean(0), dim=-1), Y),
      'accuracy_std': float(np.std(accs)),
    }
    
    if self.term_track:  # also add to terminal/tqdm
      res['term'] = {
        'acc': res['accuracy']
      }
    
    return res


class Callback(object):
  def __init__(self) -> None:
    pass
  
  def epoch_end(self, epoch, trainer, model_bs, X, Y, track, *args, **kwargs):
    """ Logs the metrics """
    pass


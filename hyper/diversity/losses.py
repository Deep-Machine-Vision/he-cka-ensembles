""" Common loss functions used in batched particle models """
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from hyper.diversity.uncertainty import entropy, entropy_prob


AVAILABLE_LOSSES = {}

def register_loss(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_LOSSES[name] = cls
    return cls
  return decorator


def build_loss(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_LOSSES[name]
  config = cls.from_config(config)
  return cls(**config)


class LossFunction(nn.Module):
  """ Base class for loss functions """
  def __init__(self):
    super(LossFunction, self).__init__()
    
  def forward(self, model_bs, x, pred, gt):
    """ Forward pass of the loss function
    
    Args:
      model_bs (torch.nn.Module): The model batch size (useful for batched models)
      x (torch.Tensor): The input data
      pred (torch.Tensor): The predictions of the model
      gt (torch.Tensor): The ground truth labels
    """
    raise NotImplementedError("Must implement forward method")

  @staticmethod
  def from_config(config: dict):
    """ Create a loss function from a configuration dictionary """
    return config


@register_loss("cross_entropy")
class BatchedCrossEntropy(LossFunction):
  """ Cross entropy loss """
  def __init__(self, reduction: str='mean'):
    super(BatchedCrossEntropy, self).__init__()
    self.reduction = reduction
  
  def forward(self, model_bs, x, pred, gt):
    """ Independently calculate the cross entropy loss for each model in the batch

    Args:
      model_bs (torch.nn.Module): The model batch size (useful for batched models)
      x (torch.Tensor): The input data
      pred (torch.Tensor): The predictions of the model
      gt (torch.Tensor): The ground truth labels
    """
    Y = gt.repeat(model_bs)
    return F.cross_entropy(pred.reshape(Y.shape[0], -1), Y, reduction=self.reduction)


@register_loss("mean_softmax_entropy")
class MeanSoftmaxEntropy(LossFunction):
  """ OOD entropy loss calculated as the entropy of the mean softmax prediction across the particles"""
  def __init__(self):
    super(MeanSoftmaxEntropy, self).__init__()
  
  def forward(self, model_bs, x, pred, gt):
    """ Independently calculate the cross entropy loss for each model in the batch

    Args:
      model_bs (torch.nn.Module): The model batch size (useful for batched models)
      x (torch.Tensor): The input data
      pred (torch.Tensor): The predictions of the model
      gt (torch.Tensor): The ground truth labels
    """
    return -entropy_prob(F.softmax(pred, dim=-1).mean(0)).mean()


@register_loss("inlier_outlier_entropy")
class InlierOutlierEntropy(LossFunction):
  """ A balancing loss function for learning hyperparameters to balance inlier and outlier entropies with a weight"""
  def __init__(self, weight=1.0):
    super(InlierOutlierEntropy, self).__init__()
    self.weight = weight
    self.ce = BatchedCrossEntropy()
    self.ood_entropy = MeanSoftmaxEntropy()
  
  def forward(self, model_bs, x, forward_data, gt):
    """ Independently calculate the cross entropy loss for each model in the batch

    Args:
      model_bs (torch.nn.Module): The model batch size (useful for batched models)
      x (torch.Tensor): The input data
      forward_data (dict): The predictions of the model
      gt (torch.Tensor): The ground truth labels
    """
    return self.ce(model_bs, x, forward_data['pred_ind'], gt) + self.weight * self.ood_entropy(model_bs, x, forward_data['pred_ood'], None)


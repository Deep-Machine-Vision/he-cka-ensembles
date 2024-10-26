from typing import List, Union
import torch.nn as nn
import torch
import copy


AVAILABLE_WEIGHTING = {}

def register_weighting(name: str):
  """ Decorator to register a model """
  def decorator(cls):
    AVAILABLE_WEIGHTING[name] = cls
    return cls
  return decorator


def build_weighting(config: dict):
  """ Builds a model """
  config = copy.deepcopy(config)
  
  try:
    name = config.pop('name')
  except KeyError as err:
    raise RuntimeError('Method configurations not found in configs. Must define a method and contain the name of the method as a key in that dictionary')
  
  cls = AVAILABLE_WEIGHTING[name]
  config = cls.from_config(config)
  return cls(**config)


class LayerWeighting(nn.Module):
  def __init__(self) -> None:
    """ Class to calculate the weighting of each layer in a model. """
    super(LayerWeighting, self).__init__()
    self.total = None
  
  def get_weight(self, layer: int, total: int, norm: bool=True):
    raise NotImplementedError("Must implement get_weight")

  def total_sum(self, total: int):
    """ Get the total sum of the weights """
    return sum(self.get_weight(i, total, norm=False) for i in range(total))
  
  def save_total(self, total: int):
    """ Calculate the sum of the weights """
    self.total = self.total_sum(total)
  
  def save_total_if_needed(self, total: int):
    """ Save the total if it has not been saved """
    if self.total is None:
      self.save_total(total)

  def reset_total(self):
    """ Reset the total """
    self.total = None

  @staticmethod
  def from_config(config: dict):
    return config


@register_weighting("tensor")
class TensorLayerWeighting(LayerWeighting):
  def __init__(self, weights: Union[List[float], torch.Tensor], normalize: bool=True, learnable: bool=False, eps: float=1e-6) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    Useful to ensure that the weights are saved as parameters and gradients can be calculated.
    
    Args:
      weights (List[float]): The weights for each layer
      normalize (bool, optional): Whether to normalize the weights
      eps (float, optional): The epsilon value for normalization
    """
    super(TensorLayerWeighting, self).__init__()
    
    # if weights is int then specify uniform weights of length weights
    if isinstance(weights, int):
      weights = [1.0 / weights] * weights
    
    if not isinstance(weights, (torch.Tensor, list)):
      raise ValueError("Weights must be a tensor or a list")
    if isinstance(weights, list):
      weights = torch.tensor(weights, dtype=torch.float32, requires_grad=learnable)
    else:
      weights = weights.clone().detach().requires_grad_(learnable)
    self.weights = nn.Parameter(weights, requires_grad=learnable)
    self.norm = normalize
    self.eps = eps
  
  def get_weight(self, layer: int, total: int, norm: bool=True):
    """ Get the weight for a given layer """
    if layer >= len(self.weights):
      raise ValueError("Layer {} not in weights".format(layer))
    
    # normalize if needed
    if self.norm and norm:
      self.save_total_if_needed(total)
      return self.weights[layer] / self.total
    return self.weights[layer]

  def total_sum(self, total: int):
    """ Get the total sum of the weights """
    return torch.clamp(self.weights.sum(), min=self.eps)
  
  def save_total(self, total: int):
    """ Calculate the sum of the weights """
    self.total = self.total_sum(total)

  def reset_total(self):
    """ Reset the total. Also resetting grad! """
    
    # renormalize weights if so also 
    self.weights.grad = None
    with torch.no_grad():
      if self.norm and self.total is not None:
        self.weights.data = torch.abs(self.weights.data)  # project to positive
        self.weights.data = self.weights.data / self.total_sum(None)
    
    self.total = None


@register_weighting("list")
class ListLayerWeighting(LayerWeighting):
  def __init__(self, weights: Union[List[float], torch.Tensor]) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    This is the most simple class that just takes a list of weights for each layer.
    The number of layers must be known in advance.
    
    Args:
      weights (List[float]): The weights for each layer
    """
    super(ListLayerWeighting, self).__init__()
    
    if weights is None:
      raise ValueError("Weights must be a list")
    
    self.weights = weights
  
  def get_weight(self, layer: int, total: int, norm: bool=True):
    """ Get the weight for a given layer """
    if layer >= len(self.weights):
      raise ValueError("Layer {} not in weights".format(layer))
    
    if norm:
      self.save_total_if_needed(total)
      return self.weights[layer] / self.total  # get normalized variant
    return self.weights[layer]


@register_weighting("linear")
class LinearLayerWeighting(LayerWeighting):
  def __init__(self, start_weight: float=1.0, weight_increase: float=1.0) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    This class linearly interpolates between the start and end weights. Where weight_increase determines the rate of change.
    The number of layers does not need to be known in advanced.
    
    Args:
      start_weight (float, optional): The weight for the first layer
      weight_increase (float, optional): The rate of change of the weights
    """
    super(LinearLayerWeighting, self).__init__()
    self.start = start_weight
    self.increase = weight_increase

  def get_weight(self, layer: int, total: int, norm: bool=True):
    value = (self.start + self.increase * layer)
    if norm:
      self.save_total_if_needed(total)
      return value / self.total
    return value


@register_weighting("uniform")
class UniformLayerWeighting(LinearLayerWeighting):
  def __init__(self) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    This class uniformly sets the weights for each layer.
    """
    super(UniformLayerWeighting, self).__init__(1.0, 0.0)


@register_weighting("exponential")
class ExponentialLayerWeighting(LayerWeighting):
  def __init__(self, start_weight: float=1.0, weight_increase: float=1.0) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    This class exponentially interpolates between the start and end weights. Where weight_increase determines the rate of change.
    The number of layers does not need to be known in advanced.
    
    Args:
      start_weight (float, optional): The weight for the first layer
      weight_increase (float, optional): The rate of change of the weights
    """
    super(ExponentialLayerWeighting, self).__init__()
    self.start = start_weight
    self.increase = weight_increase

  def get_weight(self, layer: int, total: int, norm: bool=True):
    value = (self.start * self.increase ** layer)
    if norm:
      self.save_total_if_needed(total)
      return (self.start * self.increase ** layer) / self.total
    return value


@register_weighting("start_end_linear")
class StartEndLinearLayerWeighting(LayerWeighting):
  def __init__(self, first_layer: float, middle_start: float, middle_increase: float, last_layer: float=None) -> None:
    """ Class to calculate the weighting of each layer in a model.
    
    This class linearly interpolates between the start and end weights. Where weight_increase determines the rate of change.
    The number of layers does not need to be known in advanced. Sets the first and last layers as something different.
    
    Args:
      first_layer (float): The weight for the first layer
      middle_start (float): The weight for the middle layers
      middle_increase (float): The rate of change of the middle weights
      last_layer (float): The weight for the last layer
    """
    super(StartEndLinearLayerWeighting, self).__init__()
    self.first = first_layer
    self.middle_start = middle_start
    self.increase = middle_increase
    self.last = last_layer
  
  def get_weight(self, layer: int, total: int, norm: bool=True):
    if layer == 0:
      value = self.first
    elif layer == total - 1 and self.last is not None:
      value = self.last
    else:
      value = (self.middle_start + self.increase * (layer - 1))

    if norm:
      self.save_total_if_needed(total)
      return value / self.total
    return value

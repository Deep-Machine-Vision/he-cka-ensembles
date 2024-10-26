""" Activation functions and respective gain calculations for NF networks """
import torch
import torch.nn as nn
import math


# functional version of crater
SQRT2OVER2 = math.sqrt(2) / 2.0
COEFFIX = SQRT2OVER2 + 0.008274364699
def crater(x):
  """ Activation function of the crater (similar to GELU) """
  return COEFFIX * x * (1.0 + torch.tanh(SQRT2OVER2 * (x + 1.0)))


class Crater(nn.Module):
  """ Simple module wrapper for crater activation function """
  def forward(self, x):
    return crater(x)


def softstep(x, k=0.2, clamp=True):
  step = torch.floor(x).detach()
  
  # smooth steps
  x = 0.5 - (x - step)
  s = torch.sign(x).detach()
  x = torch.abs(x) * 2.0  # torch.clamp(torch.abs(x) * 2.0, -1e-9, 1.0 + 1e-9)
  
  # return smooth
  y = step + 0.5 + 0.5*(s * x / (x * (k - 1.0) - k))
  if clamp:
    return torch.clamp(y, 0.0)
  return y


def leakysoftstep(x, k=0.55, alpha=0.1):
  # return smooth
  y = softstep(x, k=k, clamp=False)
  return torch.where(y > 0, y, alpha*x)


class LeakySoftStep(nn.Module):
  """ Simple module wrapper for crater activation function """
  def forward(self, x):
    return leakysoftstep(x)


def activation_gamma(act: str):
  """ Gathers the variance scaling gamma used for the specified function

  Args:
    func (str): name of activation function or method itself if it's from torch.nn.function.act.__name__ 
  """
  # if none then assume identity
  if act is None:
    return 1.0

  # assumed already define
  if isinstance(act, (int, float)):
    return float(act)

  # convert from class to name
  maps = {
    nn.ReLU: 'relu',
    nn.LeakyReLU: 'leaky_relu',
    nn.GELU: 'gelu',
    Crater: 'crater'
  }

  if isinstance(act, type):
    if act in maps:
      act = maps[act]

  if isinstance(act, str):
    if act == 'crater':
      return 1.0000000001  # fairly close to 1/not really necessary
    elif act == 'gelu':
      return 1.70092624336
    elif act == 'leaky_relu':
      return 1.70483162718
    elif act == 'relu':
      return math.sqrt(2.0 / (1.0 - (1.0 / math.pi)))
    return 1.0
  raise ValueError(f'The provided activation {act} is not a valid choice for gamma calculation. Please add it to net/activation.py')

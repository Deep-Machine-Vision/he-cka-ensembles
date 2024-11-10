""" Contains some fun general utilities """
from . import collections
# from ..nfnet import models
from argparse import Namespace
from typing import Generator, Iterable, Iterator, Optional, Tuple, Union
from torch import TensorType
from tqdm import tqdm
from torch import DeviceObjType
import torch.optim as optim
import numpy as np
import argparse
import math
from bisect import bisect_right, bisect_left
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import numpy as np
import torch
import random

LoopType = Generator[Tuple[int, Tuple[TensorType, TensorType]], None, None]
DeviceType = Union[str, DeviceObjType]


def null_fn(*args, **kwargs):
  """ Does nothing """
  pass


def set_seed(seed):
  """ Sets the seeds """
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def tqdm_train(iter: Iterable) -> LoopType:
  """ Returns the customized tqdm call for a training loop """
  return tqdm(enumerate(iter), desc='Training ', total=len(iter), colour='green', unit='batch')


def tqdm_test(iter: Iterable) -> LoopType:
  """ Returns the customized tqdm call for a testing loop """
  return tqdm(enumerate(iter), desc='Testing ', total=len(iter), colour='blue', unit='batch')


def namespace_to_dict(namespace: Namespace) -> dict:
  """ Converts an argparse Namespace into a dictionary

  Args:
      namespace (Namespace): the Namespace of arguments

  Returns:
      dict: the same Namespace represented as a dictionary
  """
  return {
      k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
      for k, v in vars(namespace).items()
  }

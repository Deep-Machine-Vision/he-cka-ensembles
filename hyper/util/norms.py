import torch
import torch.linalg


def unitwise_norm(x: torch.Tensor):
  # got from nfnets-pytorch https://github.com/vballoli/nfnets-pytorch/blob/main/nfnets/utils.py
  if x.ndim <= 1:
      dim = 0
      keepdim = False
  elif x.ndim in [2, 3]:
      dim = 0
      keepdim = True
  elif x.ndim == 4:
      dim = [1, 2, 3]
      keepdim = True
  else:
      raise ValueError('Wrong input dimensions')

  return  torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=keepdim)


def batched_unitwise_norm(x):
  """ Computes unitwise norm of a batched tensor """
  
  if x.ndim == 0: # cannot do anything with only batch dim
    raise ValueError('cannot compute batched norm of a scalar')
  
  # first dim is always batch
  if x.ndim == 1:  # batched scalars (just return self)
    return x
  if x.ndim == 2:  # batched vectors
    dim = 1
  elif x.ndim in [3, 4]:  # batched matrices
    dim = 1
  elif x.ndim == 5:  # batched 2d convs
    dim = [2, 3, 4]
  else:
    raise ValueError('Wrong input dimensions')
  
  return torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
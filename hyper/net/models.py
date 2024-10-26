""" Contains basic model building function """
from typing import List, Union, Tuple
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init
import math

from .gpnorm import calculate_mlp_groupnorm_groups
from .activation import Crater, activation_gamma
from .spectral import spectral_norm_fc
from .linear import ScaledWLinear


def build_nfnet_mlp(in_features: int, out_features: int, mlp_dims: Union[int, List[int]]=3, dim_multiplier: float=2, bias: bool=False, gamma=1.0, activation=nn.GELU, out_act=nn.GELU) -> nn.Module:
  """ Builds a basic MLP model using the specified activation/dimensions

  Args:
    in_features (int): number of input features to the MLP
    out_features (int): number of output features of the MLP
    mlp_dims (Union[int, List[int]]): the center dimensions of the layer mlp (including first layer excluding the last linear to project to parameter space). If it's an int it's just the number of layers to generate (look at mlp_multiplier), else it's a list of dimensions
    dim_multiplier (float, optional): if mlp_dims is an integer then use this multiplier to scale the number of parameters of each layer up linearly from latent_size*mlp_multiplier (only use if mlp_dims is int). Defaults to 2.
    bias (bool, optional): to use the bias term in the mlp. Defaults to False.
  
  Note: the output layer does NOT contain an activation/norm
  """
  # define a multiplier from the latent size
  if isinstance(mlp_dims, (int, float)):
    if isinstance(mlp_dims, float):
      mlp_dims = round(mlp_dims)
    current_dims = in_features * dim_multiplier
    dims = [round(current_dims)]
    for i in range(1, mlp_dims):
      dims.append(round(mlp_dims[i - 1] * dim_multiplier))
    mlp_dims = dims

  # from the latent dim to first out
  layers = OrderedDict([
    ('input_linear', ScaledWLinear(in_features, mlp_dims[0], bias=bias, gamma=gamma)),
    # ('input_norm', nn.GroupNorm(4 if mlp_dims[0] % 2 == 0 else 1, mlp_dims[0])),
    ('input_act', activation())
  ])

  # center mlp dims
  for ind in range(1, len(mlp_dims)):
    layers[f'center_linear_{ind}'] = ScaledWLinear(mlp_dims[ind - 1], mlp_dims[ind], bias=bias, gamma=activation)
    # layers[f'center_norm_{ind}'] = nn.GroupNorm(4 if mlp_dims[ind] % 2 == 0 else 1, mlp_dims[ind])
    layers[f'center_act_{ind}'] = activation()

  # layers[f'output_norm'] = nn.GroupNorm(1 if out_features % 2 == 0 else 1, out_features)
  layers[f'output_linear'] = nn.Linear(mlp_dims[-1], out_features, bias=bias)  # , gamma=activation)
  init.normal_(layers[f'output_linear'].weight, 0.0, activation_gamma(activation) * math.sqrt(1.0 / mlp_dims[-1]))
  if layers[f'output_linear'].bias is not None:
    init.normal_(layers[f'output_linear'].bias, 0.0, activation_gamma(activation) * math.sqrt(1.0 / mlp_dims[-1]))
  if out_act is not None:
    layers[f'output_act'] = out_act()
  return nn.Sequential(layers)


def build_norm_block(in_features: int, out_features: int, bias: bool, act: object, norm: bool, gamma: object, norm_groups: int=None, eps=1e-5, affine: bool=True, coeff: float=None):
  """ Internal builds a norm block see build_groupnorm_mlp for more details/parameter meaning """
  
  linear = nn.Linear(
    in_features=in_features,
    out_features=out_features,
    bias=bias
  )
  
  # use a spectrally normalized linear layer
  if coeff is not None:
    linear = spectral_norm_fc(linear, coeff, n_power_iterations=1)

  # apply initialization similar to scaled W except just for init
  with torch.no_grad():
    # unit variance gaussian initialization (variance fixed later for weights)
    invsq = math.sqrt(1.0 / in_features)
    init.normal_(linear.weight, 0.0, 1.0)
    if bias:
      init.normal_(linear.bias, 0.0, invsq)

    # correct the sampled weights expectation and variance (minor fix)
    mean = torch.mean(linear.weight, dim=1, keepdim=True)
    std = torch.std(linear.weight, dim=1, keepdim=True, unbiased=False)

    # update weight
    linear.weight.sub_(mean)
    linear.weight.mul_((invsq * activation_gamma(gamma)) / (std + eps))

  # basic/dumb way to pick number of groups
  # seems like 32 groups was good from original paper https://arxiv.org/pdf/1803.08494.pdf
  # and the best also was roughly 16 channels per group

  return nn.Sequential(
    OrderedDict([
        ('linear', linear),
      ] + (
        [
          ('norm', nn.GroupNorm(
            num_groups=calculate_mlp_groupnorm_groups(out_features) if norm_groups is None else norm_groups,
            num_channels=out_features,
            affine=affine
          ))
        ] if norm else []
      ) + (
        [
          ('act', act())
        ] if act is not None else []
      )
    )
  )

def build_groupnorm_mlp(in_features: int, out_features: int, mlp_dims: Union[int, List[int]]=3, dim_multiplier: float=2, bias: bool=False, gamma=1.0, norm_every: int=2, norm_before_last: bool=False, norm_last: bool=True, norm_last_groups: int=None, affine_last: bool=True, activation=Crater, out_act=Crater, coeff: float=None) -> nn.Module:
  """ Builds a basic MLP model using groupnorm layers with STD initialized weights the specified activation/dimensions

  Args:
    in_features (int): number of input features to the MLP
    out_features (int): number of output features of the MLP
    mlp_dims (Union[int, List[int]]): the center dimensions of the layer mlp (including first layer excluding the last linear to project to parameter space). If it's an int it's just the number of layers to generate (look at mlp_multiplier), else it's a list of dimensions
    dim_multiplier (float, optional): if mlp_dims is an integer then use this multiplier to scale the number of parameters of each layer up linearly from latent_size*mlp_multiplier (only use if mlp_dims is int). Defaults to 2.
    bias (bool, optional): to use the bias term in the mlp. Defaults to False.
    norm_every (int): apply group norm after every n layers.
    norm_before_last (bool): apply a group norm right before last MLP (don't also set norm_last to true)
    norm_last (bool): apply a group norm right after before last MLP (usually good to ensure variance on output)
    norm_last_groups (int): the number of groups to have on output norm (if norm_last=True). Default autopick using some "randomly" found values.
    
  Note: the output layer does NOT contain an activation/norm
  """
  # define a multiplier from the latent size
  if isinstance(mlp_dims, (int, float)):
    if isinstance(mlp_dims, float):
      mlp_dims = round(mlp_dims)
    current_dims = round(in_features * dim_multiplier)
    dims = [current_dims]
    for i in range(1, mlp_dims):
      dims.append(round(dims[i - 1] * dim_multiplier))
    mlp_dims = dims

  # from the latent dim to first out
  layers = nn.Sequential()
  layers = OrderedDict([
    ('input', build_norm_block(
      in_features=in_features,
      out_features=mlp_dims[0],
      bias=bias,
      act=activation,
      norm=True,
      gamma=gamma,
      coeff=coeff
    ))
  ])

  # center mlp dims
  for ind in range(1, len(mlp_dims)):
    use_norm = (ind % norm_every == 0)
    layers[f'center_{ind}'] = build_norm_block(
      in_features=mlp_dims[ind - 1],
      out_features=mlp_dims[ind],
      bias=bias,
      gamma=activation,
      act=activation,
      norm=use_norm,
      coeff=coeff
    )


  # ensure no double norms
  assert not (norm_before_last and norm_last), 'Double norming please only set one to true'

  if norm_before_last:
    layers[f'group_norm'] = nn.GroupNorm(
      num_groups=calculate_mlp_groupnorm_groups(mlp_dims[-1]),
      num_channels=mlp_dims[-1],
      affine=True
    )

  # build last projection/output layer
  layers[f'output'] = build_norm_block(
    in_features=mlp_dims[-1],
    out_features=out_features,
    bias=bias,
    act=out_act,
    gamma=activation,
    norm=norm_last,
    affine=affine_last,
    norm_groups=1,  # do layernorm on output
    coeff=coeff
  )
  return nn.Sequential(layers)


def old_build_mlp(in_features: int, out_features: int, mlp_dims: Union[int, List[int]]=3, dim_multiplier: float=2, bias: bool=False, activation=nn.GELU, out_act=nn.GELU) -> nn.Module:
  """ Builds a basic MLP model using the specified activation/dimensions

  Args:
    in_features (int): number of input features to the MLP
    out_features (int): number of output features of the MLP
    mlp_dims (Union[int, List[int]]): the center dimensions of the layer mlp (including first layer excluding the last linear to project to parameter space). If it's an int it's just the number of layers to generate (look at mlp_multiplier), else it's a list of dimensions
    dim_multiplier (float, optional): if mlp_dims is an integer then use this multiplier to scale the number of parameters of each layer up linearly from latent_size*mlp_multiplier (only use if mlp_dims is int). Defaults to 2.
    bias (bool, optional): to use the bias term in the mlp. Defaults to False.
  
  Note: the output layer does NOT contain an activation/norm
  """
  # define a multiplier from the latent size
  if isinstance(mlp_dims, int):
    current_dims = in_features * dim_multiplier
    dims = [current_dims]
    for i in range(1, mlp_dims):
      dims.append(mlp_dims[i - 1] * dim_multiplier)
    mlp_dims = dims

  # from the latent dim to first out
  lin = nn.Linear(in_features, mlp_dims[0], bias=bias)
  print('first', in_features, mlp_dims[0])
  
  # apply kaiming
  # init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
  # if bias:
  #   init.zeros_(lin.bias)

  layers = OrderedDict([
    ('input_linear', lin),
    ('input_act', nn.Tanh()), 
    # ('input_norm', nn.GroupNorm(4 if mlp_dims[0] % 2 == 0 else 1, mlp_dims[0]))
  ])

  # center mlp dims
  for ind in range(1, len(mlp_dims)):
    lin = nn.Linear(mlp_dims[ind - 1], mlp_dims[ind], bias=bias)
    print('center', mlp_dims[ind - 1], mlp_dims[ind])

    # apply kaiming
    init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
    if bias:
      init.zeros_(lin.bias)

    layers[f'center_linear_{ind}'] = lin
    layers[f'center_act_{ind}'] = activation()
    # layers[f'center_norm_{ind}'] = nn.GroupNorm(4 if mlp_dims[ind] % 2 == 0 else 1, mlp_dims[ind])

  # create final output linear to project to target layer parameters
  lin = nn.Linear(mlp_dims[-1], out_features, bias=bias)

  # apply kaiming
  if out_act is not None:
    pass
    # init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
    # if bias:
    #   init.zeros_(lin.bias)
  else:
    feat_in = lin.weight.shape[1]
    # k = math.sqrt(2) * math.sqrt(2 / feat_in)
    init.xavier_uniform_(lin.weight, 1.0)
    # init.uniform_(lin.weight, -k, k)
    # if bias:
    #   init.zeros_(lin.bias)  # , -k, k)
    # init.xavier_uniform_(lin.weight, gain=1.0)
    # if bias:
    #   init.zeros_(lin.bias)

  layers[f'output_linear'] = lin
  if out_act is not None:
    layers[f'output_act'] = out_act()
  return nn.Sequential(layers)
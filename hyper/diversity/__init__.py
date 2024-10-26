""" Functions to calculate CKA between two feature matrices """
import torch
from .kernels import batched_gram, center_gram, vectorize_grams


def unit_vector_rows(features: torch.Tensor, eps: float=1e-6, vectorize: bool=True) -> torch.Tensor:
  """ Converts the batched feature matrices into a new family of feature matrices that are vectorized with unit norms

  Args:
    features (torch.Tensor): [B, N*P] or [B, N, P] batched feature matrix
    eps (float): if feature matrix has close to zero Frobenius norm add a small constant to prevent zero division
    vectorize (bool): if the shape is [B, N, P] (assumed to be a gram matrix) then vectorize the grams. If False then it will just throw an error
  Returns:
    torch.Tensor: [B, N*P]
  """

  if features.ndim == 3 and vectorize:
    features = vectorize_grams(features)
  
  if features.ndim != 2:
    raise ValueError('Expected matrix of [B, P]')

  # get Frobenius of norm each gram, ie euclidean of vectorized form
  norms = torch.linalg.vector_norm(features, ord=2, dim=1, keepdim=True)

  # if not isinstance(eps, float):
  #   eps = eps.view(eps.shape[0], 1, 1)

  # create unit norm gram matrices
  return features.divide(norms + eps)


def pairwise_cossim(grams: torch.Tensor, eps: float=1e-6, reduction: str='off_diag', detach_right: bool=False) -> torch.Tensor:
  """ Computes pairwise cosine similarity between batched grams

  Args:
    grams (torch.Tensor): [B, N, P] batched gram matrix
    eps (float): if gram matrix has close to zero Frobenius norm add a small constant to prevent zero division

  Returns:
    torch.Tensor: vector of shape [B*(B-1)/2] with the pairwise cossine similarity  
  """
  B = grams.shape[0]
  unit_vec_grams = unit_vector_rows(grams, eps)

  # compute pairwise dot products (just upper triangle of inner prod)
  gram_pd = torch.matmul(unit_vec_grams, unit_vec_grams.t().detach() if detach_right else unit_vec_grams.t())
  
  # get indices of upper right triangle
  # rows, cols = torch.triu_indices(B, B, offset=1)

  # only return relevent pairwise cossim values
  # gram_pd = gram_pd[rows, cols]
  if reduction == 'off_diag':
    gram_pd = gram_pd[~torch.eye(*gram_pd.shape,dtype = torch.bool)]
  return gram_pd


def pairwise_cka(features: torch.Tensor, kernel: str='linear', detach_diag: bool=True, *args, **kwargs):
  """ Creates batched gram matrix

  Args:
    features (torch.Tensor): model batched features of shape [B, N, P] where B is number of models, N is batch size of input, P is the number of features  
    kernel (str): the type of kernel. Default is linear
    detach_diag (bool): detaches diagonal elements from backward
    *args, **kwargs: all other args are passed to specific gram function

  Returns:
    torch.Tensor: [B*(B-1)/2] pairwise cka values
  """
  
  # construct batch of gram matrices
  cgrams = batched_gram(features, features, *args, kernel=kernel, detach_diag=detach_diag, center=True, **kwargs)

  # compute pairwise cossim
  return pairwise_cossim(cgrams)


def hyperspherical_energy(features: torch.Tensor, s: float=0.0, half_space: bool=False, eps: float=3e-5, arc_eps: float=3e-4, offset: float=0.0, reduction: str ='mean', remove_diag: bool=True, detach_right: bool=False, abs_vals: bool=False, use_exp: bool=False):
  """ Calculates the minimum geodesic hyperspherical energy

  Args:
    features (torch.Tensor): model batched features of shape [B, N, P] or [B, N*P] where B is number of models, N is batch size of input, P is the number of features  
    s (float): the Riesz s kernel parameter (for geodesic dist)
    half_space (bool): when True add negated gram vectors to use half-space MHE, if "centered" features are always positive this should be set to False.
    eps (float): value to prevent zero division with/normalize some division operators
    arc_eps (float): value to prevent arccos(theta) have nan/inf results (and their gradients exploding) by smoothing out theta just slightly by theta/(1 + arc_eps)
  Returns:
    torch.Tensor: geodesic hyperspherical energy
  """

  # create unit vector grams
  B = features.shape[0]
  # N = grams.shape[1]
  unit_vec_grams = unit_vector_rows(features, eps=eps)

  # double gram features but with negated
  if half_space:
    unit_vec_grams = torch.concat(
      [
        unit_vec_grams,
        -unit_vec_grams.detach()
      ],
      dim=0
    )

  # cossim (cos(theta)) between vectors
  costheta = torch.matmul(unit_vec_grams, unit_vec_grams.t().detach() if detach_right else unit_vec_grams.t())
  
  if abs_vals:
    costheta = torch.abs(costheta)
  
  # rows, cols = torch.triu_indices(B, B, offset=1)
  # gram_pd = geodesics[rows, cols]  # now just a tensor of the isolated pairwise
  # return (1.0 / (torch.arccos(gram_pd) + 1e-5)).mean()

  # get off diagonal geodesic elements only (i != j)
  if remove_diag:
    off_diag = costheta[~torch.eye(*costheta.shape,dtype = torch.bool)]
  else:
    off_diag = costheta
  # nij_geodesics = torch.arccos((off_diag - offset) / (1.0 + arc_eps))
  nij_geodesics = torch.arccos(off_diag / (1.0 + arc_eps))

  # print('APS', 1.0 + arc_eps)
  # print('GEOD', nij_geodesics, 'COSTHE', off_diag / (1.0 + arc_eps), torch.abs(off_diag / (1.0 + arc_eps)) > 1.0)

  # calculate energy from geodesic using rho
  # when s=1.0 this is similar to Coloumb's force with unit mass/force constant 
  # if 0 we just get log of inverse
  if s == 0.0 or s == 1.0:
    energy = 1.0 / (nij_geodesics + eps)

    # apply logarithm on energy
    if s == 0.0:
      energy = torch.log(energy + eps)
  else:
    # energy = torch.pow(nij_geodesics, -torch.tensor(s, device=nij_geodesics.device, dtype=nij_geodesics.dtype))
    # energy = (1.0 + eps) / (torch.pow(nij_geodesics + eps, torch.tensor(s, device=nij_geodesics.device, dtype=nij_geodesics.dtype)) + eps)
    # energy = torch.exp(-(3.6 * nij_geodesics))  # - 0.2))
    if use_exp:
      energy = torch.exp(-(s * nij_geodesics + eps))  # - 0.2))
    else:
      energy = (1.0 + eps) / (torch.pow(nij_geodesics, torch.tensor(s, device=nij_geodesics.device, dtype=nij_geodesics.dtype)) + eps)

    # print('ENERGYYY', energy)

  # print('geod', torch.any(nij_geodesics < (1.0 - 1e-10)))
  # print('fin', torch.isfinite(torch.sum(energy)))

  # sum and normalize by number of pairings, which is just
  # the mean of all energies
  # if reduction == 'mean':
  #   return energy.mean()
  return energy


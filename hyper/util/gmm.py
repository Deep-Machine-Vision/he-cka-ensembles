""" Fits Gaussian Mixture Models

Modified from: https://github.com/omegafragger/DDU
"""
import torch
from torch import nn
from tqdm import tqdm
from hyper.generators.base import ModelGenerator
from hyper.util.collections import flatten_keys


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-200, 0, 1)]


def centered_cov_torch(x):
  n = x.shape[0]
  res = 1 / (n - 1) * x.t().mm(x)
  return res


def forward_features(hyper: ModelGenerator, params, features, X, concat_feat=True):
  """ Gets the batched features from model generator """
  feat, _ = hyper.forward(params, X)
  ffeat = flatten_keys(feat)[features]  # get the specified features

  '''
  if torch.any(torch.isnan(ffeat)):
    model_ind = ffeat.sum(dim=[1, 2]).isnan()[0]
    # print('BAD', ffeat[model_ind])

    f = flatten_keys(feat)
    ind = 0
    for k, v in f.items():
      if v is None:
        continue
      print(f'Layer {k}', v[model_ind], params[f'{ind}.self'][model_ind])
      ind += 1
    print('INPUT', torch.sum(X).isnan(), torch.max(X))
  '''

  # current shape should be [model bs, bs, out features] combine to [bs, model bs * out features]
  if concat_feat:
    # ffeat = ffeat.transpose(0, 1).reshape(ffeat.shape[1], ffeat.shape[0]*ffeat.shape[2])
    ffeat = ffeat.reshape(ffeat.shape[0]*ffeat.shape[1], ffeat.shape[2])   # ffeat.mean(0)

  # replace nans
  ffeat[torch.isnan(ffeat)] = 0.0

  return ffeat


def get_embeddings(hyper, params, features, mbs, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device, concat_feat=True):
  num_samples = len(loader.dataset)
  embeddings = torch.empty((mbs * num_samples, num_dim), dtype=dtype, device=storage_device)
  labels = torch.empty(mbs * num_samples, dtype=torch.int, device=storage_device)

  with torch.no_grad():
    start = 0
    for data, label in tqdm(loader):
      data = data.to(device)
      label = label.to(device)

      # if isinstance(hyper, nn.DataParallel):
      #   out = hyper.module(data)
      #   out = hyper.module.feature
      # else:
      #   out = hyper(data)
      #   out = hyper.feature
      out = forward_features(hyper, params, features, data, concat_feat)

      end = start + (len(data) * mbs)
      embeddings[start:end].copy_(out, non_blocking=True)
      # print(start, end, end - start, label.shape, label.repeat(mbs).shape)
      labels[start:end].copy_(label.repeat(mbs), non_blocking=True)
      start = end

  return embeddings, labels


@torch.no_grad()
def gmm_forward(hyper: ModelGenerator, params, features, labels, gaussians_model, data_B_X):
  # if isinstance(hyper, nn.DataParallel):
  #   features_B_Z = hyper.module(data_B_X)
  #   features_B_Z = hyper.module.feature
  # else:
  #   features_B_Z = hyper(data_B_X)
  #   features_B_Z = hyper.feature
  features_B_Z = forward_features(hyper, params, features, data_B_X)
  good_batches = torch.isfinite(torch.sum(features_B_Z, dim=-1))
  features_B_Z = features_B_Z[good_batches]
  labels = labels[good_batches]
  
  if len(features_B_Z) == 0:
    raise ValueError('Bad batch...')
  if not torch.sum(features_B_Z).isfinite():  # bad batch
    raise ValueError('Bad batch...')
  
  torch.cuda.synchronize()
  log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])
  return log_probs_B_Y, labels


@torch.no_grad()
def gmm_evaluate(hyper, params, features, mbs, gaussians_model, loader, device, num_classes, storage_device, concat_feat=True):
  num_samples = len(loader.dataset)
  logits_N_C = torch.empty((mbs * num_samples, num_classes), dtype=torch.float, device=storage_device)
  labels_N = torch.empty(mbs * num_samples, dtype=torch.int, device=storage_device)

  with torch.no_grad():
    start = 0
    for data, label in tqdm(loader):
      torch.cuda.synchronize()
      try:
        data = data.clone().to(device)
        label = label.clone().to(device)
      except RuntimeError as err:
        print('CUDA err', str(err))

      try:
        logit_B_C, label = gmm_forward(hyper, params, features, label.repeat(mbs), gaussians_model, data)
        # print(logit_B_C.shape, logits_N_C.shape)
        if logit_B_C.shape[0] < len(data):
          print('Lost', len(data) - logit_B_C.shape[0])

        end = start + logit_B_C.shape[0]
        logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
        labels_N[start:end].copy_(label, non_blocking=True)
        start = end
      except ValueError as err:
        print(err)

  # cut the bad batches out
  return logits_N_C[:start], labels_N[:start]


def gmm_get_logits(gmm, embeddings):
  log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
  return log_probs_B_Y


# @TODO tomorrow
# fit multiple GMMs one for each ensemble member via https://link.springer.com/chapter/10.1007/978-3-540-28651-6_98
# it is as simple as meaning all embeddings and whatever for covariance fit
def gmm_fit(embeddings, params, features, labels, num_classes):
  with torch.no_grad():
    classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
    classwise_cov_features = torch.stack(
      [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
    )

  with torch.no_grad():
    for jitter_eps in JITTERS:
      try:
        jitter = jitter_eps * torch.eye(
          classwise_cov_features.shape[1], device=classwise_cov_features.device,
        ).unsqueeze(0)
        gmm = torch.distributions.MultivariateNormal(
          loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
        )
      except RuntimeError as e:
        if "cholesky" in str(e):
          continue
      except ValueError as e:
        if "found invalid values" in str(e):
          continue
      break

  return gmm, jitter_eps


def train_gmm(model, params, features, mbs, num_dim, train_loader, device):
  embeddings, labels = get_embeddings(model,
                                      params,
                                      features,
                                      mbs,
                                      train_loader,
                                      num_dim,
                                      dtype=torch.double,
                                      device=device,
                                      storage_device=device)
  gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings,
                                        params=params,
                                        features=features,
                                        labels=labels,
                                        num_classes=10)
  return gaussians_model


def get_gmm_logits(model, params, features, mbs, gmm_model, data_loader, device):
  gmm_logits, gmm_labels = gmm_evaluate(model,
                                        params,
                                        features,
                                        mbs,
                                        gmm_model,
                                        data_loader,
                                        device=device,
                                        num_classes=10,
                                        storage_device=device)
  return gmm_logits

""" Handles loading data for most experiments """
from typing import Tuple, Optional, Union
from torchvision import datasets, transforms
from hyper.data.fast_mnist import FastMNIST, DATA_DIR
from hyper.data.fast_cifar import FastCIFAR10, FastCIFAR100
# from hyper.data.tinyimagenet import TinyImageNetDataset, TINYIMG_MEAN, TINYIMG_STD  # TODO finish writing this in the code rewrite
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, Subset
from hyper.data.util import ddp_args
from hyper.data.image_ood import get_mnist_ood_loader
from hyper.data.dirty_mnist import get_train_valid_loader as dmnist_loader
import torch

from os import PathLike
import os
import os.path
import numpy as np
import sklearn.datasets as skdatasets
import multiprocessing

MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 8))
if MAX_WORKERS > 0:
  DEFAULT_DEVICE = 'cpu'  # workers can't init cuda
else:
  DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

MNIST_LOADER_ARGS = {
  'num_workers': min(MAX_WORKERS, multiprocessing.cpu_count()),
  'pin_memory': not torch.cuda.is_available
}

# normalization params for MNIST
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# normalization for CIFAR
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CIFAR_LOADER_ARGS = {
  'num_workers': min(MAX_WORKERS, multiprocessing.cpu_count()),
  'pin_memory': not torch.cuda.is_available
}

# normalization for SVNH
SVNH_MEAN = CIFAR_MEAN
SVNH_STD = CIFAR_STD
SVNH_LOADER_ARGS = CIFAR_LOADER_ARGS

# tinyimagenet loader
TINYIMG_LOADER = CIFAR_LOADER_ARGS
TINYIMG_DATA_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200')

# dtd loader
DTD_MEAN = [0.485, 0.456, 0.406]
DTD_STD = [0.229, 0.224, 0.225]


def load_mnist(root: Optional[PathLike]=DATA_DIR, bs: int=100, drop_last=False, dirty=False, ddp=False, **extra) -> Tuple[DataLoader, DataLoader]:
  """ Handles loading the mnist train/test (if specified)

  Args:
      root (PathLike): the directory where to save the dataset 
      bs (int): batch size of loader. Default to 100
      dirty (bool): load the dirty MNIST variant
  Returns:
      Tuple[DataLoader, DataLoader]: returns a tuple of the (train DataLoader, test DataLoader)
  """

  bs_ood = extra.get('bs_ood', None)
  if bs_ood is not None:
    ood_loader = get_mnist_ood_loader(
      batch_size=bs_ood,
      id_prob=0.35,
      ddp=ddp
    )


  # if dirty load that variant
  if dirty:
    train_loader, test_loader = dmnist_loader(
      batch_size=bs,
      ddp=ddp
    )
    
    if bs_ood is None:
      return train_loader, test_loader
    return train_loader, test_loader, ood_loader

  # transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize(
  #     mean=(MNIST_MEAN,),
  #     std=(MNIST_STD,)
  #   )
  # ])
  open_K = extra.get('open_K')
  closed_data = extra.get('closed_set', True)
  print(f'Building mnist with K={open_K}')

  train_dset = FastMNIST(
    open_K=open_K,
    closed_data=closed_data,
    root=root,
    download=True,
    train=True,
    device='cpu' if ddp else DEFAULT_DEVICE
  )
  test_dset = FastMNIST(
    open_K=open_K,
    closed_data=closed_data,
    root=root,
    download=True,
    train=False,
    device='cpu' if ddp else DEFAULT_DEVICE
  )
  
  train_loader = DataLoader(
    dataset=train_dset,
    batch_size=bs,
    **ddp_args(train_dset, ddp=ddp, drop_last=drop_last, shuffle=True),
    **MNIST_LOADER_ARGS
  )

  test_loader = DataLoader(
    dataset=test_dset,
    batch_size=bs,
    **ddp_args(test_dset, ddp=ddp, drop_last=True, shuffle=False),
    **MNIST_LOADER_ARGS
  )

  if bs_ood is None:
    return train_loader, test_loader
  return train_loader, test_loader, ood_loader


def load_notmnist(root: Optional[PathLike]=DATA_DIR, bs: int=100) -> Tuple[DataLoader, DataLoader]:
  """ Handles loading the notMnist train/test (if specified)

  Args:
      root (PathLike): the directory where to save the dataset 
      bs (int): batch size of loader. Default to 100
  Returns:
      Tuple[DataLoader, DataLoader]: returns a tuple of the (train DataLoader, test DataLoader)
  """
  try:
    import deeplake
  except ImportError:
    raise ImportError('Please install deeplake to load the not-mnist dataset')

  ds = deeplake.load('hub://activeloop/not-mnist-large', read_only=True,)
  transform_train = transforms.Compose([
    transforms.RandomAffine(
      degrees=18,
      translate=(0.1, 0.1),
      scale=(0.98, 1.02),
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(
      distortion_scale=0.4,
      p=0.5
    ),
    # transforms.RandomInvert(p=0.25),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=(MNIST_MEAN,),
      std=(MNIST_STD,)
    )
  ])
  
  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=(MNIST_MEAN,),
      std=(MNIST_STD,)
    )
  ])

  train_loader = DataLoader(
    dataset=datasets.MNIST(
      root=root,
      train=True,
      download=True,
      transform=transform_train
    ),
    batch_size=bs,
    shuffle=True,
    **MNIST_LOADER_ARGS
  )

  test_loader = DataLoader(
    dataset=datasets.MNIST(
      root=root,
      train=False,
      download=False,
      transform=transform_test
    ),
    batch_size=bs,
    shuffle=False,
    **MNIST_LOADER_ARGS
  )

  return train_loader, test_loader


def generate_regression_data(n_train, n_test):
  x_train1 = torch.linspace(-6, -2, n_train//2).view(-1, 1)
  x_train2 = torch.linspace(2, 6, n_train//2).view(-1, 1)
  x_train3 = torch.linspace(-2, 2, 4).view(-1, 1)
  x_train = torch.cat((x_train1, x_train2, x_train3), dim=0)
  y_train = -(1 + x_train) * torch.sin(1.2*x_train) 
  y_train = y_train + torch.ones_like(y_train).normal_(0, 0.04)

  x_test = torch.linspace(-6, 6, n_test).view(-1, 1)
  y_test = -(1 + x_test) * torch.sin(1.2*x_test) 
  y_test = y_test + torch.ones_like(y_test).normal_(0, 0.04)
  return (x_train, y_train), (x_test, y_test)


def generate_classification_data(n_samples=400, means=[(2., 2.), (-2., 2.), (2., -2.), (-2., -2.)]):
  data = torch.zeros(n_samples, 2)
  labels = torch.zeros(n_samples)
  size = n_samples//len(means)
  for i, (x, y) in enumerate(means):
    # if i == 1:
    #   x = 8.
    #   y = 8.
    dist = torch.distributions.Normal(torch.tensor([x, y]), .4)
    samples = dist.sample([size])
    data[size*i:size*(i+1)] = samples
    labels[size*i:size*(i+1)] = torch.ones(len(samples)) * i
  
  return data, labels.long()


def ood_grid(n, ood_loc=10):
  """ samples an ood batch for a grid of points """
  # OUTLINE example to show OOD with very far away line of points
  neg_to_one = torch.linspace(-ood_loc, ood_loc, steps=n)
  one_to_neg = torch.linspace(ood_loc, -ood_loc, steps=n)
  ones = ood_loc * torch.ones(n)
  neg_ones = -1.0 * ones
  xs = torch.concat([
    neg_to_one, ones, neg_to_one, neg_ones
  ], dim=0)
  ys = torch.concat([
    ones, one_to_neg, neg_ones, neg_to_one
  ], dim=0)
  outline = torch.stack([xs, ys], dim=1)
  perms = torch.randperm(outline.shape[0])
  outline = outline[perms][:n] + torch.normal(0.0, 0.5, size=(n, 2))
  return outline


def load_toy(bs: int=100, drop_last: bool=False, ddp=False, **extra):
  """ Handles loading the a toy dataset

  Args:
      bs (int, optional): _description_. Defaults to 100.
      drop_last (bool, optional): _description_. Defaults to False.
  """
  _typ = extra['type']
  if _typ == 'fourclass':
    if bs != 100:
      print('WARNING: Note that messing with bs for training classification of toy might mess with noise ratio')
    train_data, train_targets = generate_classification_data(bs)
    test_data, test_targets = generate_classification_data(200)
    extent = 8.0
  elif _typ == 'circles':
    noise = 0.1
    def makecircles(num):
      data, target = skdatasets.make_circles(n_samples=num, noise=noise, factor=0.6, random_state=1)
      return torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.long)
    train_data, train_targets = makecircles(bs)
    test_data, test_targets = makecircles(200)
    extent = 8.0
  elif _typ == 'twomoons':
    noise = 0.1
    def makemoons(num):
      data, target = skdatasets.make_moons(n_samples=num, noise=noise)
      return torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.long)
      
    train_data, train_targets = makemoons(bs)
    test_data, test_targets = makemoons(200)
    extent = 5.0
  elif _typ == 'regressor':
    (train_data, train_targets), (test_data, test_targets) = generate_regression_data(80, 200)

  train_set = TensorDataset(train_data, train_targets)
  test_set = TensorDataset(test_data, test_targets)
  train_loader = DataLoader(
    dataset=train_set,
    batch_size=bs,
    **ddp_args(train_set, ddp=ddp, shuffle=True, drop_last=drop_last)
  )

  test_loader = DataLoader(
    dataset=test_set,
    batch_size=200,
    **ddp_args(test_set, ddp=ddp, shuffle=False, drop_last=drop_last)
  )
  
  bs_ood = extra.get('bs_ood', None)
  
  if bs_ood is None:
    return train_loader, test_loader
  
  ood_set = TensorDataset(ood_grid(bs_ood, extent))
  ood_loader = DataLoader(
    dataset=ood_set,
    batch_size=bs_ood,
    **ddp_args(ood_set, ddp=ddp, shuffle=True, drop_last=True)
  )
  
  return train_loader, test_loader, ood_loader


def load_svnh(bs: int=100, drop_last: bool=False, **extra):
  """ SVNH loader """
  val_size = 0.1
  val_seed = 1

  normalize = transforms.Normalize(
    mean=SVNH_MEAN,
    std=SVNH_STD
  )

  # define transforms
  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
  ])

  # load the dataset
  train_dataset = datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=valid_transform)

  valid_dataset = datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=valid_transform)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(val_size * num_train))

  np.random.seed(val_seed)
  np.random.shuffle(indices)

  train_idx, valid_idx = indices[split:], indices[:split]
  train_subset = Subset(train_dataset, train_idx)
  valid_subset = Subset(valid_dataset, valid_idx)

  train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=bs,
    shuffle=True,
    **SVNH_LOADER_ARGS
  )
  valid_loader = torch.utils.data.DataLoader(
    valid_subset,
    batch_size=bs,
    shuffle=False,
    **SVNH_LOADER_ARGS
  )

  return (train_loader, valid_loader)


def load_svnh_test(bs: int=100, drop_last: bool=False, **extra):
  """ SVNH dataset test """
  # define transform
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=SVNH_MEAN,
      std=SVNH_STD
    )
  ])

  dataset = datasets.SVHN(root=DATA_DIR, split="test", download=True, transform=transform,)

  data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=bs,
    shuffle=False,
    **SVNH_LOADER_ARGS
  )

  return data_loader


def load_cifar(root: Optional[PathLike]=DATA_DIR, bs: int=100, drop_last: bool=False, cifar100: bool=False, **extra) -> Tuple[DataLoader, DataLoader]:
  """ Handles loading the cifar train/test (if specified)

  Args:
      root (PathLike): the directory where to save the dataset 
      bs (int): batch size of loader. Default to 100
  Returns:
      Tuple[DataLoader, DataLoader]: returns a tuple of the (train DataLoader, test DataLoader)
  """

  root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', root)
  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), value='random')
  ])
  
  # val_transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize(
  #     mean=CIFAR_MEAN,
  #     std=CIFAR_STD
  #   )
  # ])

  open_K = extra.get('open_K')
  closed_data = extra.get('closed_set', True)
  print(f'Building cifar10 with K={open_K}')

  dset = FastCIFAR100 if cifar100 else FastCIFAR10
  train_dataset = dset(
    root=root,
    open_K=open_K,
    closed_data=closed_data,
    train=True,
    download=True,
    transform=train_transform,
    device=DEFAULT_DEVICE if extra.get('device') is None else extra.get('device')
  )
  val_dataset = dset(
    root=root,
    open_K=open_K,
    closed_data=closed_data,
    train=True,
    download=True,
    device=DEFAULT_DEVICE if extra.get('device') is None else extra.get('device')
    # transform=val_transform
  )
  
  num_train = len(train_dataset)
  indices = list(range(num_train))
  val_size = 0.1
  split = int(np.floor(val_size * num_train))

  # using same setup at ddu for replication
  val_seed = 1
  np.random.seed(val_seed)
  np.random.shuffle(indices)

  train_idx, valid_idx = indices[split:], indices[:split]

  train_subset = Subset(train_dataset, train_idx)
  valid_subset = Subset(val_dataset, valid_idx)

  train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=bs,
    shuffle=True,
    **CIFAR_LOADER_ARGS
  )
  val_loader = torch.utils.data.DataLoader(
    valid_subset,
    batch_size=bs,
    shuffle=False,
    **CIFAR_LOADER_ARGS
  )

  return train_loader, val_loader

def load_cifar_test(root: Optional[PathLike]=DATA_DIR, bs: int=100, drop_last: bool=False, cifar100: bool=False, **extra):
  """ CIFAR test loader """
  # define transform
  # transform = transforms.Compose([
  #   transforms.ToTensor(),
  #   transforms.Normalize(
  #     mean=CIFAR_MEAN,
  #     std=CIFAR_STD
  #   )
  # ])

  open_K = extra.get('open_K')
  closed_data = extra.get('closed_set', True)
  print(f'Building cifar10 with K={open_K}')

  dset = FastCIFAR100 if cifar100 else FastCIFAR10
  dataset = dset(
    root=root,
    train=False,
    download=True,
    open_K=open_K,
    closed_data=closed_data,
    device=DEFAULT_DEVICE
    # transform=transform
  )

  data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=bs,
    shuffle=False,
    **CIFAR_LOADER_ARGS
  )

  return data_loader


def load_dtd_test(root: Optional[PathLike]=DATA_DIR, size:int=64, bs: int=100, drop_last: bool=False, **extra):
  """ DTD test loader """
  # define transform
  transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=DTD_MEAN,
      std=DTD_STD
    )
  ])

  dataset = datasets.DTD(
    root=os.path.join(root, 'dtd'),
    split='test',
    transform=transform,
    download=True
  )

  data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=bs,
    shuffle=False,
    **TINYIMG_LOADER
  )

  return data_loader


def load_tinyimagenet(root: Optional[PathLike]=TINYIMG_DATA_DIR, bs: int=100, drop_last: bool=False, ddp: bool=False, **extra) -> Tuple[DataLoader, DataLoader]:
  """ Handles loading the tinyimagenet train/test (if specified)

  Args:
      root (PathLike): the directory where to save the dataset 
      bs (int): batch size of loader. Default to 100
  Returns:
      Tuple[DataLoader, DataLoader]: returns a tuple of the (train DataLoader, test DataLoader)
  """
  raise NotImplementedError

  root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', root)
  train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=TINYIMG_MEAN, std=TINYIMG_STD),
  ])
  
  val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=TINYIMG_MEAN, std=TINYIMG_STD
    )
  ])

  # train_dataset = TinyImageNetDataset(
  #   root_dir=root,
  #   mode='train',
  #   preload=True,
  #   transform=train_transform,
  #   download=False
  # )
  # val_dataset = TinyImageNetDataset(
  #   root_dir=root,
  #   mode='val',
  #   preload=True,
  #   transform=val_transform,
  #   download=False
  # )
  
  from MLclf import MLclf
  transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(64, padding=4),
    transforms.RandomAffine(
      degrees=30,
      translate=(0.1, 0.1),
      shear=10
    ),
    transforms.RandomResizedCrop(64, scale=(0.5, 1), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=TINYIMG_MEAN, std=TINYIMG_STD),
  ])
  train_dataset, val, val_dataset = MLclf.tinyimagenet_clf_dataset(data_dir=root + '/', ratio_train=0.64, ratio_val=0.16,
                                                      seed_value=0, shuffle=False,
                                                      transform=transform,
                                                      save_clf_data=True,
                                                      few_shot=False)
  
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=bs,
    **ddp_args(train_dataset, ddp=ddp, shuffle=True),
    **TINYIMG_LOADER
  )
  val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=bs,
    **ddp_args(val_dataset, ddp=ddp, shuffle=False),
    **TINYIMG_LOADER
  )

  return train_loader, val_loader


def load_tinyimagenet_test(root: Optional[PathLike]=TINYIMG_DATA_DIR, bs: int=100, drop_last: bool=False, ddp: bool=False, **extra):
  """ CIFAR test loader """
  raise NotImplementedError

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
      mean=TINYIMG_MEAN, std=TINYIMG_STD
    )
  ])
  
  test_dataset = TinyImageNetDataset(
    root_dir=root,
    mode='val',
    preload=True,
    transform=test_transform,
    download=False
  )

  test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=bs,
    **ddp_args(test_dataset, ddp=ddp, drop_last=drop_last, shuffle=False),
    **TINYIMG_LOADER
  )

  return test_loader


DATASET_MAP = {
  'mnist': load_mnist,
  'cifar': load_cifar,
  'tinyimagenet': load_tinyimagenet,
  'toy': load_toy
}

def load_dataset(name: str, bs: int=100, drop_last: bool=False, **kwargs):
  """ Basic function to load experiment datasets

  Args:
      name (str): name of dataset
      bs (int, optional): Batch size. Defaults to 100.
      drop_last (bool, optional): Drop last batch (if you need bs to be consistent). Defaults to False.

  Raises:
      KeyError: if the dataset was not found in map

  Returns:
      tuple: train loader (DataLoader), test loader (DataLoader)
  """
  if name not in DATASET_MAP:
    raise KeyError(f'The dataset {name} was not found. Please use one of the following {str(list(DATASET_MAP.keys()))}')
  return DATASET_MAP[name](bs=bs, drop_last=drop_last, **kwargs)


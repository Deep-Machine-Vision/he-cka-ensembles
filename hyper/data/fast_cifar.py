"""
Fast open set CIFAR10 dataset by smerkoud@oregonstate.edu

Inspired from FastMNISt impl
"""
import torch
import os
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from hyper.data.fast_mnist import DATA_DIR

default_device = torch.device("cuda" if torch.cuda.is_available else "cpu")


class FastCIFAR10(CIFAR10):
  def __init__(self, open_K: int=None, closed_data: bool=True, device='cpu', *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1]
    self.data = torch.from_numpy(self.data).float().div(255).to(device)

    # Normalize it with the usual CIFAR mean and std
    self._mean = torch.tensor((0.4914, 0.4822, 0.4465))
    self._std = torch.tensor((0.2023, 0.1994, 0.2010))
    self.data = self.data.sub_(self._mean).div_(self._std)

    # move to channels first
    self.data = self.data.permute(0, 3, 1, 2).to(device)

    # Put both data and targets on GPU in advance
    # self.targets = self.targets.to(device)
    self.targets = torch.tensor(self.targets)

    # sort data into open and closed sets
    self.all_classes = torch.arange(0, 10)# .to(device)
    
    self.run_open = open_K is not None and int(open_K) > 0
    if self.run_open:
      self.K = int(open_K)
      self.closed_classes = self.all_classes[:self.K]
      self.open_classes = self.all_classes[self.K:]
      
      # capture data into both groups
      self.closed_ind = (self.targets < self.K).nonzero(as_tuple=True)[0]
      self.open_ind = (self.targets >= self.K).nonzero(as_tuple=True)[0]
      self.closed_data = self.data[self.closed_ind]
      self.closed_target = self.targets[self.closed_ind]
      self.open_data = self.data[self.open_ind]
      self.open_target = self.targets[self.open_ind]
      
      # assert dims
      assert (self.open_data.shape[0] + self.closed_data.shape[0]) == self.data.shape[0], 'Open dataset index mismatch'
      
      # replace data with either the closed/open dataset
      if closed_data:
        self.data = self.closed_data
        self.targets = self.closed_target
      else:
        self.data = self.open_data
        self.targets = self.open_target
      
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target



class FastCIFAR100(CIFAR100):
  def __init__(self, open_K: int=None, closed_data: bool=True, device='cpu', *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1]
    self.data = torch.from_numpy(self.data).float().div(255)

    # Normalize it with the usual CIFAR mean and std
    self._mean = torch.tensor((0.5071, 0.4865, 0.4409))
    self._std = torch.tensor((0.2673, 0.2564, 0.2762))
    self.data = self.data.sub_(self._mean).div_(self._std)

    # move to channels first
    self.data = self.data.permute(0, 3, 1, 2).to(device)

    # Put both data and targets on GPU in advance
    # self.targets = self.targets.to(device)
    self.targets = torch.tensor(self.targets)

    # sort data into open and closed sets
    self.all_classes = torch.arange(0, 100)# .to(device)
    
    self.run_open = open_K is not None and int(open_K) > 0
    if self.run_open:
      self.K = int(open_K)
      self.closed_classes = self.all_classes[:self.K]
      self.open_classes = self.all_classes[self.K:]
      
      # capture data into both groups
      self.closed_ind = (self.targets < self.K).nonzero(as_tuple=True)[0]
      self.open_ind = (self.targets >= self.K).nonzero(as_tuple=True)[0]
      self.closed_data = self.data[self.closed_ind]
      self.closed_target = self.targets[self.closed_ind]
      self.open_data = self.data[self.open_ind]
      self.open_target = self.targets[self.open_ind]
      
      # assert dims
      assert (self.open_data.shape[0] + self.closed_data.shape[0]) == self.data.shape[0], 'Open dataset index mismatch'
      
      # replace data with either the closed/open dataset
      if closed_data:
        self.data = self.closed_data
        self.targets = self.closed_target
      else:
        self.data = self.open_data
        self.targets = self.open_target
      
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.targets[index]

    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target


def create_CIFAR100_dataset():
  train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4)
  ])
  train_dataset = FastCIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
  test_dataset = FastCIFAR100(root=DATA_DIR, train=False, download=True)

  return train_dataset, test_dataset


def get_test100_loader(batch_size, **kwargs):
  # load the dataset
  _, cifar_test_dataset = create_CIFAR100_dataset()
  test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  return test_loader


def create_CIFAR10_dataset():
  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), value='random')
  ])
  train_dataset = FastCIFAR10(root="data", train=True, download=True, transform=train_transform)
  test_dataset = FastCIFAR10(root="data", train=False, download=True)

  return train_dataset, test_dataset


def get_test_loader(batch_size, **kwargs):
  # load the dataset
  _, cifar_test_dataset = create_CIFAR10_dataset()
  test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  return test_loader


def get_open_loaders(batch_size, open_K, train=False, num_workers=0, pin_memory=True, **kwargs):
  # define transforms
  dataset = FastCIFAR10(
    root=DATA_DIR,
    train=train,
    download=True,
    closed_data=False,
    open_K=open_K
  )

  loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
  )
  
  id_dataset = FastCIFAR10(
    root=DATA_DIR,
    train=train,
    download=True,
    closed_data=True,
    open_K=open_K
  )

  id_loader = torch.utils.data.DataLoader(
    id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
  )

  return id_loader, loader

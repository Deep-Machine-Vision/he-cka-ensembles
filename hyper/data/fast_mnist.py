"""
Taken from https://github.com/omegafragger/DDU

and modified for open datasets by smerkoud@oregonstate.edu

FastMNIST taken from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
"""
import torch
import os
from torchvision.datasets import MNIST

from hyper.data.util import ddp_args


default_device = torch.device("cuda" if torch.cuda.is_available else "cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


class FastMNIST(MNIST):
  def __init__(self, open_K: int=None, closed_data: bool=True, device='cpu', *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)

    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)

    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)
    
    # sort data into open and closed sets   
    self.all_classes = torch.arange(0, 10, 1).to(device)
    
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

    # only works with tensors not PIL now
    if self.transform is not None:
        img = self.transform(img)

    if self.target_transform is not None:
        target = self.target_transform(target)


    return img, target


def create_MNIST_dataset():
  train_dataset = FastMNIST(root="data", train=True, download=True, device=default_device)
  test_dataset = FastMNIST(root="data", train=False, download=True, device=default_device)

  return train_dataset, test_dataset


def get_test_loader(batch_size, ddp=False, **kwargs):
  # load the dataset
  _, mnist_test_dataset = create_MNIST_dataset()
  test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=batch_size, num_workers=0, **ddp_args(mnist_test_dataset, ddp=ddp, shuffle=False))
  return test_loader


# def get_open_loaders(batch_size, open_K, train=False, num_workers=0, pin_memory=True, **kwargs):
#   # define transforms
#   dataset = FastMNIST(
#     root=ROOT,
#     train=train,
#     download=True,
#     closed_data=False,
#     open_K=open_K
#   )

#   loader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
#   )
  
#   id_dataset = FastMNIST(
#     root=ROOT,
#     train=train,
#     download=True,
#     closed_data=True,
#     open_K=open_K
#   )

#   id_loader = torch.utils.data.DataLoader(
#     id_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
#   )

#   return id_loader, loader
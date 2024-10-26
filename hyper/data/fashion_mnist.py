"""
Taken from https://github.com/omegafragger/DDU

Fashion-MNIST used as an OOD dataset.
"""

import torch

from torchvision import datasets
from torchvision import transforms
from hyper.data.fast_mnist import DATA_DIR
import os


def get_loaders(batch_size, train=False, num_workers=4, pin_memory=True, **kwargs):
  # define transforms
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,),)])
  dataset = datasets.FashionMNIST(root=DATA_DIR, train=train, download=True, transform=transform,)

  loader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
  )

  return loader
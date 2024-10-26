""" Simple OOD Dataset constructor for images that doesn't require semantically meaningful output
but is still useful for forcing non-linear relationships between models

David Smerkous
"""
from typing import Callable, Optional, Union, Any
from torch.utils.data import Dataset
from torchvision import transforms as TF

from hyper.data.fast_mnist import DATA_DIR
from hyper.data.imagenet import ADImageNet
from hyper.data.perlin import perlin_noise as gen_perlin_noise
from hyper.data.simplex import Simplex
from hyper.data.util import ddp_args

import numpy as np
import multiprocessing
import torch
import random
import math
import tqdm
import os


MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 8))
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
SVNH_MEAN = (0.4914, 0.4822, 0.4465)
SVNH_STD = (0.2023, 0.1994, 0.2010)


simplex = Simplex()
ID2OOD_DEFAULT_TRANSFORMS = TF.RandomOrder([
  TF.RandomInvert(p=0.1),
  TF.RandomErasing(p=0.1),
  TF.RandomAffine(
    degrees=30,
    translate=(0.1, 0.1),
    scale=(0.8, 1.2),
  ),
  TF.RandomPerspective(
    distortion_scale=0.6
  ),
  TF.ElasticTransform(
    alpha=180.0,
    sigma=6.0,
  ),
  # TF.ColorJitter(
  #   contrast=0.3
  # ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=1.0,
    p=0.5
  ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=-1.0,
    p=0.5
  ),
  TF.ElasticTransform(
    alpha=180.0,
    sigma=6.0,
  ),
  TF.RandomHorizontalFlip(p=0.5),
  TF.RandomVerticalFlip(p=0.5),
  TF.RandomApply(torch.nn.ModuleList([
      TF.GaussianBlur(
        kernel_size=(5, 5),
        sigma=(0.1, 4.)
      ),
    ]), p=0.6
  )
])


# we made the cifar transforms a bit less aggressive
# to make boundary between ID and OOD less clear
ID2OOD_CIFAR_TRANSFORMS = TF.RandomOrder([
  TF.RandomInvert(),
  TF.RandomErasing(p=0.1),
  TF.ElasticTransform(
    alpha=80.0,
    sigma=5.0,
  ),
  # TF.ColorJitter(
  #   brightness=0.005,
  #   contrast=0.005,
  #   saturation=0.0015
  # ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=1.0,
    p=0.25
  ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=0.0,
    p=0.25
  ),
  TF.ElasticTransform(
    alpha=80.0,
    sigma=5.0,
  ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=-1.0,
    p=0.25
  ),
  TF.ElasticTransform(
    alpha=80.0,
    sigma=5.0
  ),
  TF.RandomApply(torch.nn.ModuleList([
    TF.GaussianBlur(
      kernel_size=(3, 3),
      sigma=(0.1, 3.)
    ),
  ]), p=0.3
  )
])


# we made the imagenet transforms a bit less aggressive
# to make boundary between ID and OOD less clear
ID2OOD_IMAGENET_TRANSFORMS = TF.RandomOrder([
  TF.RandomInvert(),
  TF.RandomErasing(p=0.1),
  TF.ElasticTransform(
    alpha=90.0,
    sigma=8.0,
  ),
  # TF.RandomChannelPermutation(),
  # TF.ColorJitter(
  #   brightness=0.005,
  #   contrast=0.005,
  #   saturation=0.0015
  # ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=1.0,
    p=0.2
  ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=0.0,
    p=0.25
  ),
  TF.ElasticTransform(
    alpha=80.0,
    sigma=5.0,
  ),
  TF.RandomErasing(
    scale=(0.1, 0.5),
    value=-1.0,
    p=0.25
  ),
  TF.RandomApply(torch.nn.ModuleList([
    TF.ElasticTransform(
      alpha=80.0,
      sigma=5.0
    ),
  ]), p=0.6
  ),
  TF.RandomApply(torch.nn.ModuleList([
    TF.ElasticTransform(
      alpha=80.0,
      sigma=5.0
    ),
  ]), p=0.6
  ),
  TF.RandomApply(torch.nn.ModuleList([
    TF.GaussianBlur(
      kernel_size=(3, 3),
      sigma=(0.1, 3.)
    ),
  ]), p=0.3
  ),
  TF.RandomAffine(
    degrees=10,
    translate=None,
    scale=(1.0, 3.0),
    shear=50
  )
])

""" Generation of data """
def prop(im, prop: float):
  """ Generate a proportion in px the smallest shape in image """
  if isinstance(prop, int):
    return prop
  return math.ceil(min(im.shape[:2]) * prop)


@ torch.no_grad()
def norm_channel_augment(img, z: float=1.0, aug_factor: float=0.7):
  """ Normalize values within image, or channel, and augment by a factor @TODO channel factors """
  aug_factor = 1.0 + np.random.normal(0.0, aug_factor)
  img[:] /= max(img.std().item(), 0.01)
  img[:] *= z * aug_factor


@torch.no_grad()
def horz_lines(im, z: float=1.0, max_width: int=0.1, max_space: float=0.2):
  """ Generate horizontal lines """
  h, _ = im.shape[:2]
  pw = prop(im, max_width)
  ps = prop(im, max_space)
  
  cur_line = 0
  while cur_line < h:
    width = np.random.randint(1, pw + 1)
    space = np.random.randint(1, ps + 1)
    
    im[(space + cur_line):(space + cur_line + width), :] = z
    cur_line += space + width
  norm_channel_augment(im, z)

@torch.no_grad()
def vert_lines(im, z: float=1.0, max_width: int=0.1, max_space: float=0.2):
  """ Generate vertical lines """
  _, w = im.shape[:2]
  pw = prop(im, max_width)
  ps = prop(im, max_space)
  
  cur_line = 0
  while cur_line < w:
    width = np.random.randint(1, pw + 1)
    space = np.random.randint(1, ps + 1)
    
    im[:, (space + cur_line):(space + cur_line + width)] = z
    cur_line += space + width
  norm_channel_augment(im, z)

@torch.no_grad()
def alternating_grid(grid, z: float=1.0):
  """ Alternating values between -z and z """
  grid[:] = -z
  grid[1::2, ::2] = z # Set every odd row, even column to 1
  grid[::2, 1::2] = z  # Set every even row, odd column to 1
  norm_channel_augment(grid, z)

@torch.no_grad()
def alternating_grid_random_sizes(grid, z: float=1.0, min_block_size: int=1, max_block_size: float=0.4):
  """ Alternating blocks of pixels with random "rows" of various block sizes """
  size = min(grid.shape[0], grid.shape[1])
  min_block_size = prop(grid, min_block_size)
  max_block_size = prop(grid, max_block_size)
  
  for i in range(0, size, max_block_size):
    block_size = np.random.randint(min_block_size, max_block_size + 1)

    for j in range(0, size, 2 * block_size):
      grid[i:i+block_size, j:j+block_size] = z  # Set block of pixels to 1
      grid[i+block_size:i+2*block_size, j+block_size:j+2*block_size] = z  # Set next block of pixels to 1
  norm_channel_augment(grid, z)
  return grid


@torch.no_grad()
def negate(img):
  """ Negates values within image """
  img[:] *= -1.0


@torch.no_grad()
def fill_image(img, z: float=1.0):
  """ Fills image with z """
  img[:] = z


@torch.no_grad()
def gaussian_noise(img, mean: float=0.0, z: float=1.0):
  """ Adds gaussian noise to image """
  img[:] += torch.normal(mean, z, size=img.shape, device=img.device)


@torch.no_grad()
def perlin_noise(img, z: float=1.0):
  """ Generate perlin noise for specified channels """
  # specify by channels
  if len(img.shape) == 2:
    channels = 1
    shape = img.shape
  elif len(img.shape) == 3:
    channels = img.shape[0]
    shape = img.shape[1:]
  else:
    raise ValueError(f"Image shape {img.shape} not supported. See image_ood.py for details")
  
  # create perlin noise
  noise = gen_perlin_noise(
    grid_shape=(4, 4),
    out_shape=shape,
    batch_size=channels
  )

  # normalize to have z std
  aug_factor = 1.0 + np.random.normal(0.0, 0.6)
  noise /= max(noise.std().item(), 0.01)
  noise *= z * aug_factor
  img[:] = noise.to(img.device)


@torch.no_grad()
def simplex_noise(img, z: float=1.0):
  """ Generate simplex noise for specified channels """
  global simplex
  
  # specify by channels
  if len(img.shape) == 2:
    channels = 1
    shape = img.shape
  elif len(img.shape) == 3:
    channels = img.shape[0]
    shape = img.shape[1:]
  else:
    raise ValueError(f"Image shape {img.shape} not supported. See image_ood.py for details")
  
  
  # create simplex noise
  channel_data = []
  for i in range(channels):
    channel_data.append(torch.from_numpy(simplex.rand_2d_octaves(shape, 6, 0.6)).to(img.device).view(1, *shape))
  noise = torch.cat(channel_data, dim=0)

  # normalize to have z std
  aug_factor = 1.0 + np.random.normal(0.0, 0.6)
  noise /= max(noise.std().item(), 0.01)
  noise *= z * aug_factor
  img[:] = noise


def threshold_in_random_boxes(image, z: float=1.0, max_num_boxes: int = 4, max_box_size: float = 0.3, threshold_prob: float = 0.0):
  """ Thresholds a random number of boxes of random size in the image. """
  h, w = image.shape[:2]
  num_boxes = np.random.randint(1, max_num_boxes + 1)
  max_box_size = prop(image, max_box_size)
  z_max = image.abs().max()

  for _ in range(num_boxes):
    box_size = np.random.randint(1, max_box_size + 1)
    x_min = np.random.randint(0, w - box_size)
    y_min = np.random.randint(0, h - box_size)
    image[y_min:y_min + box_size, x_min:x_min + box_size] = torch.where(image[y_min:y_min + box_size, x_min:x_min + box_size] >= threshold_prob, -z * z_max, z * z_max)


# function that does nothing
def null_apply(*args, **kwargs):
  pass

# composition function
def compose_apply(*functions):
  def apply(x):
    for f in functions:
      f(x)
    return x
  return apply


""" Definition of dataset """


class ImageOODDataset(Dataset):
  def __init__(self, num_images: int, image_size: tuple, image_channels: int, id_dataset = None, id_transforms = ID2OOD_DEFAULT_TRANSFORMS, id_prob: float=0.5, device='cpu',  ood_invert_p=0.5, min_zoom=0.8) -> None:
    super().__init__()
    self.num = int(num_images)
    self.channels = image_channels
    self.image_size = image_size
    self.id_dataset = id_dataset
    self.id_transforms = id_transforms
    self.id_prob = id_prob
    self.device = device
    self.pregen = False
    self.cache_load_all = False
    
    if id_dataset is not None:
      self.id_size = len(self.id_dataset)
    else:
      self.id_size = 0
    
    # options to apply to base image of each channel
    self.channel_options = [
      horz_lines,
      vert_lines,
      alternating_grid,
      alternating_grid_random_sizes,
      compose_apply(horz_lines, vert_lines),
      perlin_noise,
      simplex_noise,
      null_apply,
      fill_image
    ]
    
    # options to select generating image options
    self.gen_image_options = [
      self.full_channel_random,
      self.same_channel_random,
      self.full_perlin_image,
      self.full_simplex_image
    ]
    self.gen_weights = [
      0.35,
      0.35,
      0.15,
      0.15
    ]
    
    # apply after generating channels
    self._ood_transforms = [
      TF.RandomInvert(p=ood_invert_p),
      TF.RandomApply(torch.nn.ModuleList([
        TF.GaussianBlur(
          kernel_size=(3, 3),
          sigma=(0.05, 2.0)
        )
      ]), p=0.3)
    ]
    
    # applies ood transform with random fill for background
    self.ood_transforms = lambda x: (TF.RandomOrder([
      TF.RandomApply(
        torch.nn.ModuleList([
          TF.RandomAffine(
            degrees=180,
            translate=(0.1, 0.1),
            scale=(min_zoom, 1.5),
            fill=np.random.uniform(-2.0, 2.0)
          )
      ]), p=0.75)] + self._ood_transforms)(x))

  def __len__(self) -> int:
    return self.num
  
  def base_image(self, chan_num=None):
    """ Generates a base image for the dataset """
    if chan_num is None:
      chan_num = self.channels
    return torch.zeros((chan_num, *self.image_size), dtype=torch.float32, device=self.device)
  
  def channel_random(self, option=None):
    """ Generates a random patterned channel """
    # pick a channel option
    base = self.base_image(chan_num=1).squeeze(0)
    
    # apply to specified channel
    if option is None:
      random.choice(self.channel_options)(base)
      
      # also apply random box threshold with prob 0.25
      if np.random.random() < 0.25:
        threshold_in_random_boxes(base)
    else:
      option(base)
    
    # now do affine/invert
    channel = self.ood_transforms(base.unsqueeze(0))
    return channel
  
  def full_channel_random(self):
    """ Generates an image with each channel random pattern """
    if self.channels == 1:
      return self.channel_random()
    else:
      channels = []
      for _ in range(self.channels):
        channels.append(self.channel_random())
      return torch.cat(channels, dim=0)
  
  def same_channel_random(self):
    """ Generates an image with each channel the same random pattern. With probability 0.5 exact same pattern across channels """    
    if np.random.random() < 0.5:
      # same pattern across channels
      channel = self.channel_random()
      channels = [channel for _ in range(self.channels)]
      return torch.cat(channels, dim=0)
    else:
      # same choice of pattern but different inits across channels
      option = random.choice(self.channel_options)
      channels = []
      for _ in range(self.channels):
        channels.append(self.channel_random(option=option))
      return torch.cat(channels, dim=0)
  
  def full_perlin_image(self):
    """ Generate all channels with perlin noise """
    base = self.base_image(chan_num=self.channels)
    perlin_noise(base)
    return self.ood_transforms(base)
  
  def full_simplex_image(self):
    """ Generate all channels with simplex noise """
    base = self.base_image(chan_num=self.channels)
    simplex_noise(base)
    return self.ood_transforms(base)
  
  def preload_all(self, root: str='ood', device=None):
    """ If pregenerated OOD dataset created and cache files exist it will attempt to load all of them into memory.
    
    Only use this for small datasets like MNIST/CIFAR to save ~10x time per epoch (on our A100 tests)
    """
    if self.pregen:
      parts = self.num_parts
      concats = []
      for i in range(parts):
        p_path = os.path.join(root, f'part_{i}.pt')
        if not os.path.isfile(p_path):
          raise RuntimeError(f'Part file {i} not found')
        data = torch.load(p_path)
        if isinstance(data, list):
          data = torch.concat(data)
        concats.append(data)
      
      # join batch dimension
      self.cache_load_all = True
      self.cache_all = torch.concat(concats)
      
      # move to device
      if device is not None:
        self.cache_all = self.cache_all.to(device=device)
    else:
      raise RuntimeError('Dataset not pregenerated. Cannot call preload all, please call pregenerate() first')
  
  def pregenerate(self, root: str='ood', regenerate: bool=False, parts: int=10, ddp=False):
    """ Pregenerate the OOD set for more efficient compute later. Split into parts worth of pt files """
    if not os.path.isdir(root):
      os.mkdir(root)
    
    regen = False
    self.part_files = {}
    for i in range(parts):
      p_path = os.path.join(root, f'part_{i}.pt')
      if not os.path.isfile(p_path):
        print('Creating...', p_path)
        regen = True
        
        if ddp:
          raise RuntimeError('Please consider generating pregenerated OOD dataset using a single device! Do not use DDP')
      self.part_files[i] = p_path
    self.num_parts = parts
    self.part_size = int(self.num / parts)
    
    # generate the data in the part files
    # print('REGEN', regen, regenerate)
    if regen or regenerate:
      print('Caching/pregenerating OOD dataset')
      # use self as dataloader
      ood_loader = torch.utils.data.DataLoader(self, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
      
      # index tracking
      index = 0
      part_index = 0
      part_data = []
      
      # load each generated image and save into respective part file
      for data in tqdm.tqdm(ood_loader, total=len(ood_loader), desc='Pregenerating OOD images'):    
        cur_part_index = int(index / self.part_size)
        if cur_part_index > part_index and cur_part_index < parts:
          # save the data
          torch.save(part_data, self.part_files[part_index])
          part_index += 1
          part_data = []
        
        # save the data
        part_data.append(data)
        index += 1
      
      # save final parts file (might have more or less than part_size)
      torch.save(part_data, self.part_files[len(self.part_files) - 1])
      
    # set flag we're using pregenerated dataset
    self.pregen = True
    self._cache_part = None
    self._cache_data = None
    
  def __getitem__(self, idx: int) -> Any:
    """ Generates a random image/indexing does not matter """
    # if already pregenerated then take from specified part file
    load = None
    if self.pregen:
      if self.cache_load_all:
        return self.cache_all[idx]  # everything already cached/loaded
      else:  # have to load individual part files
        load_part = False
        if self._cache_part is None or self._cache_data is None:
          load_part = True
        elif self._cache_part != int(idx / self.part_size):
          load_part = True
        
        # cached part is not loaded
        if load_part:
          self._cache_part = int(idx / self.part_size)
          self._cache_data = torch.load(self.part_files[self._cache_part])
        
        # pull from cache part
        load = self._cache_data[idx % self.part_size].squeeze(0)
    
    # do in distribution sample to ood movement with probability id_prob
    if load is None and self.id_size > 0:
      if np.random.random() < self.id_prob:
        # id dataset
        id_example = self.id_dataset[idx % self.id_size][0]

        # apply id transforms
        if self.id_transforms is not None:
          id_example = self.id_transforms(id_example)
      
        # apply transforms for id to appear ood  
        load = id_example
    
    with torch.no_grad():
      # otherwise use generated "border" patterns
      if load is None:
        load = random.choices(self.gen_image_options, weights=self.gen_weights)[0]()
      
      # apply transforms
      load = self.ood_transforms(load)
      
      # apply gaussian noise
      if np.random.random() < 0.25:
        std = np.random.uniform(0.01, 0.4)
        load += torch.normal(0.0, std, size=load.shape, device=load.device)
    
    return load


MNIST_OOD_DIR = os.path.join(DATA_DIR, 'mnist_ood')
def get_mnist_ood_loader(batch_size, root_dir=MNIST_OOD_DIR, ood_size=200_000, id_prob=0.35, pregenerate: bool=True, preload: bool=True, ddp=False, **kwargs):
  from torchvision.datasets import MNIST
  
  # load the ID dataset
  mnist = MNIST(DATA_DIR, train=True, download=True)
  device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # create OOD sampler
  mnist_ood_ds = ImageOODDataset(
    num_images=ood_size,
    image_size=(28, 28),
    image_channels=1,
    id_dataset=mnist,
    id_transforms=TF.Compose([
      TF.ToTensor(),
      TF.Normalize(
        mean=MNIST_MEAN,
        std=MNIST_STD
      ),
      ID2OOD_DEFAULT_TRANSFORMS
    ]),
    id_prob=id_prob,
    device=device,
    ood_invert_p=0.05
  )
  
  if pregenerate:
    mnist_ood_ds.pregenerate(
      root_dir,
      regenerate=False,
      parts=10
    )
    
    if preload:
      mnist_ood_ds.preload_all(root_dir)
  
  # make the ood loader
  # @IMPORTANT shuffle=False so that we don't continually reload part files
  # it's already shuffled/randomly generated beforehand
  ood_loader = torch.utils.data.DataLoader(
    mnist_ood_ds,
    batch_size=batch_size,
    # shuffle=preload,  # SHOULD NOT SHUFFLE if preload = False!
    num_workers=min(MAX_WORKERS, multiprocessing.cpu_count()),
    # pin_memory=True,
    **ddp_args(mnist_ood_ds, ddp=ddp, drop_last=True, shuffle=preload)
  )
  return ood_loader


CIFAR_OOD_DIR = os.path.join(DATA_DIR, 'cifar_ood')
def get_cifar_ood_loader(batch_size, root_dir=CIFAR_OOD_DIR, ood_size=200_000, id_prob=0.4, pregenerate: bool=True, preload: bool=True, **kwargs):
  from torchvision.datasets import CIFAR10
  
  # load the ID dataset
  cifar = CIFAR10(DATA_DIR, train=True, download=True)
  device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # create OOD sampler
  cifar_ood_ds = ImageOODDataset(
    num_images=ood_size,
    image_size=(32, 32),
    image_channels=3,
    id_dataset=cifar,
    id_transforms=TF.Compose([
      TF.ToTensor(),
      TF.RandomCrop(32, padding=4),
      TF.Normalize(
        mean=CIFAR_MEAN,
        std=CIFAR_STD
      ),
      ID2OOD_CIFAR_TRANSFORMS
    ]),
    id_prob=id_prob,
    device=device
  )
  
  # pregenerate the OOD dataset and save to disk
  # since generating the dataset in realtime is slow
  if pregenerate:
    cifar_ood_ds.pregenerate(
      root_dir,
      regenerate=False,
      parts=10
    )
    
    if preload:
      cifar_ood_ds.preload_all(root_dir)
  
  # make the ood loader
  # @IMPORTANT shuffle=False so that we don't continually reload part files
  # it's already shuffled/randomly generated beforehand
  ood_loader = torch.utils.data.DataLoader(
    cifar_ood_ds,
    batch_size=batch_size,
    shuffle=preload,  # SHOULD NOT SHUFFLE if preload = False!
    num_workers=min(MAX_WORKERS, multiprocessing.cpu_count()),
    drop_last=True,
    pin_memory=True
  )
  return ood_loader


TINYIMAGENET_OOD = os.path.join(DATA_DIR, 'tinyimagenet_ood')
def get_tinyimagenet_ood_loader(batch_size, root_dir=TINYIMAGENET_OOD, ood_size=200_000, id_prob=0.4, pregenerate: bool=True, preload: bool=True, **kwargs):
  # from torchvision.datasets import CIFAR10
  from MLclf import MLclf
  
  # load the ID dataset
  trainset, val, testset = MLclf.tinyimagenet_clf_dataset(data_dir=DATA_DIR + '/data_tinyimagenet/tiny-imagenet-200/', ratio_train=0.95, ratio_val=0.0, seed_value=None, shuffle=True,
                                                      transform=TF.Compose([TF.ToTensor()]),
                                                      save_clf_data=True,
                                                      few_shot=False)
  device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # create OOD sampler
  cifar_ood_ds = ImageOODDataset(
    num_images=ood_size,
    image_size=(64, 64),
    image_channels=3,
    id_dataset=trainset,
    id_transforms=TF.Compose([
      # TF.ToTensor(),
      TF.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
      ),
      ID2OOD_IMAGENET_TRANSFORMS
    ]),
    id_prob=id_prob,
    device=device
  )
  
  # pregenerate the OOD dataset and save to disk
  # since generating the dataset in realtime is slow
  if pregenerate:
    cifar_ood_ds.pregenerate(
      root_dir,
      regenerate=False,
      parts=10
    )
    
    if preload:
      cifar_ood_ds.preload_all(root_dir)
  
  # make the ood loader
  # @IMPORTANT shuffle=False so that we don't continually reload part files
  # it's already shuffled/randomly generated beforehand
  ood_loader = torch.utils.data.DataLoader(
    cifar_ood_ds,
    batch_size=batch_size,
    shuffle=preload,  # SHOULD NOT SHUFFLE if preload = False!
    num_workers=min(MAX_WORKERS, multiprocessing.cpu_count()),
    drop_last=True,
    pin_memory=True
  )
  return ood_loader


def get_imagenet_ood_loader(batch_size, root_dir='imagenet_ood/', ood_size=200_000, id_prob=0.4, pregenerate: bool=True, preload: bool=True, **kwargs):
  # from torchvision.datasets import CIFAR10
  from MLclf import MLclf
  
  # load the ID dataset
  # trainset, val, testset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.95, ratio_val=0.0, seed_value=None, shuffle=True,
  #                                                     transform=TF.Compose([TF.ToTensor()]),
  #                                                     save_clf_data=True,
  #                                                     few_shot=False)
  # trainset = ImageNet22K(
  #   root='/nfs/hpc/dgx2-2/imagenet/train'
  # )
  transform = TF.RandomResizedCrop(224)
  dsets = ADImageNet(
    root='/nfs/stak/users/smerkoud/hpc-share/imagenet30',  # '/nfs/hpc/dgx2-2/imagenet',
    train_transform=transform,
    test_transform=transform,
    normal_classes=list(range(30)),
    raw_shape=(3, 224, 224),
    nominal_label=0
  )
  
  trainset = dsets.train_set
  device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # create OOD sampler
  cifar_ood_ds = ImageOODDataset(
    num_images=ood_size,
    image_size=(224, 224),
    image_channels=3,
    id_dataset=trainset,
    id_transforms=TF.Compose([
      TF.RandomHorizontalFlip(),
      TF.ToTensor(),
      TF.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
      ),
      ID2OOD_IMAGENET_TRANSFORMS
    ]),
    id_prob=id_prob,
    device=device
  )
  
  # pregenerate the OOD dataset and save to disk
  # since generating the dataset in realtime is slow
  if pregenerate:
    cifar_ood_ds.pregenerate(
      root_dir,
      regenerate=False,
      parts=10
    )
    
    if preload:
      cifar_ood_ds.preload_all(root_dir)
  
  # make the ood loader
  # @IMPORTANT shuffle=False so that we don't continually reload part files
  # it's already shuffled/randomly generated beforehand
  ood_loader = torch.utils.data.DataLoader(
    cifar_ood_ds,
    batch_size=batch_size,
    shuffle=preload,  # SHOULD NOT SHUFFLE if preload = False!
    num_workers=min(MAX_WORKERS, multiprocessing.cpu_count()),
    drop_last=True,
    pin_memory=True
  )
  return ood_loader


# test dataset
if __name__ == '__main__':
  from hyper.data.fast_mnist import FastMNIST
  import matplotlib.pyplot as plt
  from mpl_toolkits.axes_grid1 import ImageGrid
  import time
  MAX_WORKERS = 0
  mnist = False
  tiny = True
  img = False
  
  if img:
    ood_loader = get_imagenet_ood_loader(1, id_prob=0.7, pregenerate=False)
  elif tiny:
    ood_loader = get_tinyimagenet_ood_loader(1, id_prob=0.7, pregenerate=False)
  else:
    if mnist:
      ood_loader = get_mnist_ood_loader(1, id_prob=0.4, pregenerate=False)
    else:
      ood_loader = get_cifar_ood_loader(1, pregenerate=False)
  
  import cv2
  with torch.no_grad():
    ood_iter = iter(ood_loader)
    # for i in range(50):
    #   next(ood_iter)
    # im_list = []
    
    ROWS = 7
    COLS = 6
    fig = plt.figure(figsize=(8., 10.))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(ROWS, COLS),
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    for i in range(ROWS * COLS):
      img = next(ood_iter)
      
      if mnist:
        img = torch.clamp(((img.clone().cpu()[0] * torch.tensor(MNIST_STD).view(1, 1, 1)).permute(1, 2, 0) + torch.tensor(MNIST_MEAN).view(1, 1, 1)) * 255.0, 0, 255).to(torch.uint8).numpy()
      else:
        img = torch.clamp(((img.clone().cpu()[0] * torch.tensor(CIFAR_STD).view(3, 1, 1)).permute(1, 2, 0) + torch.tensor(CIFAR_MEAN).view(1, 1, 3)) * 255.0, 0, 255).to(torch.uint8).numpy()
      
      cv2.imwrite(f'ood_{i}.png', img)
      grid[i].imshow(img, cmap='gray' if mnist else None)
    
    plt.savefig(f'{"tiny" if tiny else ("mnist" if mnist else "cifar")}_grid.png')  
    plt.show(block=True)
    
      
    # print(img.shape, torch.mean(img), torch.std(img))
    # plt.imshow(torch.clamp(((img.clone().cpu()[0] * torch.tensor(CIFAR_STD).view(3, 1, 1)).permute(1, 2, 0) + torch.tensor(CIFAR_MEAN).view(1, 1, 3)) * 255.0, 0, 255).to(torch.uint8).numpy())
    # plt.imshow(torch.clamp(((img.clone().cpu()[0] * torch.tensor(MNIST_STD).view(1, 1, 1)).permute(1, 2, 0) + torch.tensor(MNIST_MEAN).view(1, 1, 1)) * 255.0, 0, 255).to(torch.uint8).numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.show(block=True)
    # time.sleep(0.5)
      
  
  exit(0)
  
  from hyper.data.fast_mnist import FastMNIST
  import matplotlib.pyplot as plt
  import time
  
  mnist = FastMNIST(DATA_DIR, train=True, download=True)
  mnist_transforms = TF.RandomOrder([
    TF.RandomInvert(p=0.4),
    TF.RandomErasing(p=0.5),
    TF.ElasticTransform(
      alpha=90.0,
      sigma=6.0,
    ),
    TF.ColorJitter(
      contrast=(0.5, 1.5)
    ),
    TF.RandomErasing(
      scale=(0.3, 0.6),
      value=1.0,
      p=0.8
    ),
    TF.ElasticTransform(
      alpha=90.0,
      sigma=6.0,
    ),
    TF.GaussianBlur(
      kernel_size=(5, 5),
      sigma=(2.0, 6.)
    ),
  ])
  mnist_ood = ImageOODDataset(
    num_images=20,
    image_size=(28, 28),
    image_channels=1,
    id_dataset=mnist,
    id_transforms=mnist_transforms,
    id_prob=0.5
  )
  
  # pnoise = perlin_noise(ellipses_image)
  # print(pnoise.min(), pnoise.max(), pnoise.mean(), pnoise.std())
  for i in range(len(mnist_ood)):
    plt.imshow(mnist_ood[i].cpu().squeeze(0))
    plt.show(block=True)
    # time.sleep(0.5)
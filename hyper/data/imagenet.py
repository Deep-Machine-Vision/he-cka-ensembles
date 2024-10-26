# pulled from https://github.com/liznerski/eoe/blob/main/src/eoe/datasets/imagenet.py
import os.path as pt
import sys
from multiprocessing import shared_memory
from sre_constants import error as sre_constants_error
from typing import List, Tuple, Callable, Union

import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import UnidentifiedImageError
from multiprocessing.resource_tracker import unregister  # careful with this!
from torch.utils.data import Subset
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS, has_file_allowed_extension
from torchvision.datasets.imagenet import ImageFolder
from torchvision.datasets.imagenet import verify_str_arg
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import json
import os.path as pt
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose
from tqdm import tqdm

# from eoe.utils.logger import Logger
# from eoe.utils.stats import RunningStats
# from eoe.utils.transformations import ConditionalCompose
# from eoe.utils.transformations import GPU_TRANSFORMS, Normalize, GlobalContrastNormalization

GCN_NORM = 1
STD_NORM = 0
NORM_MODES = {  # the different transformation dummies that will be automatically replaced by torchvision normalization instances
    'norm': STD_NORM, 'normalise': STD_NORM, 'normalize': STD_NORM,
    'gcn-norm': GCN_NORM, 'gcn-normalize': GCN_NORM, 'gcn-normalise': GCN_NORM,
}


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Applies global contrast normalization to a tensor; i.e., subtracts the mean across features (pixels) and normalizes by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note that this is a *per sample* normalization globally across features (and not across the dataset).
    """
    assert scale in ('l1', 'l2')
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x


class BaseADDataset(ABC):
    def __init__(self, root: str):
        """
        An abstract Anomaly Detection (AD) dataset. All AD datasets have a _train_set and a _test_set split that need
        to be prepared during their __init__. They also have a list of normal and anomaly classes.
        @param root: Defines the root directory for all datasets. Most of them get automatically downloaded if not present
            at this directory. Each dataset has its own subdirectory (e.g., eoe/data/datasets/imagenet/).
        """
        super().__init__()
        self.root: str = root  # root path to data

        self.n_classes: int = 2  # 0: normal, 1: outlier
        self.normal_classes: List[int] = None  # tuple with original class labels that define the normal class
        self.outlier_classes: List[int] = None  # tuple with original class labels that define the outlier class

        self._train_set: torch.utils.data.Subset = None  # the actual dataset for training data
        self._test_set: torch.utils.data.Subset = None  # the actual dataset for test data

        self.shape: Tuple[int, int, int] = None  # shape of datapoints, c x h x w
        self.raw_shape: Tuple[int, int, int] = None  # shape of datapoint before preprocessing is applied, c x h x w

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """ Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set. """
        pass

    def __repr__(self):
        return self.__class__.__name__



import numpy as np
import scipy.fftpack as fp
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.ImageFilter import GaussianBlur, UnsharpMask
# from kornia import gaussian_blur2d
from torch import Tensor
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor, to_pil_image

BLUR_ID = 100
SHARPEN_ID = 101
HPF_ID = 102
KHPF_ID = 103
LPF_ID = 104
TRANSFORMS = {'blur': BLUR_ID, 'sharpen': SHARPEN_ID, 'hpf': HPF_ID, 'lpf': LPF_ID}


class ConditionalCompose(Compose):
    def __init__(self, conditional_transforms: List[Tuple[int, Callable, Callable]], gpu=False):
        """
        This composes multiple torchvision transforms. However, each transformation has two versions.
        ConditionalCompose executes the first version if the label matches the condition and the other if not.
        Note that this class should not be used for data transformation during testing as it uses class labels!
        We used it to experiment with different version of frequency filters in our frequency analysis experiments, however,
        have only reported results for equal filters on all labels in the paper.
        @param conditional_transforms: A list of tuples (cond, trans1, trans2).
            ConditionalCompose iterates of all elements and at each time executes
            trans1 on the data if the label equals cond and trans2 if not cond.
        @param gpu: whether to move the data that is to be transformed to the gpu first.
        """
        super(ConditionalCompose, self).__init__(None)
        self.conditional_transforms = conditional_transforms
        self.gpu = gpu

    def __call__(self, img, tgt):
        for cond, t1, t2 in self.conditional_transforms:
            t1 = (lambda x: x) if t1 is None else t1
            t2 = (lambda x: x) if t2 is None else t2
            if not self.gpu:
                if tgt == cond:
                    img = t1(img)
                else:
                    img = t2(img)
            else:
                tgt = torch.Tensor(tgt) if not isinstance(tgt, torch.Tensor) else tgt
                tgt = tgt.to(img.device)
                img = torch.where(tgt.reshape(-1, 1, 1, 1) == cond, t1(img), t2(img))
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.conditional_transforms:
            format_string += '\n'
            format_string += F'    {repr(t[1])} if {repr(t[0])} else {repr(t[2])}'
        format_string += '\n)'
        return format_string


def get_transform(transform: int, magnitude: float) -> Callable:
    if transform == BLUR_ID:
        transform = CpuGaussianBlur(magnitude)
    elif transform == SHARPEN_ID:
        transform = PilUnsharpMask(magnitude)
    elif transform == HPF_ID:
        transform = DFTHighPassFilter(int(magnitude))
    elif transform == LPF_ID:
        transform = DFTLowPassFilter(int(magnitude))
    else:
        raise NotImplementedError()
    return transform


class PilGaussianBlur(object):
    def __init__(self, magnitude: float):
        """ applies a Gaussian blur to PIL images (cpu)"""
        self.magnitude = magnitude

    def __call__(self, x: Image):
        return x.filter(GaussianBlur(radius=self.magnitude))

    def __repr__(self):
        return f"PilGaussianBlur radius {self.magnitude}"


class CpuGaussianBlur(object):
    def __init__(self, magnitude: float):
        """ applies a Gaussian blur to PIL images using Kornia, for which it temporarily transforms the image to a tensor (CPU)"""
        self.magnitude = magnitude
        self.sigma = self.magnitude
        self.k = 2 * int(int(self.sigma / 2) + 0.5) + 1

    def __call__(self, img: Image) -> Tensor:
        if self.sigma <= 0:
            return img
        else:
            img = to_tensor(img).unsqueeze(0)
            k = max(min(self.k, 2 * int(int(img.size(-1) / 2) + 0.5) - 1), 3)
            img = gaussian_blur2d(img, (k, k), (self.sigma, self.sigma))
            img = to_pil_image(img.squeeze(0))
            return img

    def __repr__(self) -> str:
        return 'CPU-BLUR'

    def __str__(self) -> str:
        return 'CPU-BLUR'


class PilUnsharpMask(object):
    def __init__(self, magnitude: float):
        """ applies an UnsharpMask to PIL images (CPU)"""
        self.magnitude = magnitude

    def __call__(self, x: Image):
        return x.filter(UnsharpMask(percent=int(self.magnitude * 100)))

    def __repr__(self):
        return f"PilUnsharpMask percent {self.magnitude}"


class Normalize(object):
    def __init__(self, normalize: transforms.Normalize):
        """ applies a typical torchvision normalization to tensors (GPU compatible) """
        self.normalize = normalize

    def __call__(self, img: Tensor) -> Tensor:
        return self.normalize(img)

    def __repr__(self) -> str:
        return 'GPU-'+self.normalize.__repr__()

    def __str__(self) -> str:
        return 'GPU-'+self.normalize.__str__()


class Blur(object):
    def __init__(self, blur: PilGaussianBlur):
        """ applies a Gaussian blur to tensors using Kornia (GPU compatible), reuses the parameters of a given PilGaussianBlur """
        self.blur = blur
        self.sigma = blur.magnitude
        self.k = 2 * int(int(self.sigma / 2) + 0.5) + 1

    def __call__(self, img: Tensor) -> Tensor:
        if self.sigma <= 0:
            return img
        else:
            k = max(min(self.k, 2 * int(int(img.size(-1) / 2) + 0.5) - 1), 3)
            return gaussian_blur2d(img, (k, k), (self.sigma, self.sigma))

    def __repr__(self) -> str:
        return 'GPU-'+self.blur.__repr__()

    def __str__(self) -> str:
        return 'GPU-'+self.blur.__str__()


class ToGrayscale(object):
    def __init__(self, t: transforms.Grayscale):
        """ removes the color channels of a given tensor of images (GPU compatible) """
        pass

    def __call__(self, img: Tensor) -> Tensor:
        return img.mean(1).unsqueeze(1)

    def __repr__(self) -> str:
        return 'GPU-Grayscale'

    def __str__(self) -> str:
        return 'GPU-Grayscale'


class MinMaxNorm(object):
    def __init__(self, norm: 'MinMaxNorm' = None):
        """ applies an image-wise min-max normalization (brings to 0-1 range) to a tensor of images (GPU compatible) """
        pass

    def __call__(self, img: Tensor) -> Tensor:
        img = img.flatten(1).sub(img.flatten(1).min(1)[0].unsqueeze(1)).reshape(img.shape)
        img = img.flatten(1).div(img.flatten(1).max(1)[0].unsqueeze(1)).reshape(img.shape)
        return img

    def __repr__(self) -> str:
        return 'GPU-MinMaxNorm'

    def __str__(self) -> str:
        return 'GPU-MinMaxNorm'


class DFTHighPassFilter(object):
    def __init__(self, magnitude: int = 1):
        """ applies a true high pass filter to a PIL image using numpy and an FFT (CPU) """
        self.magnitude = magnitude

    def __call__(self, img: Image) -> Image:
        if self.magnitude <= 0:
            return img
        else:
            img = np.asarray(img).astype(float) / 255
            gray = len(img.shape) == 2
            if gray:
                img = img[:, :, None]
            h, w, c = img.shape
            n = min(self.magnitude, min(w // 2, h // 2))
            for cc in range(c):
                f1 = fp.fft2(img[:, :, cc])
                f2 = fp.fftshift(f1)
                f2[w//2-n:w//2+n, h//2-n:h//2+n] = 0
                img[:, :, cc] = fp.ifft2(fp.ifftshift(f2)).real
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(np.uint8)
            if gray:
                img = img[:, :, 0]
            return Image.fromarray(img)

    def __repr__(self) -> str:
        return f'DFT-HPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'DFT-HPF-{self.magnitude}'


class GpuDFTHighPassFilter(object):
    def __init__(self, hpf: DFTHighPassFilter):
        """
        Applies a true high pass filter to a tensor of images using torch and an FFT (GPU compatible).
        Reuses the params of a given CPU HPF.
        """
        self.magnitude = hpf.magnitude
        self.norm = MinMaxNorm()

    def __call__(self, img: Tensor) -> Tensor:
        if self.magnitude <= 0:
            return img
        else:
            n, c, h, w = img.shape
            e = min(self.magnitude, min(w // 2, h // 2))
            f1 = torch.fft.fft2(img)
            f2 = torch.fft.fftshift(f1)
            f2[:, :, h//2-e:h//2+e, w//2-e:w//2+e] = 0
            img = torch.fft.ifft2(torch.fft.ifftshift(f2)).real
            img = self.norm(img)
            return img

    def __repr__(self) -> str:
        return f'GPU-DFT-HPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'GPU-DFT-HPF-{self.magnitude}'


class DFTLowPassFilter(object):
    def __init__(self, magnitude: int = 1):
        """ applies a true low pass filter to a PIL image using numpy and an FFT (CPU) """
        self.magnitude = magnitude

    def __call__(self, img: Image) -> Image:
        if self.magnitude <= 0:
            return img
        else:
            img = np.asarray(img).astype(float) / 255
            gray = len(img.shape) == 2
            if gray:
                img = img[:, :, None]
            h, w, c = img.shape
            n = min(self.magnitude, min(w // 2, h // 2))
            for cc in range(c):
                f1 = fp.fft2(img[:, :, cc])
                f2 = fp.fftshift(f1)
                f2[:, :n, :] = 0
                f2[:, -n:, :] = 0
                f2[:, :, :n] = 0
                f2[:, :, -n:] = 0
                img[:, :, cc] = fp.ifft2(fp.ifftshift(f2)).real
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(np.uint8)
            if gray:
                img = img[:, :, 0]
            return Image.fromarray(img)

    def __repr__(self) -> str:
        return f'DFT-LPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'DFT-LPF-{self.magnitude}'


class GpuDFTLowPassFilter(object):
    def __init__(self, lpf: DFTLowPassFilter):
        """
        Applies a true low pass filter to a tensor of images using torch and an FFT (GPU compatible).
        Reuses the params of a given CPU LPF.
        """
        self.magnitude = lpf.magnitude
        self.norm = MinMaxNorm()

    def __call__(self, img: Tensor) -> Tensor:
        if self.magnitude <= 0:
            return img
        else:
            n, c, h, w = img.shape
            e = min(self.magnitude, min(w // 2, h // 2))
            f1 = torch.fft.fft2(img)
            f2 = torch.fft.fftshift(f1)
            f2[:, :, :e, :] = 0
            f2[:, :, -e:, :] = 0
            f2[:, :, :, :e] = 0
            f2[:, :, :, -e:] = 0
            img = torch.fft.ifft2(torch.fft.ifftshift(f2)).real
            img = self.norm(img)
            return img

    def __repr__(self) -> str:
        return f'GPU-DFT-LPF-{self.magnitude}'

    def __str__(self) -> str:
        return f'GPU-DFT-LPF-{self.magnitude}'


class GlobalContrastNormalization(object):
    def __init__(self, gcn=None, scale='l1'):
        """
        Applies a global contrast normalization to a tensor of images;
        i.e. subtract mean across features (pixels) and normalize by scale,
        which is either the standard deviation, L1- or L2-norm across features (pixels).
        Note this is a *per sample* normalization globally across features (and not across the dataset).
        This is GPU compatible.
        """
        self.scale = scale
        if gcn is not None:
            assert gcn.scale == scale

    def __call__(self, x: torch.Tensor):  # x in [n, c, h, w]
        assert self.scale in ('l1', 'l2')
        n_features = int(np.prod(x.shape[1:]))
        mean = torch.mean(x.flatten(1), dim=1)[:, None, None, None]  # mean over all features (pixels) per sample
        x -= mean
        if self.scale == 'l1':
            x_scale = torch.mean(torch.abs(x.flatten(1)), dim=1)[:, None, None, None]
        if self.scale == 'l2':
            x_scale = torch.sqrt(torch.sum(x.flatten(1) ** 2, dim=1))[:, None, None, None] / n_features
        x /= x_scale
        return x


GPU_TRANSFORMS = {  # maps CPU versions of transformations to corresponding GPU versions
    transforms.Normalize: Normalize, CpuGaussianBlur: Blur, type(None): lambda x: None,
    Normalize: Normalize, Blur: Blur, MinMaxNorm: MinMaxNorm,
    DFTHighPassFilter: GpuDFTHighPassFilter,
    GlobalContrastNormalization: GlobalContrastNormalization,   # there is no CPU version implemented so far
    DFTLowPassFilter: GpuDFTLowPassFilter
}



class TorchvisionDataset(BaseADDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int, train_transform: Compose,
                 test_transform: Compose, classes: int, raw_shape: Tuple[int, int, int],
                 logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        An implementation of a Torchvision-style AD dataset. It provides a data loader for its train and test split each.
        There is a :method:`preview` that returns a collection of random batches of image samples from the loaders.

        TorchvisionDataset optimizes the transformation pipelines.
        It replaces normalization dummy strings (see :attr:`NORM_MODES`) with actual torchvision normalization instances for which
        it automatically extracts the empirical mean and std of the normal training data and caches it for later use.
        It also moves some transformations automatically to the GPU, for which it removes them from the pipeline
        and stores them in a separate attribute for later use in the ADTrainer (see :class:`eoe.training.ad_trainer.ADTrainer`).

        Implementations of TorchvisionDataset need to create the actual train and test dataset
        (i.e., self._train_set and self._test_set). They also need to create suitable subsets if `limit_samples` is not None.
        Note that self._train_set and self._test_set should always be instances of :class:`torch.utils.data.Subset` even if
        `limit_samples` is None. The training subset still needs to be set so that it excludes all anomalous
        training samples, and even if `normal_classes` contains all classes, the subset simply won't be a proper subset;
        i.e., the Subset instance will have all indices of the complete dataset.
        There is :method:`TorchvisionDataset.create_subset` that can be used for all this.

        @param root: Defines the root directory for all datasets. Most of them get automatically downloaded if not present
            at this directory. Each dataset has its own subdirectory (e.g., eoe/data/datasets/imagenet/).
        @param normal_classes: A list of normal classes. Normal training samples are all from these classes.
            Samples from other classes are not available during training. During testing, other classes will be anomalous.
        @param nominal_label: The integer defining the normal (==nominal) label. Usually 0.
        @param train_transform: Preprocessing pipeline used for training, includes all kinds of image transformations.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param test_transform: Preprocessing pipeline used for testing,
            includes all kinds of image transformations but no data augmentation.
            May contain the dummy transformation 'norm' that will be replaced with a torchvision normalization instance.
            The required mean and std of the normal training data will be extracted automatically.
        @param classes: The number of overall raw classes of this dataset. Static per dataset.
        @param raw_shape: The raw shape of the dataset samples before preprocessing is applied, shape: c x h x w.
        @param logger: Optional. Some logger instance. Is only required for logging warnings related to the datasets.
        @param limit_samples: Optional. If given, limits the number of different samples. That is, instead of using the
            complete dataset, creates a subset that is to be used. If `limit_samples` is an integer, samples a random subset
            with the provided size. If `limit_samples` is a list of integers, create a subset with the indices provided.
        @param train_conditional_transform: Optional. Similar to `train_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        @param test_conditional_transform: Optional. Similar to `test_transform` but conditioned on the label.
            See :class:`eoe.utils.transformations.ConditionalCompose`.
        """
        super().__init__(root)

        self.raw_shape = raw_shape
        self.normal_classes = tuple(normal_classes)
        normal_set = set(self.normal_classes)
        self.outlier_classes = [c for c in range(classes) if c not in normal_set]
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0
        self.logger = logger
        self.limit_samples = limit_samples

        # self.target_transform = transforms.Lambda(
        #     lambda x: self.anomalous_label if x in self.outlier_classes else self.nominal_label
        # )
        self.target_transform = None
        self.train_transform = deepcopy(train_transform)
        self.test_transform = deepcopy(test_transform)
        self.gpu_train_transform = lambda x: x
        self.gpu_test_transform = lambda x: x
        self.train_conditional_transform = deepcopy(train_conditional_transform)
        self.test_conditional_transform = deepcopy(test_conditional_transform)
        self.gpu_train_conditional_transform = lambda x, y: x
        self.gpu_test_conditional_transform = lambda x, y: x

        self._unpack_transforms()
        if any([isinstance(t, str) for t in (self.train_transform.transforms + self.test_transform.transforms)]):
            self._update_transforms(self._get_raw_train_set())
            self._unpack_transforms()
        self._split_transforms()

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    def create_subset(self, dataset_split: VisionDataset, class_labels: List[int], ) -> Subset:
        """
        Creates a Subset instance for the given dataset split.
        The subset will only contain indices for samples belonging to normal classes according to :attr:`self.normal_classes`.
        Further, if :attr:`self.limit_samples` is an integer and not None, it will contain a random subset of
        these normal indices so that len(indices) == `self.limit_samples`.

        However, if `self.limit_samples` is a list of integers, it will overwrite the indices to exactly those defined by
        `self.limit_samples`. Note that in this case it is not assured that all indices are still normal because
        `limit_samples` is not checked for that.

        Since this method uses :attr:`self.normal_classes` and :attr:`self.limit_samples`, it should be used only after
        those have been initialized. In other words, invoke this method after the implementation of TorchvisionDataset
        invoked super().__init__(...).

        @param dataset_split: The prepared dataset split (e.g., CIFAR-100).
        @param class_labels: A list of all sample-wise integer class labels
            (i.e., not for 'normal' and 'anomalous' but, e.g., 'airplane', 'car', etc.). The length of this list has
            to equal the size of the dataset.
        @return: The subset containing samples as described above.
        """
        if self.normal_classes is None:
            raise ValueError('Subsets can only be created once the dataset has been initialized.')
        # indices of normal samples according to :attr:`normal_classes`
        normal_idcs = np.argwhere(
            np.isin(np.asarray(class_labels), self.normal_classes)
        ).flatten().tolist()
        if isinstance(self.limit_samples, (int, float)) and self.limit_samples < np.infty:
            # sample randomly s.t. len(normal_idcs) == :attr:`limit_samples`
            normal_idcs = sorted(np.random.choice(normal_idcs, min(self.limit_samples, len(normal_idcs)), False))
        elif not isinstance(self.limit_samples, (int, float)):
            # set indices to :attr:`limit_samples`, note that these are not necessarily normal anymore
            normal_idcs = self.limit_samples
        return Subset(dataset_split, normal_idcs)

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples.
        @param train: Whether to consider training or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        ds = self.train_set if train else self.test_set
        return dict(Counter([self.target_transform(t) for t in np.asarray(ds.dataset.targets)[list(set(ds.indices))]]))

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, replacement=False,
                num_workers: int = 0, persistent=False, prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Returns torch loaders for the train and test split of the dataset.
        @param batch_size: the batch size for the loaders.
        @param shuffle_train: whether to shuffle the train split at the start of each iteration of the data loader.
        @param shuffle_test: whether to shuffle the test split at the start of each iteration of the data loader.
        @param replacement: whether to sample data with replacement.
        @param num_workers: See :class:`torch.util.data.dataloader.DataLoader`.
        @param persistent: See :class:`torch.util.data.dataloader.DataLoader`.
        @param prefetch_factor: See :class:`torch.util.data.dataloader.DataLoader`.
        @return: A tuple (train_loader, test_loader).
        """
        # classes = None means all classes
        train_loader = DataLoader(
            dataset=self.train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
            persistent_workers=persistent, prefetch_factor=prefetch_factor,
            sampler=RandomSampler(self.train_set, replacement=replacement) if shuffle_train else None
        )
        test_loader = DataLoader(
            dataset=self.test_set, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
            persistent_workers=persistent, prefetch_factor=prefetch_factor,
            sampler=RandomSampler(self.test_set, replacement=replacement) if shuffle_test else None
        )
        return train_loader, test_loader

    def preview(self, percls=40, train=True, classes=(0, 1)) -> torch.Tensor:
        """
        Generates a preview of the dataset; i.e., generates a figure of some randomly chosen outputs of the dataloader.
        Therefore, the data samples have already passed the complete preprocessing pipeline.

        @param percls: How many samples (per label) are shown.
        @param train: Whether to show training samples or test samples.
        @param classes: The labels for which images are shown. Defaults to (0, 1) for normal and anomalous.
        @return: A Tensor of images (n x c x h x w).
        """
        if train:
            loader, _ = self.loaders(10, num_workers=0, shuffle_train=True)
        else:
            _, loader = self.loaders(10, num_workers=0, shuffle_test=False)
        x, y, out = torch.FloatTensor(), torch.LongTensor(), []
        for xb, yb, _ in loader:
            xb = xb.cuda()
            if train:
                if isinstance(self, CombinedDataset):
                    xb[yb == self.nominal_label] = self.normal.gpu_train_conditional_transform(
                        xb[yb == self.nominal_label], [self.nominal_label] * len(xb[yb == self.nominal_label])
                    )
                    xb[yb == self.nominal_label] = self.normal.gpu_train_transform(xb[yb == self.nominal_label])
                    xb[yb != self.nominal_label] = self.oe.gpu_train_conditional_transform(
                        xb[yb != self.nominal_label], [self.anomalous_label] * len(xb[yb != self.nominal_label])
                    )
                    xb[yb != self.nominal_label] = self.oe.gpu_train_transform(xb[yb != self.nominal_label])
                else:
                    xb = self.gpu_train_conditional_transform(xb, yb)
                    xb = self.gpu_train_transform(xb)
            else:
                if isinstance(self, CombinedDataset):
                    xb = self.normal.gpu_test_conditional_transform(xb, yb)
                    xb = self.normal.gpu_test_transform(xb)
                else:
                    xb = self.gpu_test_conditional_transform(xb, yb)
                    xb = self.gpu_test_transform(xb)
            xb = xb.cpu()
            x, y = torch.cat([x, xb]), torch.cat([y, yb])
            if all([x[y == c].size(0) >= percls for c in classes]):
                break
        for c in sorted(set(y.tolist())):
            out.append(x[y == c][:percls])
        percls = min(percls, *[o.size(0) for o in out])
        out = [o[:percls] for o in out]
        return torch.cat(out)

    def _update_transforms(self, train_dataset: torch.utils.data.Dataset):
        """
        Replaces occurrences of the string 'Normalize' (or others, see :attr:`NORM_MODES`) within the train and test transforms
        with an actual `transforms.Normalize`. For this, extracts, e.g., the empirical mean and std of the normal data.
        Other transformations might require different statistics, but they will always be used as a mean and std in
        `transforms.Normalize`. For instance, GCN uses a max/min normalization, which can also be accomplished with
        `transforms.Normalize`.
        @param train_dataset: some raw training split of a dataset. In this context, raw means no data augmentation.
        """
        if any([isinstance(t, str) for t in (self.train_transform.transforms + self.test_transform.transforms)]):
            train_str_pos, train_str = list(
                zip(*[(i, t.lower()) for i, t in enumerate(self.train_transform.transforms) if isinstance(t, str)])
            )
            test_str_pos, test_str = list(
                zip(*[(i, t.lower()) for i, t in enumerate(self.test_transform.transforms) if isinstance(t, str)])
            )
            strs = train_str + test_str
            if len(strs) > 0:
                if not all([s in NORM_MODES.keys() for s in strs]):
                    raise ValueError(
                        f'Transforms for dataset contain a string that is not recognized. '
                        f'The only valid strings are {NORM_MODES.keys()}.'
                    )
                if not all([NORM_MODES[strs[i]] == NORM_MODES[strs[j]] for i in range(len(strs)) for j in range(i)]):
                    raise ValueError(f'Transforms contain different norm modes, which is not supported. ')
                if NORM_MODES[strs[0]] == STD_NORM:
                    if self.load_cached_stats(NORM_MODES[strs[0]]) is not None:
                        self.logger.print(f'Use cached mean/std of training dataset with normal classes {self.normal_classes}')
                        mean, std = self.load_cached_stats(NORM_MODES[strs[0]])
                    else:
                        loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, )
                        acc = RunningStats()
                        desc = f'Extracting mean/std of training dataset with normal classes {self.normal_classes}'
                        for x, _, _ in tqdm(loader, desc=desc):
                            acc.add(x.permute(1, 0, 2, 3).flatten(1).permute(1, 0))
                        mean, std = acc.mean(), acc.std()
                        self.cache_stats(mean, std, NORM_MODES[strs[0]])
                    norm = transforms.Normalize(mean, std, inplace=False)
                else:
                    loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
                    all_x = []
                    for x, _, _ in tqdm(loader, desc='Extracting max and min of GCN normalized training dataset'):
                        x = torch.stack([global_contrast_normalization(xi, scale='l1') for xi in x])
                        all_x.append(x)
                    all_x = torch.cat(all_x)
                    tmin, tmax = all_x.min().item(), all_x.max().item()
                    norm = transforms.Compose([
                        GlobalContrastNormalization(scale='l1'),
                        transforms.Normalize([tmin] * all_x.size(1), [tmax - tmin] * all_x.size(1), inplace=False)
                    ])
                for i in train_str_pos:
                    self.train_transform.transforms[i] = norm
                for i in test_str_pos:
                    self.test_transform.transforms[i] = norm

    def load_cached_stats(self, norm_mode: int) -> Tuple[torch.Tensor, torch.Tensor]:  # returns mean and std of dataset
        """
        Tries to load cached statistics of the normal dataset. :method:`_update_transforms` will first try to use the cache
        before trying to compute the statistics again.
        @param norm_mode: The norm_mode for which the statistics are to be loaded.
        @return: The "mean" and "std" for the corresponding norm_mode (see :attr:`NORM_MODES`)
        """
        file = pt.join(self.root, 'stats_cache.json')
        if pt.exists(file):
            with open(file, 'r') as reader:
                cache = json.load(reader)
            if str(type(self)) in cache and str(norm_mode) in cache[(str(type(self)))] \
                    and json.dumps(self.normal_classes) in cache[str(type(self))][str(norm_mode)]:
                mean, std = cache[str(type(self))][str(norm_mode)][json.dumps(self.normal_classes)]
                return torch.tensor(mean), torch.tensor(std)
        return None

    def cache_stats(self, mean: torch.Tensor, std: torch.Tensor, norm_mode: int):  # caches mean and std of dataset
        """
        Caches the "mean" and "std" for some norm_mode (see :attr:`NORM_MODES`). Is used by :method:`_update_transforms`.
        @param mean: the "mean" (might actually be some other statistic but will be used as a mean for `transforms.Normalize`).
        @param std: the "std" (might actually be some other statistic but will be used as a std for `transforms.Normalize`).
        @param norm_mode: the norm_mode for which the "mean" and "std" are cached.
        """
        file = pt.join(self.root, 'stats_cache.json')
        if not pt.exists(file):
            with open(file, 'w') as writer:
                json.dump({str(type(self)): {str(norm_mode): {}}}, writer)
        with open(file, 'r') as reader:
            cache = json.load(reader)
        if str(type(self)) not in cache:
            cache[str(type(self))] = {}
        if str(norm_mode) not in cache[(str(type(self)))]:
            cache[(str(type(self)))][str(norm_mode)] = {}
        cache[(str(type(self)))][str(norm_mode)][json.dumps(self.normal_classes)] = (mean.numpy().tolist(), std.numpy().tolist())
        with open(file, 'w') as writer:
            json.dump(cache, writer)

    def _split_transforms(self):
        """
        This moves some parts of the preprocessing pipelines (self.train_transform, self.test_transform, etc.)
        to a GPU pipeline. That is, for instance, self.gpu_train_transform. The method automatically looks for transformations
        that appear in :attr:`eoe.utils.transformations.GPU_TRANSFORMS` and replaces them with corresponding GPU versions.
        The :class:`eoe.training.ad_trainer.ADTrainer` accesses self.gpu_train_transform and the other gpu pipelines and
        applies them right after having retrieved the tensors from the dataloader and putting them to the GPU.
        """
        gpu_trans, n = [], 0
        for i, t in enumerate(deepcopy(self.train_transform.transforms)):
            if type(t) in GPU_TRANSFORMS:
                gpu_trans.append(GPU_TRANSFORMS[type(t)](t))
                del self.train_transform.transforms[i-n]
                n += 1
            elif n > 0 and not isinstance(t, transforms.ToTensor):
                raise ValueError('A CPU-only transform follows a GPU transform. This is not supported atm.')
        self.gpu_train_transform = Compose(gpu_trans)
        if not all([isinstance(t, (Normalize, GlobalContrastNormalization)) for t in gpu_trans]):
            raise ValueError(f'Since gpu_train_conditional_transform is applied before gpu_train_transform, '
                             f'gpu_train_transform is not allowed to contain transforms other than Normalize. '
                             f'Otherwise the conditional transforms that are used for the multiscale experiments would be '
                             f'influenced by multiscale generating augmentations.')

        gpu_trans, n = [], 0
        for i, t in enumerate(deepcopy(self.test_transform.transforms)):
            if type(t) in GPU_TRANSFORMS:
                gpu_trans.append(GPU_TRANSFORMS[type(t)](t))
                del self.test_transform.transforms[i-n]
                n += 1
            elif n > 0 and not isinstance(t, transforms.ToTensor):
                raise ValueError('A CPU-only transform follows a GPU transform. This is not supported atm.')
        self.gpu_test_transform = Compose(gpu_trans)
        if not all([isinstance(t, (Normalize, GlobalContrastNormalization)) for t in gpu_trans]):
            raise ValueError(f'Since gpu_test_conditional_transform is applied before gpu_test_transform, '
                             f'gpu_test_transform is not allowed to contain transforms other than Normalize. '
                             f'Otherwise the conditional transforms that are used for the multiscale experiments would be '
                             f'influenced by multiscale generating augmentations.')

        gpu_trans, n = [], 0
        for i, (cond, t1, t2) in enumerate(deepcopy(self.train_conditional_transform.conditional_transforms)):
            if type(t1) in GPU_TRANSFORMS and type(t2) in GPU_TRANSFORMS:
                gpu_trans.append((cond, GPU_TRANSFORMS[type(t1)](t1), GPU_TRANSFORMS[type(t2)](t2)))
                del self.train_conditional_transform.conditional_transforms[i-n]
                n += 1
            elif n > 0:
                raise ValueError('A CPU-only transform follow a GPU transform. This is not supported atm.')
        self.gpu_train_conditional_transform = ConditionalCompose(gpu_trans, gpu=True)

        gpu_trans, n = [], 0
        for i, (cond, t1, t2) in enumerate(deepcopy(self.test_conditional_transform.conditional_transforms)):
            if type(t1) in GPU_TRANSFORMS and type(t2) in GPU_TRANSFORMS:
                gpu_trans.append((cond, GPU_TRANSFORMS[type(t1)](t1), GPU_TRANSFORMS[type(t2)](t2)))
                del self.test_conditional_transform.conditional_transforms[i-n]
                n += 1
            elif n > 0:
                raise ValueError('A CPU-only transform follow a GPU transform. This is not supported atm.')
        self.gpu_test_conditional_transform = ConditionalCompose(gpu_trans, gpu=True)

    def _unpack_transforms(self):
        """ This "unpacks" preprocessing pipelines so that there is Compose inside of a Compose """
        def unpack(t):
            if not isinstance(t, Compose):
                return [t]
            elif isinstance(t, Compose):
                return [tt for t in t.transforms for tt in unpack(t)]
        self.train_transform.transforms = unpack(self.train_transform)
        self.test_transform.transforms = unpack(self.test_transform)

        if self.train_conditional_transform is None:
            self.train_conditional_transform = ConditionalCompose([])
        for cond, t1, t2 in self.train_conditional_transform.conditional_transforms:
            assert not isinstance(t1, Compose) and not isinstance(t2, Compose), 'No Compose inside a ConditionalCompose allowed!'
        if self.test_conditional_transform is None:
            self.test_conditional_transform = ConditionalCompose([])
        for cond, t1, t2 in self.test_conditional_transform.conditional_transforms:
            assert not isinstance(t1, Compose) and not isinstance(t2, Compose), 'No Compose inside a ConditionalCompose allowed!'

    @abstractmethod
    def _get_raw_train_set(self):
        """
        Implement this to return a training set with the corresponding normal class that is used for extracting the mean and std.
        See :method:`_update_transforms`.
        """
        raise NotImplementedError()


class CombinedDataset(TorchvisionDataset):
    def __init__(self, normal_ds: TorchvisionDataset, oe_ds: TorchvisionDataset):
        """
        Creates a combined dataset out of a normal dataset and an Outlier Exposure (OE) dataset.
        The test split and test dataloader of the combined dataset will be the same as the ones of the normal dataset, which has
        both normal and anomalous samples for testing.
        The train split, however, will be a combination of normal training samples and anomalous OE samples.
        For this, it creates a ConcatDataset as a train split.
        More importantly, it creates a :class:`BalancedConcatLoader` for the train split that yields balanced batches
        of equally many normal and OE samples. Note that, the overall returned training batches thus have two times the original
        batch size. If there are not enough OE samples to have equally many different OE samples for the complete normal
        training set, start a new iteration of the OE dataset.
        @param normal_ds: The normal dataset containing only normal training samples but both anomalous and normal test samples.
        @param oe_ds: The Outlier Exposure (OE) dataset containing auxiliary anomalies for training.
        """
        self.normal = normal_ds
        self.oe = oe_ds
        self._train_set = ConcatDataset([self.normal.train_set, self.oe.train_set])
        self._test_set = self.normal.test_set

        self.raw_shape = self.normal.raw_shape
        self.normal_classes = self.normal.normal_classes
        self.outlier_classes = self.normal.outlier_classes
        self.nominal_label = self.normal.nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0
        self.logger = self.normal.logger
        self.limit_samples = self.oe.limit_samples

    def n_normal_anomalous(self, train=True) -> dict:
        """
        Extract the number of normal and anomalous data samples.
        @param train: Whether to consider training (including OE) or test samples.
        @return: A dictionary like {0: #normal_samples, 1: #anomalous_samples} (may change depending on the nominal label)
        """
        if train:
            normal = self.normal.n_normal_anomalous()
            oe = self.oe.n_normal_anomalous()
            return {k: normal.get(k, 0) + oe.get(k, 0) for k in set.union(set(normal.keys()), set(oe.keys()))}
        else:
            return self.normal.n_normal_anomalous(train)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0, persistent=False, prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        Returns the normal datasets' test loader.
        For training, returns a :class:`BalancedConcatLoader` that yields balanced batches
        of equally many normal and OE samples. Note that, the overall returned training batches thus have two times the original
        batch size. If there are not enough OE samples to have equally many different OE samples for the complete normal
        training set, start a new iteration of the OE dataset.
        For a description of the parameters see :method:`eoe.datasets.bases.TorchvisionDataset.loaders`.
        @return: a tuple of (train_loader, test_loader)
        """
        # classes = None means all classes
        normal_train_loader, test_loader = self.normal.loaders(
            batch_size, shuffle_train, shuffle_test, False, num_workers, persistent, prefetch_factor
        )
        oe_train_loader, _ = self.oe.loaders(
            batch_size, shuffle_train, shuffle_test, len(self.oe.train_set.indices) >= 10000, num_workers,
            persistent, prefetch_factor,
        )
        return BalancedConcatLoader(normal_train_loader, oe_train_loader), test_loader

    def _get_raw_train_set(self):
        return None  # doesn't make sense for a combined dataset


class BalancedConcatLoader(object):
    def __init__(self, normal_loader: DataLoader, oe_loader: DataLoader):
        """
        The balanced concat loader samples equally many samples from the normal and oe loader per batch.
        Both types of batches simply get concatenated to form the final batch.
        @param normal_loader: The normal data loader.
        @param oe_loader: The OE data loader.
        """
        self.normal_loader = normal_loader
        self.oe_loader = oe_loader
        if len(self.oe_loader.dataset) < len(self.normal_loader.dataset):
            r = int(np.ceil(len(self.normal_loader.dataset) / len(self.oe_loader.dataset)))
            self.oe_loader.dataset.indices = np.asarray(
                self.oe_loader.dataset.indices
            ).reshape(1, -1).repeat(r, axis=0).reshape(-1).tolist()

    def __iter__(self):
        self.normal_iter = iter(self.normal_loader)
        self.oe_iter = iter(self.oe_loader)
        return self

    def __next__(self):
        normal = next(self.normal_iter)  # imgs, lbls, idxs
        oe = next(self.oe_iter)
        while oe[1].size(0) < normal[1].size(0):
            oe = [torch.cat(a) for a in zip(oe, next(self.oe_iter))]
        oe[-1].data += len(self.normal_loader.dataset.dataset)  # offset indices of OE dataset with normal dataset length
        return [torch.cat([i, j[:i.shape[0]]]) for i, j in zip(normal, oe)]

    def __len__(self):
        return len(self.normal_loader)



import numpy as np


def encode_shape_and_image(img: np.ndarray) -> np.ndarray:
    # encodes the shape and the actual data into one flat uint8 array by using the first 15 bytes to encode the shape
    assert img.dtype == np.uint8 and img.ndim == 3, "requires a uint8 image"
    res = np.ndarray(shape=(15 + img.nbytes,), dtype=img.dtype)
    for i, dim in enumerate(img.shape):
        for j, digit in enumerate(f"{dim:0>5d}"):
            res[i*5+j] = int(digit)
    res[15:] = img.flatten()[:]
    return res


def decode_shape_and_image(shpimg: np.ndarray) -> np.ndarray:
    # decodes a flat uint8 array into an image of correct shape by parsing the first 15 bytes assuming they encode the shape
    assert shpimg.dtype == np.uint8 and shpimg.ndim == 1, "requires a flat uint8 representations of an image"
    shp = (
        int(''.join([str(i) for i in shpimg[:5]])),
        int(''.join([str(i) for i in shpimg[5:10]])),
        int(''.join([str(i) for i in shpimg[10:15]]))
    )
    return shpimg[15:].reshape(shp)


from typing import Callable
from typing import List, Tuple


class ADImageNet(TorchvisionDataset):
    ad_classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
                  'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
                  'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner',
                  'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']  # the 30 AD classes
    base_folder = 'imagenet_ad'  # appended to root directory as a subdirectory

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose, 
                 raw_shape: Tuple[int, int, int], logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for ImageNet-30. Following Hendrycks et al. (https://arxiv.org/abs/1812.04606) this AD dataset
        is limited to 30 of the 1000 classes of ImageNet (see :attr:`ADImageNet.ad_classes`). Accordingly, the
        class indices are redefined, ranging from 0 to 29, ordered alphabetically.
        Implements :class:`eoe.datasets.bases.TorchvisionDataset`.

        This dataset doesn't provide an automatic download. The data needs to be either downloaded from
        https://github.com/hendrycks/ss-ood, which already contains only the AD classes, or from https://www.image-net.org/.
        It needs to be placed in `root`/imagenet_ad/.
        """
        # root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 30, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = MyImageNet(
            self.root, split='train', transform=self.train_transform, target_transform=self.target_transform,
            conditional_transform=self.train_conditional_transform, logger=logger
        )
        # The following removes all samples from classes not in ad_classes
        # This shouldn't be necessary if the data from https://github.com/hendrycks/ss-ood is used
        self.train_ad_classes_idx = [self._train_set.class_to_idx[c] for c in self.ad_classes]
        self._train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.train_ad_classes_idx.index(t) if t in self.train_ad_classes_idx else np.nan for t in self._train_set.targets
        ]
        self._train_set.samples = [(s, tn) for (s, to), tn in zip(self._train_set.samples, self._train_set.targets)]

        # create a subset using only normal samples and limit the variety according to :attr:`limit_samples`
        self._train_set = self.create_subset(self._train_set, self._train_set.targets)

        self._test_set = MyImageNet(
            root=self.root, split='val', transform=self.test_transform, target_transform=self.target_transform,
            conditional_transform=self.test_conditional_transform, logger=logger
        )
        # The following removes all samples from classes not in ad_classes
        # This shouldn't be necessary if the data from https://github.com/hendrycks/ss-ood is used
        self.test_ad_classes_idx = [self._test_set.class_to_idx[c] for c in self.ad_classes]
        self._test_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            self.test_ad_classes_idx.index(t) if t in self.test_ad_classes_idx else np.nan
            for t in self._test_set.targets
        ]
        self._test_set.samples = [(s, tn) for (s, to), tn in zip(self._test_set.samples, self._test_set.targets)]
        self._test_set = Subset(
            self._test_set,
            np.argwhere(
                np.isin(np.asarray(self._test_set.targets), list(range(len(self.ad_classes))))
            ).flatten().tolist()
        )
        
        assert self.test_ad_classes_idx == self.train_ad_classes_idx

    def _get_raw_train_set(self):
        train_set = MyImageNet(
            self.root, split='train',
            transform=transforms.Compose([transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(), ]),
            target_transform=self.target_transform, logger=self.logger
        )
        train_ad_classes_idx = [train_set.class_to_idx[c] for c in self.ad_classes]
        train_set.targets = [  # t = nan if not in ad_classes else give id in order of ad_classes
            train_ad_classes_idx.index(t) if t in train_ad_classes_idx else np.nan for t in train_set.targets
        ]
        train_set.samples = [(s, tn) for (s, to), tn in zip(train_set.samples, train_set.targets)]
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class MyImageNet(ImageFolder):
    cache = {'train': {}, 'val': {}}

    def __init__(self, root: str, split: str = 'train', transform: Callable = None, target_transform: Callable = None,
                 conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's ImageNet s.t. it handles the optional conditional transforms, caching of file paths,
        and shared memory loading. See :class:`eoe.datasets.bases.TorchvisionDataset` for conditional transforms.
        Also, returns (img, target, index) in __get_item__ instead of (img, target).

        Creating a list of all image file paths can take some time for the full ImageNet-1k dataset, which is why
        we simply cache this in RAM (see :attr:`MyImageNet.cache`) once loaded at the start of the training so that we
        don't need to repeat this at the start of training each new class-seed combination
        (see :method:`eoe.training.ad_trainer.ADTrainer.run`).

        This implementation uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using MyImageNet run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        MyImageNet automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.
        """
        self.logger = kwargs.pop('logger', None)
        super(DatasetFolder, self).__init__(root, transform=transform, target_transform=target_transform, **kwargs)
        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.split_folder = pt.join(self.root, self.split)

        # ------ used cached file paths and other metadata or load and cache them if not available yet
        if len(self.cache[self.split]) == 0:
            print('Load ImageNet meta and cache it...')

            self.classes, self.class_to_idx = self.find_classes(self.split_folder)
            self.imgs = self.samples = self.make_dataset(
                self.split_folder, self.class_to_idx, is_valid_file=self.is_valid_file,
            )
            self.targets = [s[1] for s in self.samples]

            self.cache[self.split] = {}
            self.cache[self.split]['classes'] = self.classes
            self.cache[self.split]['class_to_idx'] = self.class_to_idx
            self.cache[self.split]['samples'] = self.samples
            self.cache[self.split]['targets'] = self.targets
            
            if self.logger is not None:
                size = sys.getsizeof(self.cache[self.split]['samples']) + sys.getsizeof(self.cache[self.split]['targets'])
                size += sys.getsizeof(self.cache[self.split]['classes']) + sys.getsizeof(self.cache[self.split]['class_to_idx'])
                self.logger.logtxt(
                    f"Cache size of {str(type(self)).split('.')[-1][:-2]}'s meta for split {self.split} is {size * 1e-9:0.3f} GB"
                )
        else:
            print('Use cached ImageNet meta.')
            self.classes = self.cache[self.split]['classes']
            self.class_to_idx = self.cache[self.split]['class_to_idx']
            self.imgs = self.samples = self.cache[self.split]['samples']
            self.targets = self.cache[self.split]['targets']

        self.loader = kwargs.get('loader', default_loader)
        self.extensions = kwargs.get('extensions', None)
        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        path, target = self.samples[index]
        
        try:  # try to use shared memory if available
            img = self._load_image_from_shared_memory(index)
            # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
        except FileNotFoundError:  # Shared memory (cached imagenet) not found, load from disk
            img = self.loader(path)
            # self.logger.logtxt(f'{self.split: >5}: Disk load for {index}', prnt=False)

        if self.target_transform is not None:
            print('TRANSFORM!')
            target = self.target_transform(target)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)
                
        try:
            if img.std() < 1e-15:  # random crop might yield completely white images (in case of nail)
                # img, target, index = self[index]
                img, target = self[index]
        except:
            pass
        # print('IMG', img.shape, 'T', target)
        return img, target  # , index

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        shm = shared_memory.SharedMemory(name=f'imagenet_{self.split}_{index}')
        img = decode_shape_and_image(np.copy(np.ndarray(shm.size, dtype=np.uint8, buffer=shm.buf)))
        shm.close()
        # The following line is a fix to make shared_memory work with unrelated processes.
        # Shared memory can cause memory leaks (indefinitely persisting memory allocations)
        # if the resources are not properly released/cleaned-up.
        # Python makes sure to prevent this leak by automatically starting a hidden resource_tracker process that
        # outlives the parent. It terminates and cleans up any shared memory (i.e., releases it) once
        # the parent dies. Unfortunately, each unrelated python process starts its own resource_tracker, which is
        # unaware of the other trackers and processes. This results in the tracker releasing all shared memory instances that
        # the parent process has been linked to--even the ones that it read but didn't write--once the parent terminates.
        # Other unrelated processes--e.g., the one that created the shared memory instance or other processes
        # that still want to read the data--can thus not access the data anymore since it has been released already.
        # One solution would be to make sure that all the processes use the same resource_tracker.
        # However, this would require to have one mother process that starts all experiments on the machine, which
        # would be very annoying in practice (e.g., one would have to wait until all processes are finished until new
        # experiments can be started that use the shared memory).
        # The other solution is presented below. Since the datasets only read and never write shared memory, we
        # can quite safely tell the resource_tracker that "we are going to deal with the clean-up manually" by unregistering
        # the read shared memory from this resource_tracker.
        # It is, however, very important to not do this with the shared-memory-creating process since this could cause
        # memory leaks as at least one process must release the resources!!!
        # See https://stackoverflow.com/questions/64102502/shared-memory-deleted-at-exit.
        unregister(shm._name, 'shared_memory')
        img = to_pil_image(img)
        return img

    def is_valid_file(self, file: str) -> bool:
        # check for file extension and ignore corrupt file in hendrycks' imagenet_30 dataset
        return has_file_allowed_extension(file, IMG_EXTENSIONS) and not file.endswith('airliner/._1.JPEG')


class ADImageNet21k(TorchvisionDataset):
    base_folder = pt.join('imagenet22k', 'fall11_whole_extracted')  # appended to root directory as subdirectories
    img_cache_size = 10000  # cache up to this many MB of images to RAM

    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger = None, limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None, test_conditional_transform: ConditionalCompose = None):
        """
        AD dataset for ImageNet-21k. Implements :class:`eoe.datasets.bases.TorchvisionDataset`.
        Doesn't use any class labels, and doesn't have a test split. Therefore, this is only suitable to be used as OE.

        This implementation also automatically caches some images in RAM if limit_samples is not np.infty.
        It only caches up to ~10 GB of data. The rest will be loaded from disk or shared memory as usual.
        Caching samples in RAM only makes sense for experiment with very limited amount of OE.
        For example, if there are only 2 OE samples, it doesn't make sense to reload them from the disk all the time.
        Note that data augmentation will still be applied on images loaded from RAM.

        ADImageNet21k doesn't provide an automatic download. The data needs to be downloaded from https://www.image-net.org/
        and placed in `root`/imagenet22k/fall11_whole_extracted/.
        """
        root = pt.join(root, self.base_folder)
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 21811, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform,
        )

        self._train_set = ImageNet22K(
            self.root, transform=self.train_transform, target_transform=self.target_transform, logger=self.logger,
            conditional_transform=self.train_conditional_transform, subset='_subset' in self.base_folder
        )
        normal_idcs = np.argwhere(
            np.isin(np.asarray(self._train_set.targets), self.normal_classes)
        ).flatten().tolist()
        if isinstance(limit_samples, (int, float)) and limit_samples < np.infty:
            normal_idcs = sorted(np.random.choice(normal_idcs, min(limit_samples, len(normal_idcs)), False))
        elif not isinstance(limit_samples, (int, float)):
            normal_idcs = limit_samples
        if limit_samples != np.infty:
            self._train_set.cache(normal_idcs[:ADImageNet21k.img_cache_size])
        self._train_set = Subset(self._train_set, normal_idcs)

    def _get_raw_train_set(self):
        train_set = ImageNet22K(
            self.root, transform=transforms.Compose([
                transforms.Resize(self.raw_shape[-1]), transforms.CenterCrop(224), transforms.ToTensor(),
            ]),
            target_transform=self.target_transform, subset='_subset' in self.base_folder
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.targets), self.normal_classes)
            ).flatten().tolist()
        )


class ImageNet22K(ImageFolder):
    imagenet1k_pairs = [
        ('acorn', 'n12267677'),
        ('airliner', 'n02690373'),
        ('ambulance', 'n02701002'),
        ('american_alligator', 'n01698640'),
        ('banjo', 'n02787622'),
        ('barn', 'n02793495'),
        ('bikini', 'n02837789'),
        ('digital_clock', 'n03196217'),
        ('dragonfly', 'n02268443'),
        ('dumbbell', 'n03255030'),
        ('forklift', 'n03384352'),
        ('goblet', 'n03443371'),
        ('grand_piano', 'n03452741'),
        ('hotdog', 'n07697537'),
        ('hourglass', 'n03544143'),
        ('manhole_cover', 'n03717622'),
        ('mosque', 'n03788195'),
        ('nail', 'n03804744'),
        ('parking_meter', 'n03891332'),
        ('pillow', 'n03938244'),
        ('revolver', 'n04086273'),
        ('rotary_dial_telephone', 'n03187595'),
        ('schooner', 'n04147183'),
        ('snowmobile', 'n04252077'),
        ('soccer_ball', 'n04254680'),
        ('stingray', 'n01498041'),
        ('strawberry', 'n07745940'),
        ('tank', 'n04389033'),
        ('toaster', 'n04442312'),
        ('volcano', 'n09472597')
    ]
    imagenet1k_labels = [label for name, label in imagenet1k_pairs]
    cached_samples = None
    cached_targets = None
    cached_classes = None
    cached_class_to_idx = None

    def __init__(self, root: str, *args, transform: Callable = None, target_transform: Callable = None, logger = None,
                 exclude_imagenet1k=True, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Implements a torchvision-style ImageNet22k dataset similar to torchvision's ImageNet implementation.
        Based on torchvision's ImageFolder implementation.
        The data needs to be downloaded manually from https://www.image-net.org/ and put in `root`/.

        Creating a list of all image file paths can take some time for the full ImageNet-22k dataset, which is why
        we simply cache this in RAM (see :attr:`ImageNet22K.cached_samples` etc.) once loaded at the start of the training
        so that we don't need to repeat this at the start of training each new class-seed combination
        (see :method:`eoe.training.ad_trainer.ADTrainer.run`).

        This implementation uses shared memory if prepared by other scripts (see experiments/caching folder).
        Loading data from shared memory speeds up data loading a lot if multiple experiments using ImageNet22k run in parallel
        on the same machine. However, using shared memory can cause memory leaks, which is why we don't recommend using it.
        ImageNet22k automatically falls back to loading the data from disk as usual if a sample is not found in shared memory.

        @param root: Root directory for data.
        @param transform: A preprocessing pipeline of image transformations.
        @param target_transform: A preprocessing pipeline of label (integer) transformations.
        @param logger: Optional logger instance. Only used for logging warnings.
        @param exclude_imagenet1k: Whether to exclude ImageNet-1k classes.
        @param conditional_transform: Optional. A preprocessing pipeline of conditional image transformations.
            See :class:`eoe.datasets.bases.TorchvisionDataset`. Usually this is None.
        @param args: See :class:`torchvision.DatasetFolder`.
        @param kwargs: See :class:`torchvision.DatasetFolder`.
        """
        self.subset = kwargs.pop('subset', False)
        super(DatasetFolder, self).__init__(root, *args, transform=transform, target_transform=target_transform, **kwargs)
        self.logger = logger
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            # splits transform at ToTensor(); apply pre_transform - conditional_transform - post_transform (including ToTensor())
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

        # ------ used cached file paths and other metadata or load and cache them if not available yet
        if self.cached_samples is None:
            assert exclude_imagenet1k, 'Caching always excludes imagenet1k'
            print('Load ImageNet21k meta and cache it...')
            self.classes, self.class_to_idx = self.find_classes(self.root)
            self.samples = self.imgs = self.make_dataset(
                self.root, self.class_to_idx, 
                kwargs.get('extensions', IMG_EXTENSIONS if kwargs.get('is_valid_file', None) is None else None), 
                kwargs.get('is_valid_file', None)
            )
            self.targets = [s[1] for s in self.samples]

            if exclude_imagenet1k:
                imagenet1k_idxs = tuple([self.class_to_idx.get(label) for label in self.imagenet1k_labels])
                self.samples = self.imgs = [s for s in self.samples if s[1] not in imagenet1k_idxs]  # s = ('<path>', idx) pair
                self.targets = [s[1] for s in self.samples]
                for label in self.imagenet1k_labels:
                    try:
                        self.classes.remove(label)
                        del self.class_to_idx[label]
                    except:
                        pass

            ImageNet22K.cached_samples = self.samples
            ImageNet22K.cached_targets = self.targets
            ImageNet22K.cached_classes = self.classes
            ImageNet22K.cached_class_to_idx = self.class_to_idx
            if self.logger is not None:
                size = sys.getsizeof(ImageNet22K.cached_samples) + sys.getsizeof(ImageNet22K.cached_targets)
                size += sys.getsizeof(ImageNet22K.cached_classes) + sys.getsizeof(ImageNet22K.cached_class_to_idx)
                self.logger.logtxt(
                    f"Cache size of {str(type(self)).split('.')[-1][:-2]}'s meta is {size * 1e-9:0.3f} GB"
                )
        else:
            print('Use cached ImageNet21k meta.')
            self.samples = self.imgs = self.cached_samples
            self.targets = self.cached_targets
            self.classes = self.cached_classes
            self.class_to_idx = self.cached_class_to_idx

        self.loader = kwargs.get('loader', default_loader)
        self.extensions = kwargs.get('extensions', None)
        # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

        self.exclude_imagenet1k = exclude_imagenet1k
        self.cached_images = {}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Override the original method of the ImageFolder class to catch some errors.
        For example, it seems like some ImageNet22k images are broken. If this occurs, just sample the next index.
        Further, this implementation supports conditional transforms and shared memory loading.
        Also, returns (img, target, index) instead of (img, target).
        """
        path, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        try:
            if self.load_cached(index) is not None:
                sample = self.load_cached(index)
            else:
                try:  # try to use shared memory if available
                    sample = self._load_image_from_shared_memory(index)
                    # self.logger.logtxt(f'{self.split: >5}: Used shm for {index}', prnt=False)
                except FileNotFoundError:  # Shared memory (cached imagenet) not found, load from disk
                    sample = self.loader(path)
                    # self.logger.logtxt(f'{self.split: >5}: Disk load for {index}', prnt=False)

        except UnidentifiedImageError as e:
            msg = 'ImageNet22k could not load picture at {}. Unidentified image error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]
        except OSError as e:
            msg = 'ImageNet22k could not load picture at {}. OS Error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]
        except sre_constants_error as e:
            msg = 'ImageNet22k could not load picture at {}. SRE Constants Error.'.format(path)
            self.logwarning(msg, e)
            return self[(index + 1) % len(self)]

        if self.transform is not None:
            if self.conditional_transform is not None:
                sample = self.pre_transform(sample)
                sample = self.conditional_transform(sample, target)
                sample = self.post_transform(sample)
            else:
                sample = self.transform(sample)

        return sample, target  # , index

    def cache(self, ids: List[int]):
        self.cached_images = {}
        mem = 0
        procbar = tqdm(ids, desc=f'Caching {len(ids)} resized images for ImageNet22k (current cache size is {mem: >9.4f} GB)')
        for index in procbar:
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
            except UnidentifiedImageError as e:
                continue
            except OSError as e:
                continue
            except sre_constants_error as e:
                continue
            if isinstance(self.pre_transform.transforms[0], transforms.Resize):
                sample = self.pre_transform.transforms[0](sample)
            elif isinstance(self.transform.transforms[0], transforms.Resize):
                sample = self.transform.transforms[0](sample)
            self.cached_images[index] = sample
            mem += np.prod(sample.size) * 3 * 1e-9
            procbar.set_description(f'Caching {len(ids)} resized images for ImageNet22k (current cache size is {mem: >9.4f} GB)')

    def load_cached(self, id: int) -> PIL.Image.Image:
        if id in self.cached_images:
            return self.cached_images[id]
        else:
            return None

    def _load_image_from_shared_memory(self, index: int) -> PIL.Image.Image:
        # see :method:`MyImageNet._load_image_from_shared_memory` for some documentation on this!
        shm = shared_memory.SharedMemory(name=f'{"imagenet21k" if not self.subset else "imagenet21ksubset"}_train_{index}')
        img = decode_shape_and_image(np.copy(np.ndarray(shm.size, dtype=np.uint8, buffer=shm.buf)))
        shm.close()
        unregister(shm._name, 'shared_memory')
        img = to_pil_image(img)
        return img

    def logwarning(self, s, err):
        if self.logger is not None:
            self.logger.warning(s, print=False)
        else:
            raise err


class ADImageNet21kSubSet(ADImageNet21k):
    """
    This uses the :class:`ADImageNet21k` implementation but looks at a different root folder.
    That is, instead of `root`/imagenet22k/fall11_whole_extracted/ it uses the data found in `root`/imagenet21k_subset/.
    """
    base_folder = 'imagenet21k_subset'

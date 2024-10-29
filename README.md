# Enhancing Diversity in Bayesian Deep Learning via Hyperspherical Energy Minimization of CKA
This repository is the official implementation of related to our [paper (todo link)](#) published in NeurIPS 2024.

## Requirements

Setup the environment and install requirements:

```setup
mamba env create --name he-cka-ensemble --file environment.yml
```
To install mamba, a super fast drop in replacement for conda, please read the following [installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). If you prefer to use Anaconda just replace `mamba` with `conda`.

### Datasets
Most datasets will automatically be downloaded to the `data/` folder in the repo directory. 


## Basic Usage
A basic ensemble, and feature tracking, can now be easily implemented with this library. More in-depth examples with 

```python
import torch

from hyper.generators.ensemble import FixedEnsembleModel as Ensemble
from hyper.layers import Conv2d, Linear, SequentialModule, Flatten
from hyper.util.collections import flatten_keys


# construct a simple CNN for MNIST (28x28x1) input
# all modules in hyper.layers support batched weights
lenet = SequentialModule(
    Conv2d(1, 6, 5, act='relu', pooling='max', track='detach'),  # track features but detach from graph
    Conv2d(6, 16, 5, act='relu', pooling='max'),
    Flatten(track=False),  # track=False => do not track feature
    Linear(256, 120, act='relu'),
    Linear(120, 84, act='relu'),
    Linear(84, 10)
)

# create ensemble or equivalently a hypernetwork using MLPLayerModelGenerator in generators.mlp
ens = Ensemble(
    target=lenet,
    ensemble_size=5  # make an ensemble of 5 members
)

# test input
X = torch.randn((8, 1, 28, 28))

# feed through all 5 ensemble members
feat, Y = ens(5, X)
print(f'Output shape {Y.shape}') # output of [5, 8, 10] or [models, batch size, classes]


# flatten keys turns nested ordered dictionaries into a single dictionary with a separator '.' 
for mod_name, mod_feat in flatten_keys(feat).items():
    if mod_feat is not None:  # if not tracking features are None
        print(f'Module {mod_name} shape {mod_feat.shape} variance {mod_feat.var()}')
```


## Training

Training all experiments use the `train.py` script and all the configurations listed in `configs/`. 

Example usage for the mnist ensemble is
```train
python train.py configs/mnist/ens/ensemble.yml
```

Checkpoints and plots are saved to the `outputs/` folder.

Use `--help` for more options such as multiple runs `--runs` or wandb logging `--wandb`.

## Evaluation

To evaluate the models and handle relevant plotting use `eval.py`, which accepts three arguments

```eval
python eval.py <training configuration file> <experiment type: [mnist (only)]> <weight file name>.pt
```

Example usage on an mnist ensemble, evaluating the model checkpoint at epoch 50.

```eval
python eval.py configs/mnist/ens/ensemble.yml mnist model-50.pt
```

### TODO for public code rewrite
- [x] Rewrite hypernet and training modules
- [x] Rewrite Toy experiments
- [x] Rewrite MNIST experiments
-  Rewrite ResNet18/ResNet32 modules and verify consistency with old implementation
-  Rewrite TinyImageNet experiments
-  Rewrite CIFAR10/100 experiments 
-  Include hypernetwork examples/documentation
-  Fix seeding issue with ood examples
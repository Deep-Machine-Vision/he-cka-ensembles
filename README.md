# Enhancing Diversity in Bayesian Deep Learning via Hyperspherical Energy Minimization of CKA
This repository is the official implementation of related to our [paper (todo link)](#) published in NeurIPS 2024.


## Requirements

Setup the environment and install requirements:

```setup
mamba env create --name he-cka-ensemble --file environment.yml
```
To install mamba, a super fast drop in replacement for conda, please read the following [installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). If you prefer to use Anaconda just replace `mamba` with `conda`.

### Datasets
Most datasets will automatically be downloaded to the `data/` folder in the repo directory. Note currently this repository only supports the mnist experiments.


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

### Notes
Investigate seeds, although the seed has been set on each training run the runs have slight differences, not sure why setting the seed did not work, you can still expect results to be very close to paper results.

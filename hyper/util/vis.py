""" Common visualization utilities. """

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from hyper.diversity.uncertainty import entropy, entropy_prob
from hyper.generators.base import ModelGenerator
from hyper.generators.ensemble import FixedEnsembleModel

# default sample of models
MODEL_SAMPLE = 30


def plot_entropy_grid(hyper: ModelGenerator, plot_points: torch.Tensor, extent: float, figname: str=None, figsize: float=5.5, show: bool=False):
    """ Plots entropy grid for a hypernetwork

    Args:
        hyper (ModelGenerator): the hypernetwork to sample from
        plot_points (torch.Tensor): the points to plot
        extent (float): the extent of the plot
        figname (str): the name of the figure
        figsize (float): the size of the figure
        show (bool): whether to show the plot
    """

    plt.rcParams.update({"font.size": 20})

    # visualize entropy grid
    x = torch.linspace(-extent, extent, 280, device='cuda')
    y = torch.linspace(-extent, extent, 280, device='cuda')
    gridx, gridy = torch.meshgrid(x, y)
    grid = torch.stack((gridx.reshape(-1), gridy.reshape(-1)), -1)
    
    # select all in order if fixed ensemble/known size
    if isinstance(hyper, FixedEnsembleModel):
      models = None  
    else:
      models = MODEL_SAMPLE
    
    # sample models from hypernetwork
    hyper.eval()
    _, outputs = hyper(models, grid, sample_params=True)
    
    outputs = torch.nn.functional.softmax(outputs, -1).detach()  # [B, D]

    conf_std = entropy_prob(outputs.mean(0))
    max_entropy = entropy_prob((1.0 / outputs.shape[-1]) * torch.ones((1, outputs.shape[-1]))).cpu().item()
    # print(conf_std.shape)

    # plot and save figure
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sc = ax.scatter(grid[:, 0].cpu(), grid[:, 1].cpu(), c=conf_std.cpu().view_as(grid[:, 0]).numpy(), cmap='YlGnBu', alpha=1.0, vmin=0, vmax=max_entropy, marker='.')  # 1.0)
    ax.scatter(plot_points[:, 0].cpu(), plot_points[:, 1].cpu(), c='black', alpha=0.2, s=15.0, marker='.')
    ax.set_ylim([-extent, extent])
    ax.set_xlim([-extent, extent])
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=ax, fraction=0.045)
    fig.tight_layout()
    
    if figname is not None:
        fig.savefig(figname)
    
    if show:
      fig.show()
    else:
      plt.close(fig)
from typing import List, Union, Tuple, Dict
import torch
import torch.nn as nn
from .base import GeneratedLayerCodeModelGenerator, LayerCodeModelGenerator, register_generator
from ..net import models


@register_generator('mlp_layer_code_generator')
class MLPLayerModelGenerator(GeneratedLayerCodeModelGenerator):
  def __init__(self, latent_size: int, layer_code_generator: LayerCodeModelGenerator, mlp_dims: Union[int, List[int]]=3, dim_multiplier: int=2, bias: bool=False, gamma: Union[str, float, object]=1.0, sn_coeff: float=1.5):
    """ A model generator that uses the layer code generator to generate individual layers of the target model
    along with the basic code mixer MLP. 

    Args:
        latent_size (int): size of input latent to code mixer
        mlp_dims (List[int]|int): if just an int it specifies the number of layers along with a multiplier in mlp_multiplier otherwise it's the dims of the layers to product
        dim_multiplier (int): how much each layer multiplies from the latent_size to the final code layer if mlp_dims is an int
        layer_code_generator (LayerCodeModelGenerator): the layer code model generator to build for
        bias (bool, optional): to use bias in the linear layers of the hyper-network. Defaults to False.
        gamma (see nfnet/activation.py): weight variance scaling gamma.
    """
    # get target model definition as described in model generator
    super(MLPLayerModelGenerator, self).__init__(latent_size, layer_code_generator, code_generator=None)

    # now set the code generator
    self.set_code_generator(
        models.build_groupnorm_mlp(
        in_features=latent_size,
        out_features=layer_code_generator.get_total_codes() * layer_code_generator.code_size,
        mlp_dims=mlp_dims,
        dim_multiplier=dim_multiplier,
        gamma=gamma,
        out_act=None,
        norm_last=True,
        activation=torch.nn.GELU,
        norm_last_groups=layer_code_generator.get_total_layers(),  # norm each code group
        bias=bias,
        coeff=sn_coeff  #3.5
      )
    )

  def sample_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    return torch.randn(size, self.latent_size, device=device)

from typing import List, Union, Tuple, Dict
import torch
import torch.nn as nn
from collections import OrderedDict
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from x_transformers import TransformerWrapper
from x_transformers.x_transformers import AttentionLayers, ScaledSinusoidalEmbedding, AbsolutePositionalEmbedding
from x_transformers import TransformerWrapper, Decoder
from ..layers.linear import Linear
from ..layers.conv import _ConvNd
from ..layers.generators.conv import SampledConvLayerGenerator
from .base import GeneratedLayerCodeModelGenerator, LayerCodeModelGenerator
from ..net import models
from ..net.activation import crater


class BasicTransformerBlock(nn.Module):
  """ First version just kept as reference/comparison """
  def __init__(self, encoder: bool, num_layers: int, d_model: int, dim_feedforward: int, nhead: int, dropout: float, activation: object=crater, bias: bool=True, max_len: int=100):
    super(BasicTransformerBlock, self).__init__()

    self.encoder = encoder
    if encoder:
      self.transformer = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(
          d_model=d_model,
          dim_feedforward=dim_feedforward,
          nhead=nhead,
          dropout=dropout,
          activation=activation,
          layer_norm_eps=1e-5,
          norm_first=False,
          batch_first=False,
          bias=bias
        ),
        num_layers=num_layers
      )
    else:
      self.transformer = nn.TransformerDecoder(
        decoder_layer=nn.TransformerDecoderLayer(    
          d_model=d_model,
          dim_feedforward=dim_feedforward,
          nhead=nhead,
          dropout=dropout,
          activation=activation,
          layer_norm_eps=1e-5,
          norm_first=False,
          batch_first=False,
          bias=bias
        ),
        num_layers=num_layers
      )
    
    self.pos_encoding = Summer(
      PositionalEncoding1D(
        d_model
      )
    )

  def forward(self, x, cross):
    # apply positional encoding
    x_f = self.pos_encoding(x)
    if self.encoder:
      y = self.transformer(x_f)
    else:
      y = self.transformer(x_f, cross)  # include cross attention

    # next layer will get cross and skip first code
    return y.transpose(0, 1)

  def sample_codes(self, model_bs: int, size: int, device=None):
    """ Returns a sample of input to parameters """
    x = torch.zeros(size, model_bs, self.dim, device=device)
    x[0] = torch.normal(0.0, 1.0, size=(model_bs, self.dim), device=None)  # initial code sample
    return x


class XTransformerBlock(nn.Module):
  def __init__(self, encoder: bool, num_layers: int, d_model: int, dim_feedforward: int, nhead: int, dropout: float, activation: object=crater, bias: bool=None, max_len: int=100):
    super(XTransformerBlock, self).__init__()

    self.encoder = encoder
    self.dim = d_model
    self.transformer = AttentionLayers(
      cross_attend=not encoder,  # decoder style cross attention
      only_cross=False,  # include self-attention block
      causal=False,  # non-autoregressive
      heads=nhead,  # number of attention heads
      dim=d_model,  # code size
      depth=num_layers,  # number of attention blocks

      # optimizations
      attn_flash=True,  # enable flash attention (more memory efficient)
      attn_head_scale=True,  # see x_transformers
      # use_scalenorm=True,  # applies scale norm from https://arxiv.org/abs/1910.05895
      use_simple_rmsnorm=True,  # see x_transformers
      # ff_post_act_ln=True,  # ensure good normalization throughout
      ff_glu=True,  # enable gating using GELU
      ff_no_bias=True,  # bias not needed in/doesn't hurt performance according to PaLM
      # residual_attn=True
      # need to test
      # sandwich_coef=6
      # gate_residual=True,
      # scale_residual=True
    )
    
    self.pos_encoding = AbsolutePositionalEmbedding(
      dim=d_model,
      max_seq_len=max_len,
      l2norm_embed=True
    )

  def forward(self, x, cross=None):
    # apply positional encoding
    x = x + self.pos_encoding(x)
    y = self.transformer(x, context=cross)

    # next layer will get cross and skip first code
    return y

  def sample_codes(self, model_bs: int, size: int, device=None):
    """ Returns a sample of input to parameters """
    x = torch.zeros(model_bs, size, self.dim, device=device)
    x[:] = torch.normal(0.0, 1.0, size=(model_bs, size, self.dim), device=None)  # initial code sample
    return x


class TransformerCrossSampledLayerModelGenerator(GeneratedLayerCodeModelGenerator):
  def __init__(self,
      latent_size: int,
      layer_code_generator: LayerCodeModelGenerator,
      num_layers: int=4,
      dim_feedforward: int=128,
      nhead: int=4,
      dropout: float=0.05,
      activation: object=crater,
      bias: bool=True,
      transformer_block=XTransformerBlock,
      gamma: Union[str, float, object]=1.0):
    """ A model generator that uses the layer code generator to generate individual layers of the target model
    along with the "horizontal" cross attention transformer for convolutional sampled filters  

    Args:
        latent_size (int): size of input latent to code mixer
        mlp_dims (List[int]|int): if just an int it specifies the number of layers along with a multiplier in mlp_multiplier otherwise it's the dims of the layers to product
        dim_multiplier (int): how much each layer multiplies from the latent_size to the final code layer if mlp_dims is an int
        layer_code_generator (LayerCodeModelGenerator): the layer code model generator to build for
        bias (bool, optional): to use bias in the linear layers of the hyper-network. Defaults to False.
        gamma (see nfnet/activation.py): weight variance scaling gamma.
    """
    # get target model definition as described in model generator
    super(TransformerCrossSampledLayerModelGenerator, self).__init__(latent_size, layer_code_generator, code_generator=None)

    # we need to iterate through the code layer
    # generators and continually build in order the type of code generators required
    # we take special cases into account such as 
    self.tran_sequence = nn.ModuleList()
    self.code_size = layer_code_generator.code_size
    self.num_codes = []
    self.gen_sequence = []

    # create recursive builder
    def traverse_generators(gen_dict: nn.ModuleDict, gen_seq: nn.Module):
      # traverse through sub-modules (some will include self)
      for key, val in gen_dict.items():
        if isinstance(val, (nn.ModuleDict, OrderedDict, dict)):
          traverse_generators(val, gen_seq)
        else:
          gen_seq.append(val)

    # traverse through the layer code layer generators
    traverse_generators(layer_code_generator.layer_generators, self.gen_sequence)

    # we assume basic architectures like Conv2d -> MLP or Conv2d all throughout
    # @TODO figure out how to do grouping with more complex architectures
    encoder = True  # first layer is encoder
    accounted_for = 0
    agg_codes = 0
    for gen in self.gen_sequence:
      if isinstance(gen, SampledConvLayerGenerator):
        self.tran_sequence.append(
          transformer_block(
            encoder=encoder,
            d_model=self.code_size,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            bias=bias,
            max_len=gen.get_expected_input_size()
          )
        )

        # get expected number of codes
        self.num_codes.append(gen.get_expected_input_size())
        
        # now do cross attention
        encoder = False
        accounted_for += 1
      else:  # @TODO add other special cases here
        agg_codes += gen.get_expected_input_size()
        accounted_for += 1

    # we build linear in groups so identify all linear groups
    self.tran_sequence.append(
      transformer_block(
        encoder=encoder,
        d_model=self.latent_size,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        nhead=nhead,
        dropout=dropout,
        activation=activation,
        bias=bias,
        max_len=agg_codes
      )
    )
    self.num_codes.append(agg_codes)  # aggregated code size

  def sample_codes(self, batch_size: int, device='cuda'):
    """ Creates the list of "starter" codes to sample from """
    codes = []
    for tran, size in zip(self.tran_sequence, self.num_codes):
      codes.append(tran.sample_codes(batch_size, size, device=device))  # each transformer might have a different method

      # batch_query = torch.zeros(size, batch_size, self.latent_size, device=device)
      # batch_query[0, :, :] = torch.randn(batch_size, self.latent_size, device=device)
      # batch_query = torch.normal(0.0, 1.0, size=(size, batch_size, self.latent_size), device=device)
      
      # for i in range(batch_size):
      #   torch.nn.init.orthogonal_(batch_query[i])
      # codes.append(batch_query)
    return codes

  def forward_params(self, x):
    """ Forward input sample of latent_size through the code generator and then the layer codes to generate the parameters """
    
    cross = None  # nothing to cross
    codes = []
    for ind, tran in enumerate(self.tran_sequence):
      current = tran(x[ind], cross)
      codes.append(current)

      # next iteration pass in cross codes
      cross = current
    
    # combine code-wise from transformer outputs
    codes = torch.concat(codes, dim=1)
    return self.forward_layer_params(codes)

  def sample_params(self, size: int, device=None):
    """ Returns a sample of input to parameters """
    return self.sample_codes(size, device)
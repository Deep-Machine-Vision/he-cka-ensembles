import pytest
from torch.testing import assert_close
from ..linear import Linear as BatchedLinear
import torch.nn as nn
import numpy as np
import torch

# @pytest.mark.parametrize("device", ("cpu", "cuda"))
class TestLinear:
  def prep_lin(self):
    conv = BatchedLinear(
      in_features=10,
      out_features=20,
      bias=True
    )
    return conv

  def get_weights(self, lin: BatchedLinear):
    """ Create test weights """
    O = lin.out_features
    I = lin.in_features

    # special test half output
    spec = -1.0 * torch.ones(O, I)
    spec[:(O//2)] = 0.0

    return [
      torch.zeros(O, I),
      torch.ones(O, I),
      torch.normal(0.0, 1.0, size=(O, I)),
      spec
    ]
  
  def get_biases(self, lin: BatchedLinear):
    """ Create test biases """
    O = lin.out_features

    # special test half output
    spec = -1.0 * torch.ones(O)
    spec[:(O//2)] = 0.0

    return [
      torch.zeros(O),
      torch.normal(0.0, 1.0, size=(O,)),
      spec
    ]

  def get_examples(self, in_feat: int=10):
    """ Create test input examples """
    # create batches of examples
    examples = []

    # create various features to test against
    spec = -1.0 * torch.ones(in_feat,)
    spec[:(in_feat//2)] = 0.0

    # yes random would work fine, but had some weird behaviour before
    # just trying to make sure it works every time
    examples.append(
      [
        torch.zeros(in_feat,),
        torch.normal(0.0, 1.0, size=(in_feat,)),
        spec
      ]
    )
    return examples

  def make_param(self, lin: BatchedLinear, weights, bias):
    """ Make parameters for the batched linear """
    
    # stack if they are lists
    if isinstance(weights, list):
      weights = torch.stack(weights)
    
    if bias is not None and isinstance(bias, list):
      bias = torch.stack(bias)

    print('stacked shape', weights.shape)

    B = weights.shape[0]  # model batch size
    params = weights.view(B, lin.out_features*lin.in_features)

    if lin.bias:
      params = torch.concat([params, bias.view(B, lin.out_features)], dim=1).contiguous()  # combine along model weights
    
    return {
      'self': params
    }

  def test_batched(self):
    example_sizes = [torch.stack(ex) for ex in self.get_examples()]
    linb = self.prep_lin()
    weights = self.get_weights(linb)
    biases = self.get_biases(linb)

    with torch.no_grad():
      # test different sizes
      for index, examples in enumerate(example_sizes):
        print(f'Testing example examples {index + 1}')
        # test various cases
        for bias in biases:
          # construct outputs via normal conv2d
          lin_outs = []
          for weight in weights:
            lin = nn.Linear(in_features=10, out_features=20, bias=True).cuda()
            lin.weight.data = weight.cuda()
            lin.bias.data = bias.cuda()
            lin_outs.append(lin(examples.cuda()))

          # stack outputs
          lin_outs = torch.stack(lin_outs).cuda()     

          # now run with batched version
          params = self.make_param(linb, [w.cuda() for w in weights], [bias.cuda() for _ in weights])

          # assert reshaping is correct weights (see conv.py)
          at_ind = lin.out_features
          print('Params length', params['self'].shape, 'weight length', params['self'][:, :-at_ind].shape, 'bias length', params['self'][:, -at_ind:].shape)
          weight_test = params['self'][:, :-at_ind].reshape(len(weights), lin.out_features, lin.in_features)
          if lin.bias is not None:
            bias_test = params['self'][:, -at_ind:].reshape(len(weights), lin.out_features)
          assert_close(
            actual=weight_test,
            expected=torch.stack(weights).cuda(),
            msg='Unpacked weights incorrect'
          )
          assert_close(
            actual=bias_test,
            expected=torch.stack([bias for _ in weights]).cuda(),
            msg='Unpacked bias incorrect'
          )

          # disable variancescaling for now
          linb.scale = 1.0
          linb.invsq = 1.0

          _, linb_outs = linb(params, examples.cuda())

          # conv2d batched is N then B
          linb_outs = linb_outs.contiguous()
          # print('C2DB spec\n', conv2db_outs[:, 2].view(4, 5), conv2db_outs[:, 2].shape)

          # assert they're equal
          assert_close(
            actual=linb_outs,
            expected=lin_outs,
            msg='Batched linear output missmatch'
          )

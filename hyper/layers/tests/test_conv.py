import pytest
from torch.testing import assert_close
from ..conv import Conv2d as BatchedConv2d, ScaledWConv2d
import torch.nn as nn
import numpy as np
import torch

# @pytest.mark.parametrize("device", ("cpu", "cuda"))
class TestConv:
  def prep_conv(self, groups: int=1, scaled: bool=False):
    conv = (ScaledWConv2d if scaled else BatchedConv2d)(
      in_channels=4,
      out_channels=5,
      kernel_size=3,
      padding=0,
      stride=1,
      groups=groups
    )
    return conv

  def get_weights(self, conv: BatchedConv2d):
    """ Create test weights """
    O = conv.out_channels
    I = conv.in_channels // conv.groups
    H = conv.kernel_size[0]
    W = conv.kernel_size[1]

    # special test half output
    spec = -1.0 * torch.ones(O, I, H, W)
    spec[:(O//2)] = 0.0

    return [
      torch.zeros(O, I, H, W),
      torch.ones(O, I, H, W),
      torch.normal(0.0, 1.0, size=(O, I, H, W)),
      spec
    ]
  
  def get_biases(self, conv: BatchedConv2d):
    """ Create test biases """
    O = conv.out_channels

    # special test half output
    spec = -1.0 * torch.ones(O)
    spec[:(O//2)] = 0.0

    return [
      torch.zeros(O),
      torch.normal(0.0, 1.0, size=(O,)),
      spec
    ]

  def get_examples(self, chan: int=4):
    """ Create test input examples """
    # create batches of examples
    w, h = 3, 3
    examples = []
    for scale in range(1, 5):
      # create various features to test against
      w_s, h_s = w*scale, h*scale
      spec = -1.0 * torch.ones(chan, w_s, h_s)
      spec[:(chan//2)] = 0.0

      # yes random would work fine, but had some weird behaviour before
      # just trying to make sure it works every time
      examples.append(
        [
          torch.zeros(chan, w_s, h_s),
          torch.normal(0.0, 1.0, size=(chan, w_s, h_s)),
          spec
        ]
      )
    return examples

  def make_param(self, conv: BatchedConv2d, weights, bias):
    """ Make parameters for the batched conv 2d """
    
    # stack if they are lists
    if isinstance(weights, list):
      weights = torch.stack(weights)
    
    if bias is not None and isinstance(bias, list):
      bias = torch.stack(bias)

    print('stacked shape', weights.shape)

    B = weights.shape[0]  # model batch size
    in_chan = conv.in_channels // conv.groups
    num_filter = in_chan * np.prod(conv.kernel_size)  # parameters in a single filter
    filters = weights.view(B, conv.out_channels*num_filter)

    print('filter shape', filters.shape)

    if conv.bias:
      filters = torch.concat([filters, bias.view(B, conv.out_channels)], dim=1).contiguous()  # combine along model weights
    
    return {
      'self': filters
    }

  @torch.no_grad()
  def test_variance(self):
    """ Tests to ensure proper weight variance scaling is correct for ScaledWConv2d """
    convb = self.prep_conv(scaled=True).cuda()

    # pass in various variance examples
    batch = 15
    in_chan = 4
    w, h = 30, 30

    O = convb.out_channels
    I = convb.in_channels // convb.groups
    H = convb.kernel_size[0]
    W = convb.kernel_size[1]

    # run through both
    for ind, examples in enumerate([
      torch.normal(0.0, 1.0, size=(batch, in_chan, w, h)),
      torch.normal(0.0, 0.5, size=(batch, in_chan, w, h))
    ]):
      print(f'Example index {ind}')

      # prepare unit and "non-unit" weights to be expected to be normalized
      # for two models
      unit = [torch.normal(0.0, 1.0, size=(O, I, H, W)).cuda()]
      bias = [torch.zeros(size=(O,)).cuda()]
      nonunit = [torch.normal(0.5, 0.3, size=(O, I, H, W)).cuda()]
      examples = examples.cuda()

      # create params/feed through
      params_unit = self.make_param(convb, unit, bias)
      params_nonunit = self.make_param(convb, nonunit, bias)
      _, out_unit = convb(params_unit, examples)
      _, out_nonunit = convb(params_nonunit, examples)

      # outs should be [model bs, image bs, out chan, height, width]
      # we want variance (get average) of each out pixel
      def calc_var(out):
        var, mean = torch.var_mean(out, dim=(3, 4))
        return mean.mean(), var.mean()  # get average mean and variance across spatial

      # calculate
      mean_unit, var_unit = calc_var(out_unit)
      mean_nonunit, var_nonunit = calc_var(out_nonunit) 

      # ensure weight normalization method matches that of known method
      weight = nonunit[0]
      mean = torch.mean(weight, dim=[1, 2, 3], keepdim=True)
      var = torch.var(weight, dim=[1, 2, 3], keepdim=True, unbiased=False) 
      weight = (weight - mean) * torch.rsqrt(torch.maximum(var, torch.tensor(convb.eps).to(var.device))) * convb.scale
      
      # use calculated conv2d batched weight
      calc_weight = convb._param_scale(convb.from_flat(params_nonunit['self']))['weight'][0]

      print('Expected weight (first filter)\n', weight[0], weight.shape)
      print('Got weight (first filter)\n', calc_weight[0], calc_weight.shape)
      assert_close(
        actual=calc_weight,  # get first model weight
        expected=weight,
        atol=1e-3,
        rtol=0.0,
        msg='Incorrect weight normalization'
      )

      # ensure variance
      print('Unit', mean_unit, var_unit)
      print('Nonunit', mean_nonunit, var_nonunit)
      assert torch.abs(mean_unit) < 0.1 and torch.abs(var_unit - 1.0) < 0.3, 'Unit weight variance off'
      assert torch.abs(mean_nonunit) < 0.1 and torch.abs(var_nonunit - 1.0) < 0.3, 'Non-unit weight variance off'
      assert False

  @torch.no_grad()
  def test_batched(self):
    """ Tests to make sure batched version of conv results in same output as individual convs """
    example_sizes = [torch.stack(ex) for ex in self.get_examples()]
    convb = self.prep_conv()
    weights = self.get_weights(convb)
    biases = self.get_biases(convb)
  
    # test different sizes
    for index, examples in enumerate(example_sizes):
      print(f'Testing example examples {index + 1}')
      # test various cases
      for bias in biases:
        # construct outputs via normal conv2d
        conv2d_outs = []
        for weight in weights:
          c2d = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=3, padding=0, stride=1).cuda()
          c2d.weight.data = weight.cuda()
          c2d.bias.data = bias.cuda()
          conv2d_outs.append(c2d(examples.cuda()))

        # stack outputs
        conv2d_outs = torch.stack(conv2d_outs).cuda()     

        # now run with batched version
        params = self.make_param(convb, [w.cuda() for w in weights], [bias.cuda() for _ in weights])

        # assert reshaping is correct weights (see conv.py)
        at_ind = convb.out_channels
        print('Params length', params['self'].shape, 'weight length', params['self'][:, :-at_ind].shape, 'bias length', params['self'][:, -at_ind:].shape)
        weight_test = params['self'][:, :-at_ind].reshape(len(weights), convb.out_channels, convb.in_channels // convb.groups, *convb.kernel_size)
        if convb.bias:
          bias_test = params['self'][:, -at_ind:].reshape(len(weights), convb.out_channels)
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
        convb.scale = 1.0
        convb.invsq = 1.0

        _, conv2db_outs = convb(params, examples.cuda())
        print('Inputs\n', examples[2], '\nLast weight weight\n', weight_test[len(weights) - 1, 3], weight_test[len(weights) - 1, 3].shape)
        # print('C2D spec\n', conv2d_outs[:, 2].view(4, 5), conv2d_outs[:, 2].shape)

        # conv2d batched is N then B
        conv2db_outs = conv2db_outs.contiguous()
        # print('C2DB spec\n', conv2db_outs[:, 2].view(4, 5), conv2db_outs[:, 2].shape)

        # assert they're equal
        assert_close(
          actual=conv2db_outs,
          expected=conv2d_outs,
          msg='Batched conv2d output missmatch'
        )

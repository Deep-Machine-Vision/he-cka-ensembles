from hyper.layers.conv import Conv2d, ScaledWConv2d, FinalConv2d, FinalScaledWConv2
from hyper.layers.linear import Linear, FinalLinear
from hyper.generators.base import LayerCodeModelGenerator
from hyper.layers.generators.conv import MLPSampledConvLayerGenerator
from hyper.layers.generators.base import MLPLayerGenerator
from hyper.layers.pool import AvgPool2d
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module, register_gen_module


@register_gen_module('lenet5')
class LeNet(GenModule):
  def __init__(self, in_channels: int=3, num_classes: int=10, bias: bool=True, nfnet: bool=False, flatten_dims: int=256, activation: str='crater', param_scale='gamma', gamma: float=1.0, track: bool = True):
    """ Create a batchable LeNet classification model

    Args:
        in_channels (int, optional): number of input Conv2d channels. Defaults to 3.
        num_classes (int, optional): number of output classes. Defaults to 10.
        bias (bool, optional): use bias in the layers or not. Defaults to True.
        nfnet (bool, optional): use scaled weight Conv2d layers. Defaults to False.
        flatten_dims (int, optional): input features to first Linear after flattening Conv2d layers. See lenet.py if confused. Defaults to 256.
        activation (str, optional): name of activation to use. See net/activation.py for options. Defaults to 'crater'.
        gamma (float, optional): input gamma adjustment for expected variance. Usually normalized data has gamma=1.0. Defaults to 1.0.
        track (bool, optional): whether or not to track the internal features of the modules. Defaults to True.
    """
    super().__init__(track)
    
    # specify sequence
    # SequentialModule makes it easy to use the sequence as the define_generated_modules function and not specify it manually
    conv_mod = ScaledWConv2d if nfnet else Conv2d
    self.sequence = SequentialModule(
      conv_mod(in_channels, 6, 5, pooling='max', act=activation, gamma=gamma, bias=bias, param_scale=param_scale),
      conv_mod(6, 16, 5, pooling='max', act=activation, gamma=activation, bias=bias, param_scale=param_scale),
      Flatten(track=False),  # don't track flatten
      Linear(flatten_dims, 120, bias=bias, act=activation, gamma=activation, param_scale=param_scale),
      Linear(120, 84, bias=bias, act=activation, gamma=activation, param_scale=param_scale),
      FinalLinear(84, num_classes, bias=bias, act=None, gamma=activation, param_scale=param_scale),
      track=track
    )
  
  def is_generated(self):
    """ No generated modules for 'self' all parameters will be passed to self.sequence.
    
    This function might look confusing but all it's telling the hypernetwork is that we don't need
    to allocate parameter to this module itself (so no layer codes), but this module still CAN have generated sub-modules even when is_generated returns False.
    
    Those sub-modules are defined in "define_generated_modules," where if is_generated returns True you set the shape to be generated in the 'self' key.
    """
    return False
  
  def define_generated_modules(self):
    """ Return the sequence definitions """
    return self.sequence.define_generated_modules()
  
  def forward(self, params, x):
    """ Just pass through sequence """
    return self.sequence(params, x)


""" HYPERNETWORK SECTION """
# definition of the layer generators/sizes of layer code generators

""" Batched Normalization layers based off of https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py """
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


from hyper.layers.module import ParametersGroup
from hyper.layers.module import GenModule


class _NormAffine(ParametersGroup):
  def __init__(self, num_features):
    super(_NormAffine, self).__init__(
      shapes=[
        ('weight', (num_features,)),
        ('bias', (num_features,))
      ]
    )

  def gen_initialized_params(self, dtype=None, device=None, gain=1):
    """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
    params = self.gen_empty_params(dtype=dtype, device=device, requires_grad=True)
    nn.init.ones_(params['weight'])
    nn.init.zeros_(params['bias'])
    return params


class _NormBuffers(ParametersGroup):
  def __init__(self, num_features):
    super(_NormBuffers, self).__init__(
      shapes=[
        ('mean', (num_features,)),
        ('var', (num_features,)),
        ('num_batches_tracked', (1,))  # right now a float :(
      ]
    )

  def gen_initialized_params(self, dtype=None, device=None, gain=1):
    """ By default we generate parameters but sometimes it's useful to generate an initialized version, ie for an ensemble, of the parameters  """
    params = self.gen_empty_params(dtype=dtype, device=device, requires_grad=False)
    nn.init.zeros_(params['mean'])
    nn.init.ones_(params['var'])
    nn.init.zeros_(params['num_batches_tracked'])
    return params


class _NormBase(GenModule):
  """Common base of _InstanceNorm and _BatchNorm."""

  _version = 2
  __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
  num_features: int
  eps: float
  momentum: Optional[float]
  affine: bool
  track_running_stats: bool

  def __init__(
    self,
    num_features: int,
    eps: float = 1e-5,
    momentum: Optional[float] = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
    track: bool=True
  ) -> None:
    super().__init__(track=track)
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum

    self.affine = affine
    self.track_running_stats = track_running_stats
    self._tracked_params = None
    if self.affine:
      self.affine_module = _NormAffine(num_features)
    else:
      self.affine_module = None
    
    if self.track_running_stats:
      self.buffer_module = _NormBuffers(num_features)
    else:
      self.buffer_module = None

  def is_generated(self):
    return False
  
  def define_generated_modules(self):
    mod = super().define_generated_modules()
    
    if self.affine:
      mod['affine'] = self.affine_module.define_generated_modules()
    
    # not going to be supported with hypernetworks
    # only fixed ensembles
    if self.track_running_stats:
      mod['norm_buffer'] = self.buffer_module.define_generated_modules()
    
    return mod
  
  def reset_running_stats(self) -> None:
    if self.track_running_stats and self._tracked_params is not None:
      if 'mean' in self._tracked_params:
        for m in self._tracked_params['mean']:
          m.zero_()
        for v in self._tracked_params['var']:
          v.zero_()
        for n in self._tracked_params['num_batches_tracked']:
          n.zero_()

  def reset_parameters(self) -> None:
    self.reset_running_stats()
    if self.affine and self._tracked_params is not None:
      if 'weight' in self._tracked_params:
        self._tracked_params['weight'].ones_()
        self._tracked_params['bias'].zero_()
  
  def _check_input_dim(self, input):
    raise NotImplementedError

  def extra_repr(self):
    return (
      "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
      "track_running_stats={track_running_stats}".format(**self.__dict__)
    )


class _BatchNorm(_NormBase):
  def __init__(
    self,
    num_features: int,
    eps: float = 1e-5,
    momentum: Optional[float] = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
    track: bool=True
  ) -> None:
    super().__init__(
      num_features, eps, momentum, affine, track_running_stats, track=track
    )

  def forward(self, params, input):
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = [0.0] * input.shape[0]
    else:
      exponential_average_factor = [self.momentum] * input.shape[0]

    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
      bn_training = True
    else:
      if not self.track_running_stats:
        print('WARNING: unknown/undesirable behaviours with no running stats/wont work here')
        # bn_training = (self.running_mean is None) and (self.running_var is None)
        bn_training = True
      else:
        bn_training = False  # in eval mode with previous stats already defined

    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    
    weight = None
    bias = None
    if self.affine:
      viewed = self.affine_module.from_flat(params['affine']['self'])  # unflatten parameters
      weight = viewed['weight']
      bias = viewed['bias']
      
      if self._tracked_params is None:
        self._tracked_params = {}
      self._tracked_params['weight'] = weight
      self._tracked_params['bias'] = bias
    
    running_mean = [None] * input.shape[0]
    running_var = [None] * input.shape[0]
    num_batches_tracked = None
    if self.track_running_stats:
      running_mean = params['norm_buffer']['self']['mean']
      running_var = params['norm_buffer']['self']['var']
      num_batches_tracked = params['norm_buffer']['self']['num_batches_tracked']
      
      if self._tracked_params is None:
        self._tracked_params = {}
      self._tracked_params['mean'] = running_mean
      self._tracked_params['var'] = running_var
      self._tracked_params['num_batches_tracked'] = num_batches_tracked
    
    if self.training and self.track_running_stats:
      if self.momentum is None:  # use cumulative moving average
        if num_batches_tracked is None:
          raise RuntimeError('Cannot apply exponential average factor')
        
        exponential_average_factor = [1.0 / float(num_batches_tracked[ind]) for ind in range(input.shape[0])]
      else:  # use exponential moving average
        exponential_average_factor = [self.momentum] * input.shape[0]
    # given batchnorm needs direct access to buffer we have to loop over fixed parameter vals
    res = torch.stack(
       [
        F.batch_norm(
          input[ind],
          running_mean[ind] if not self.training or self.track_running_stats else None,
          running_var[ind] if not self.training or self.track_running_stats else None,
          weight[ind] if weight is not None else None,
          bias[ind] if bias is not None else None,
          bn_training,
          exponential_average_factor[ind],  # could be different depending on model sampling
          self.eps,
        ) for ind in range(input.shape[0])  # iter over model batch dim
      ] 
    )
    
    return self.track_feature(res), res


class BatchNorm1d(_BatchNorm):
    r"""Applies a Model Batched Batch Normalization over a 2D or 3D input.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.
    At train time in the forward pass, the standard-deviation is calculated via the biased estimator,
    equivalent to ``torch.var(input, unbiased=False)``. However, the value stored in the
    moving average of the standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(M, N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(M, N, C)` or :math:`(M, N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(M, N, C)` or :math:`(M, N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(3, 20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(f"expected 3D or 4D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    r"""Applies a Model Batched Batch Normalization over a 5D input.

    5D is a model batch of a mini-batch of 2D inputs
    with additional channel dimension. Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(M, N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(M, N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(M, N, C, H, W)`
        - Output: :math:`(M, N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(3, 20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")



class BatchNorm3d(_BatchNorm):
    r"""Applies Batch Normalization over a 6D input.

    6D is a model batch of mini-batch of 3D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased  estimator, equivalent to
    ``torch.var(input, unbiased=True)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(M, N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(M, N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(M, N, C, D, H, W)`
        - Output: :math:`(M, N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(3, 20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError(f"expected 6D input (got {input.dim()}D input)")

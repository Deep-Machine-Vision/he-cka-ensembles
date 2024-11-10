from hyper.experiments.metrics import Callback
from hyper.data.fast_cifar import get_test_loader
from hyper.data import load_svnh_test
from hyper.diversity.uncertainty import get_eval_stats_ensemble
from hyper.experiments.training import HyperTrainer
from hyper.target.testing.resnetens import ResNetEnsemble

import torch
import torch.utils
import traceback


def resnet18_ensemble():
  """ Used for testing old implementation. See test/ensemble.yml file for more info"""
  return ResNetEnsemble(
    size=5,
    classes=10
  ).cuda()


class CIFARTestCallback(Callback):
  """ Handles logging of test metrics at the end of some epochs """
  def __init__(self, report=2):
    super(CIFARTestCallback, self).__init__()
    self.report = report
  
  def epoch_end(self, epoch: int, trainer: HyperTrainer, model_bs, X, Y, forward_data=None, *args, **kwargs):
    """ Log test metrics every few epochs """
    if epoch % self.report == 0:
      hyper = trainer.hyper
      
      # sample the models
      sparam = hyper.sample_params(model_bs, device=trainer.device_id)
      params = hyper.forward_params(sparam)
      
      # get auroc/ece/etc
      try:
        test_loader = get_test_loader(128)
        ood_loader = load_svnh_test(bs=128, drop_last=False)
        nlls, nlls_dang, accuracy, ece, ece_30, auroc_mi, auprc_mi, auroc_pe, auprc_pe, auroc_conf, auprc_conf = get_eval_stats_ensemble(
          hyper=hyper,
          params=params,
          test_loader=test_loader,
          ood_test_loader=ood_loader,
          device=trainer.device_id
        )
      except ValueError as e:
        traceback.print_exc()
        print('Failed to evaluate model', str(e))
      
      # log the metrics
      trainer.log({
        'nll': nlls.cpu().item(),
        'nll_dang': nlls_dang.cpu().item(),
        'accuracy/test': accuracy,
        'ece/test': ece,
        'auroc_mi/test': auroc_mi,
        'auprc_mi/test': auprc_mi,
        'auroc_pe/test': auroc_pe,
        'auprc_pe/test': auprc_pe,
        'auroc_conf/test': auroc_conf,
        'auprc_conf/test': auprc_conf,
        'performance/test': (0.9*(accuracy + ((auroc_pe + auroc_mi) / 2.0)) / 2.0) + 0.1*(1.0 - ece),  # some balanced metric/measure of overall performance of model
      })

      

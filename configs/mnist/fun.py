from hyper.experiments.metrics import Callback
from hyper.data.dirty_mnist import get_test_loader as dirty_mnist_loader
from hyper.data.fashion_mnist import get_loaders as fmnist_loader
from hyper.diversity.uncertainty import get_eval_stats_ensemble
from hyper.experiments.training import HyperTrainer
import traceback



class MNISTTestCallback(Callback):
  """ Handles logging of test metrics at the end of some epochs """
  def __init__(self):
    super(MNISTTestCallback, self).__init__()
  
  def epoch_end(self, epoch: int, trainer: HyperTrainer, model_bs, X, Y, forward_data=None, *args, **kwargs):
    """ Log test metrics every few epochs """
    if epoch % 2 == 0:
      hyper = trainer.hyper
      
      # sample the models
      sparam = hyper.sample_params(model_bs, device=trainer.device_id)
      params = hyper.forward_params(sparam)
      
      # get auroc/ece/etc
      try:
        test_loader = dirty_mnist_loader(64)
        ood_loader = fmnist_loader(64, train=False)
        
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
      

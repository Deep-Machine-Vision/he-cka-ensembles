"""
Modified from: https://github.com/omegafragger/DDU

Metrics measuring either uncertainty or confidence of a model.
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

from hyper.util.calibration import expected_calibration_error


def entropy(logits):
  """ Compute the entropy of a set of logits """
  p = F.softmax(logits, dim=1)
  logp = F.log_softmax(logits, dim=1)
  plogp = p * logp
  entropy = -torch.sum(plogp, dim=1)
  return entropy


def logsumexp(logits):
  """ Compute the logsumexp of a set of logits """
  return torch.logsumexp(logits, dim=1, keepdim=False)


def confidence(logits):
  """ Compute the confidence of a set of logits """
  p = F.softmax(logits, dim=1)
  confidence, _ = torch.max(p, dim=1)
  return confidence


def entropy_prob(probs, dim=1):
  """ Compute the entropy of a set of probabilities """
  p = probs
  eps = 1e-12
  logp = torch.log(p + eps)
  plogp = p * logp
  entropy = -torch.sum(plogp, dim=dim)
  return entropy


def mutual_information_prob(probs):
  """ Compute the mutual information of a set of probabilities """
  mean_output = torch.mean(probs, dim=0)
  predictive_entropy = entropy_prob(mean_output)

  # Computing expectation of entropies
  p = probs
  eps = 1e-12
  logp = torch.log(p + eps)
  plogp = p * logp
  exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

  # Computing mutual information
  mi = predictive_entropy - exp_entropies
  return mi


# Taken from Deep Deterministic Uncertainty repository and modified to work with our codebase
# Utility functions to get OOD detection ROC curves and AUROC scores
# Ideally should be agnostic of model architectures

def get_logits_labels(hyper, mbs, data_loader, device):
    """
    Utility function to get logits and labels.
    """
    hyper.eval()
    logits = []
    labels = []
    
    # sample some models
    sparam = hyper.sample_params(mbs, device=device)
    params = hyper.forward_params(sparam)
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            _, logit = hyper.forward(params, data)
            logits.append(logit)
            labels.append(label)
    logits = torch.cat(logits, dim=1)
    labels = torch.cat(labels, dim=1)
    return logits, labels


def get_roc_auc(net, test_loader, ood_test_loader, uncertainty, device, confidence=False):
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)

    return get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=confidence)


def get_roc_auc_logits(logits, ood_logits, uncertainty, device, confidence=False):
    uncertainties = uncertainty(logits)
    ood_uncertainties = uncertainty(ood_logits)

    # In-distribution
    bin_labels = torch.zeros(uncertainties.shape[0]).to(device)
    in_scores = uncertainties

    # OOD
    bin_labels = torch.cat((bin_labels, torch.ones(ood_uncertainties.shape[0]).to(device)))

    if confidence:
        bin_labels = 1 - bin_labels
    ood_scores = ood_uncertainties  # entropy(ood_logits)
    scores = torch.cat((in_scores, ood_scores))

    fpr, tpr, thresholds = metrics.roc_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auroc = metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return (fpr, tpr, thresholds), (precision, recall, prc_thresholds), auroc, auprc



def ensemble_forward_pass(hyper, params, data):
    """
    Single forward pass in a given ensemble providing softmax distribution,
    predictive entropy and mutual information.
    """
    _, pred = hyper.forward(params, data)
    outputs = F.softmax(pred, dim=-1)
    mean_output = outputs.mean(0)
    
    predictive_entropy = entropy_prob(mean_output)
    mut_info = mutual_information_prob(outputs)
    confidence = mean_output.max(dim=-1)[0]   # pred[0].max(dim=-1)[0] #outputs.max(dim=-1)[0].mean(0)

    return pred, mean_output, predictive_entropy, mut_info, confidence


def get_roc_auc_ensemble(hyper, params, test_loader, ood_test_loader, uncertainty, device):
    bin_labels_uncertainties = None
    uncertainties = None

    # sample some models
    hyper.eval()

    bin_labels_uncertainties = []
    rev_labels_uncertainties = []
    uncertainties = []
    confidences = []
    conf_in = []
    conf_out = []
    with torch.no_grad():
        # sparam = hyper.sample_params(mbs, device=device)
        # params = hyper.forward_params(sparam)
        
        # Getting uncertainties for in-distribution data
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.zeros(label.shape).to(device)
            rev_label_uncertainty = torch.ones(label.shape).to(device)
            if uncertainty == "mutual_information":
                outputs, net_output, _, unc, conf = ensemble_forward_pass(hyper, params, data)
            else:
                outputs, net_output, unc, _, conf = ensemble_forward_pass(hyper, params, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            rev_labels_uncertainties.append(rev_label_uncertainty)
            uncertainties.append(unc)
            confidences.append(conf)
            conf_in.append(conf)

        # Getting entropies for OOD data
        for data, label in ood_test_loader:
            data = data.to(device)
            label = label.to(device)

            bin_label_uncertainty = torch.ones(label.shape).to(device)
            rev_label_uncertainty = torch.zeros(label.shape).to(device)
            if uncertainty == "mutual_information":
                outputs, net_output, _, unc, conf = ensemble_forward_pass(hyper, params, data)
            else:
                outputs, net_output, unc, _, conf = ensemble_forward_pass(hyper, params, data)

            bin_labels_uncertainties.append(bin_label_uncertainty)
            rev_labels_uncertainties.append(rev_label_uncertainty)
            uncertainties.append(unc)
            confidences.append(conf)
            conf_out.append(conf)

        bin_labels_uncertainties = torch.cat(bin_labels_uncertainties)
        rev_labels_uncertainties = torch.cat(rev_labels_uncertainties)
        uncertainties = torch.cat(uncertainties)
        confidences = torch.cat(confidences)

    # print('CONF')
    plt.hist(torch.cat(conf_in).cpu().numpy(), bins=50, density=True, label='In dist')
    plt.hist(torch.cat(conf_out).cpu().numpy(), bins=50, density=True, label='Out dist', alpha=0.7)
    plt.legend()
    plt.savefig('res.png')
    plt.close()

    # do auroc for confidence levels
    fpr, tpr, roc_thresholds = metrics.roc_curve(rev_labels_uncertainties.cpu().numpy(), confidences.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        rev_labels_uncertainties.cpu().numpy(), confidences.cpu().numpy()
    )
    auroc_conf = metrics.roc_auc_score(bin_labels_uncertainties.cpu().numpy(), confidences.cpu().numpy())
    auprc_conf = metrics.average_precision_score(bin_labels_uncertainties.cpu().numpy(), confidences.cpu().numpy())

    # do for entropy/mutual info
    fpr, tpr, roc_thresholds = metrics.roc_curve(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    precision, recall, prc_thresholds = metrics.precision_recall_curve(
        bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy()
    )
    auroc = metrics.roc_auc_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())
    auprc = metrics.average_precision_score(bin_labels_uncertainties.cpu().numpy(), uncertainties.cpu().numpy())

    # print('CONF AUROC', auroc_conf)
    # import time
    # time.sleep(0.5)

    return (fpr, tpr, roc_thresholds), (precision, recall, prc_thresholds), auroc, auprc, auroc_conf, auprc_conf


def test_classification_net_softmax(softmax_prob, labels):
    """
    This function reports classification accuracy and confusion matrix given softmax vectors and
    labels from a model.
    """
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    
    confidence_vals, predictions = torch.max(softmax_prob, dim=1)
    labels_list.extend(labels.cpu().numpy())
    predictions_list.extend(predictions.cpu().numpy())
    confidence_vals_list.extend(confidence_vals.cpu().numpy())
    accuracy = accuracy_score(labels_list, predictions_list)
    return (
        confusion_matrix(labels_list, predictions_list),
        accuracy,
        labels_list,
        predictions_list,
        confidence_vals_list,
    )


def test_classification_net_ensemble(hyper, params, data_loader, device):
    """
    This function reports classification accuracy and confusion matrix over a dataset
    for a deep ensemble.
    """
    # for model in model_ensemble:
    #     model.eval()
    softmax_prob = []
    all_out = []
    labels = []
    hyper.eval()
    nlls = []
    with torch.no_grad():
        # sparam = hyper.sample_params(mbs, device=device)
        # params = hyper.forward_params(sparam)
        
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            outputs, softmax, _, _, conf = ensemble_forward_pass(hyper, params, data)
            all_out.append(outputs)
            softmax_prob.append(softmax)
            
            # calculate nll
            ces = []
            for i in range(outputs.shape[0]):
                ces.append(F.cross_entropy(
                    outputs[i],
                    label,
                    reduction='none'
                ))
            ces = -torch.stack(ces)
            # impl modified from https://github.com/AaltoPML/FoRDE/blob/main/train_forde.py
            bnll = torch.logsumexp(ces, dim=0) - torch.log(torch.tensor(ces.shape[0], dtype=torch.float32, device=ces.device))
            labels.append(label)
            nlls.append(bnll)
    softmax_prob = torch.cat(softmax_prob, dim=0)
    labels = torch.cat(labels, dim=0)
    nlls = torch.cat(nlls, dim=0)
    
    # D'Ang nll (i think they have a bunch of issues with their code and impl)
    # their code is just wrong... but I have to have a 1-to-1 comparison with their paper/approach
    Y_t = F.softmax(torch.cat(all_out, dim=1), dim=-1).mean(0)
    nlls_dang = -torch.log((F.one_hot(labels).expand_as(Y_t) * (Y_t)).max(1)[0].mean(0))
    
    return nlls, nlls_dang, test_classification_net_softmax(softmax_prob, labels)


def get_eval_stats_ensemble(hyper, params, test_loader, ood_test_loader, device):
    """
    Util method for getting evaluation measures taken during training time for an ensemble.
    """
    # (conf_matrix, accuracy, labels, predictions, confidences,) = test_classification_net_ensemble(
    #     hyper, mbs, test_loader, device
    # )
    # ece = expected_calibration_error(confidences, predictions, labels, num_bins=15)
    # (_, _, _), (_, _, _), auroc, auprc = get_roc_auc_ensemble(
    #     hyper, mbs, test_loader, ood_test_loader, entropy, device
    # )
    
    nlls, nlls_dang, (conf_matrix, accuracy, labels_list, predictions, confidences,) = test_classification_net_ensemble(
                hyper, params, test_loader, device
    )

    ece = expected_calibration_error(confidences, predictions, labels_list, num_bins=15)
    ece_30 = expected_calibration_error(confidences, predictions, labels_list, num_bins=30)

    (_, _, _), (_, _, _), auroc_mi, auprc_mi, _, _ = get_roc_auc_ensemble(
        hyper, params, test_loader, ood_test_loader, "mutual_information", device
    )
    (_, _, _), (_, _, _), auroc_pe, auprc_pe, auroc_conf, auprc_conf = get_roc_auc_ensemble(
        hyper, params, test_loader, ood_test_loader, "entropy", device
    )
    
    return -nlls.mean(), nlls_dang, accuracy, ece, ece_30, auroc_mi, auprc_mi, auroc_pe, auprc_pe, auroc_conf, auprc_conf

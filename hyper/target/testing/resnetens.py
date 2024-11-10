""" Do not use this module directly unless if you want to test against normal torch modules

To test against batched modules we construct resnet ensembles, and this is the old code trained for CIFAR/TinyImageNet

We adopted a new base code and this is used to ensure the implementations are close enough/the same
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion *
                            planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.num_blocks = num_blocks
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.ModuleList(layers)

  def forward_block(self, layer_ind, x, every):
    layer = {
      1: self.layer1,
      2: self.layer2,
      3: self.layer3,
      4: self.layer4
    }[layer_ind]
    feats = []
    for i in range(self.num_blocks[layer_ind - 1]):
      x = layer[i](x)
      
      if every == 0:
          feats.append(x)
      elif (i % every) == 0:
          feats.append(x)
    return feats, x
          
  def forward(self, x):
    features = []
    out = F.relu(self.bn1(self.conv1(x)))
    features.append(out)
    feats, out = self.forward_block(1, out, 2)
    features.extend(feats)
    feats, out = self.forward_block(2, out, 2)
    features.extend(feats)
    feats, out = self.forward_block(3, out, 1)
    features.extend(feats)
    feats, out = self.forward_block(4, out, 1)
    features.extend(feats)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    features.append(out)
    return features, out


def resnet18(classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=classes)


def resnet34(classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=classes)


def resnet50(classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=classes)


def resnet101(classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=classes)


def resnet152(classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=classes)



# should match genmodule definition!
class ResNetEnsemble(nn.Module):
  def __init__(self, size: int=5, constructor=resnet18, classes=10):
    super().__init__()
    self.ens_size = size
    self.ens = []
    for i in range(self.ens_size):
      net = constructor(classes=classes)
      # net.load_state_dict(data['ens'][i])
      self.ens.append(net.cuda())
    self.ens = nn.ModuleList(self.ens)
  
  def eval(self):
    for e in self.ens:
      e.eval()
  
  def train(self):
    for e in self.ens:
      e.train()
  
  def sample_params(self, *args, **kwargs):
    return None
  
  def forward_params(self, *args, **kwargs):
    return None
  
  # def load_state_dict(self, *args, **kwargs):
  #   print('Skipping load! Already loaded.')

  def target_parameters(self, *args, **kwargs):
    return []  # none to supply

  def forward(self, params, x,
      sample_params=True,
      ret_params=False,
      feature_split=True,
      ood_N=0,
      skip_empty=True,
      split_pred_only=False
    ):
    
    if ood_N is not None and ood_N > 0:
      raise NotImplementedError
    if split_pred_only:
      raise NotImplementedError
    
    # convert to typical format expected by evaluator/trainer
    feats = None
    outs = []
    for net in self.ens:
      f, o = net(x)
      
      # filter nones
      f = list(filter(lambda x: x is not None, f))
      if feats is None:
          feats = [[] for _ in range(len(f))]
      
      # append each feat
      for i in range(len(f)):
          feats[i].append(f[i])

      # add output
      outs.append(o)
    
    # stack features
    out_feats = OrderedDict()
    for i in range(len(feats)):
        out_feats[f'{i}'] = torch.stack(feats[i])

    # stack outputs
    outs = torch.stack(outs)
    
    if ret_params:
      return {'params': None, 'pred': outs, 'pred_ind': outs, 'pred_ood': None, 'feats': out_feats, 'feats_ind': out_feats, 'feats_ood': None}
    return out_feats, outs

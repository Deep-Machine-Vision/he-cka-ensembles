""" ResNets adapted for our framework and supporting D'Angelo's modifications for a ResNet32 model

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Union
from hyper.layers.linear import Linear, FinalLinear
from hyper.layers.conv import Conv2d
from hyper.layers.norm import BatchNorm2d
from hyper.layers.module import SequentialModule, GenModule, Reshape, Flatten, Activation, Module, register_gen_module, build_gen_module
from hyper.target.convnext import build_norm_layer
from collections import OrderedDict


@register_gen_module('resnet_basic_block')
class BasicBlock(GenModule):
    """ Basic block in a ResNet with support for D'Angelo mods for ResNet32 """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track: bool=True, use_norm: bool=True, dang: bool=False, act='relu', norm_block=BatchNorm2d):
        super(BasicBlock, self).__init__(track=track)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.use_norm = use_norm
        self.dang = dang
        self.act = Activation(act)
        if use_norm:
            self.norm1 = norm_block(planes, track=False)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        if use_norm:
            self.norm2 = norm_block(planes, track=False)

        self.shortcut = SequentialModule()
        self.use_short = False
        self.stride = stride
        if stride != 1 or in_planes != self.expansion*planes:
            self.use_short = True

            self.fs_in = in_planes
            self.fs_next = self.expansion*planes
            if not dang:  # d'angelo uses custom downsampling
                self.shortcut = SequentialModule(
                    Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False)
                )
            
            if use_norm:
                self.shortcut.append(norm_block(self.expansion*planes, track=False))
    
    def is_generated(self):
        return False
    
    def define_generated_modules(self):
        mod = super().define_generated_modules()
        mod['conv1'] = self.conv1.define_generated_modules()
        if self.use_norm:
            mod['norm1'] = self.norm1.define_generated_modules()
        
        mod['conv2'] = self.conv2.define_generated_modules()
        if self.use_norm:
            mod['norm2'] = self.norm2.define_generated_modules()
        
        if self.use_short and not self.dang:  # D'Anglo uses custom skip connection
          mod['shortcut'] = self.shortcut.define_generated_modules()
        return mod

    def forward(self, params, x):
        feat = OrderedDict()
        
        # to norm or not to norm. that is the question
        if self.use_norm:
            out = self.act(None, self.norm1(params['norm1'], self.conv1(params['conv1'], x)[1])[1])[1]
            out = self.norm2(params['norm2'], self.conv2(params['conv2'], out)[1])[1]
        else:
            out = self.act(None, self.conv1(params['conv1'], x)[1])[1]
            out = self.conv2(params['conv2'], out)[1]
            
        # D'Angelo uses different pooling/skip approach
        if self.use_short and self.dang:
            pad_left = (self.fs_next - self.fs_in) // 2
            pad_right = int(np.ceil((self.fs_next - self.fs_in) / 2))
            if self.stride == 2:
                shortcut_h = x[:, :, :, ::2, ::2]  # terrible
            # print('shortp', shortcut_h.shape)
            shortcut_h = F.pad(shortcut_h, (0, 0, 0, 0, pad_left, pad_right), "constant", 0)
            # print('short', shortcut_h.shape, 'x', x.shape)
            out += shortcut_h
        elif not self.use_short and self.dang:
            out += x  # simple skip for dang
        else:  # normal skip with params
            out += self.shortcut(params['shortcut'], x)[1]
        out = self.act(None, out)[1]
        
        if self._track:
          feat['block'] = out
        return feat, out

    @staticmethod
    def from_config(config):
        return config


class BasicSharedBlock(GenModule):
    """ Same as above but using parameters shared across ensemble members """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track: bool=True, use_norm: bool=True):
        raise NotImplementedError
        if not use_norm:
            raise NotImplementedError()
        super(BasicSharedBlock, self).__init__(track=track)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = build_norm_layer(planes, groups='auto')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm_layer(planes, groups='auto')

        self.use_short = False
        if stride != 1 or in_planes != self.expansion*planes:
            self.use_short = True
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.short_bn = build_norm_layer(self.expansion*planes, groups='auto')

    def forward(self, params, x):
        out = F.relu(self.bn1(None, self.conv1(x).unsqueeze(0))[1].squeeze(0))
        out = self.bn2(None, self.conv2(out).unsqueeze(0))[1].squeeze(0)
        
        # use conv for shortcut or normal shortcut
        if self.use_short:
            out += self.short_bn(None, self.shortcut(x).unsqueeze(0))[1].squeeze(0)
        else:
            out += x
        
        out = F.relu(out)
        return None, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        raise NotImplementedError('not impl yet')
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



@register_gen_module('resnet')
class ResNet(GenModule):
    def __init__(self, block, block_shared=BasicSharedBlock, num_blocks=None, num_classes=10, shared_blocks: int=0, in_chan: int=3, planes: int=[64, 64, 128, 256, 512], dang_mod: bool=False, act: str='relu', track_every_n=[2, 2, 1, 1], norm_block=BatchNorm2d):
        super(ResNet, self).__init__()
        first_out = planes[0]
        self.in_planes = first_out
        self._act = act
        self.track_every = track_every_n

        self.shared = shared_blocks > 0
        if shared_blocks > 0:
            self.conv1 = nn.Conv2d(in_chan, first_out, kernel_size=3, stride=1, padding=1, bias=False)
            shared_blocks -= 1
        else:
            self.conv1 = Conv2d(in_chan, first_out, kernel_size=3, stride=1, padding=1, bias=False)
        
        # D'Angelo impl does not use any norm
        self.use_norm = not dang_mod
        self.dang_mod = dang_mod
        self.act = Activation('relu')
        
        if self.use_norm:
            self.norm1 = norm_block(first_out, track=False)
        else:
            print('Note using dang mod and building without normalization')
        
        self.num_blocks = num_blocks
        self.layer1 = self._make_layer(block, block_shared, planes[1], num_blocks[0], stride=1, every=self.track_every[0], shared=shared_blocks, act='relu')
        shared_blocks -= num_blocks[0]
        self.layer2 = self._make_layer(block, block_shared, planes[2], num_blocks[1], stride=2, every=self.track_every[1], shared=shared_blocks, act='relu')
        shared_blocks -= num_blocks[1]
        self.layer3 = self._make_layer(block, block_shared, planes[3], num_blocks[2], stride=2, every=self.track_every[2], shared=shared_blocks, act=self._act)
        shared_blocks -= num_blocks[2]
        last_planes = planes[3]
        
        # D'Angelo modification to ResNet does not include layer 4
        if not dang_mod:
            self.layer4 = self._make_layer(block, block_shared, planes[4], num_blocks[3], stride=2, every=self.track_every[3], shared=shared_blocks)
            last_planes = planes[4]
        self.linear = Linear(last_planes*block.expansion, num_classes)

    def _make_layer(self, block, block_shared, planes, num_blocks, stride, every, shared, act='relu'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for ind, stride in enumerate(strides):
            use_block = block_shared if ind < shared else block
            layers.append(use_block(self.in_planes, planes, stride, track=(ind % every == 0), use_norm=self.use_norm, dang=self.dang_mod, act=act))
            self.in_planes = planes * block.expansion
        return SequentialModule(*layers)

    def is_generated(self):
        return False
    
    def define_generated_modules(self):
        mod = super().define_generated_modules()
        if not self.shared:
            mod['conv1'] = self.conv1.define_generated_modules()
            
            if self.use_norm:
                mod['norm1'] = self.norm1.define_generated_modules()
            
        mod['layer1'] = self.layer1.define_generated_modules()
        mod['layer2'] = self.layer2.define_generated_modules()
        mod['layer3'] = self.layer3.define_generated_modules()
        
        if not self.dang_mod:
            mod['layer4'] = self.layer4.define_generated_modules()
        mod['linear'] = self.linear.define_generated_modules()
        return mod

    def forward(self, params, x):
        feat = OrderedDict()
        
        if self.shared:  # use a shared convolution layer
            out = self.act(None, self.bn1(None, self.conv1(x).unsqueeze(0))[1].squeeze(0))[1]
        else:
            if self.use_norm:
                feat['conv1'] = out = self.act(None, self.norm1(params['norm1'], self.conv1(params['conv1'], x)[1])[1])[1]
            else:
                feat['conv1'] = out = self.act(None, self.conv1(params['conv1'], x)[1])[1]
        feat['layer1'], out = self.layer1(params['layer1'], out)
        feat['layer2'], out = self.layer2(params['layer2'], out)
        feat['layer3'], out = self.layer3(params['layer3'], out)
        
        if not self.dang_mod:
            feat['layer4'], out = self.layer4(params['layer4'], out)
        
        # global average pooling
        out = torch.mean(out, dim=[-1, -2])
        
        # flatten
        out = out.view(out.shape[0], out.shape[1], -1)
        
        # linear
        feat['linear'], out = self.linear(params['linear'], out)
        return feat, out

    @staticmethod
    def from_config(configs):
        configs['block'] = build_gen_module(configs['block'])
        return configs


@register_gen_module('resnet18')
def resnet18(num_classes=10, **kwargs):
    """ Builds a typical ResNet18 model"""
    return ResNet(BasicBlock, BasicSharedBlock, [2, 2, 2, 2], num_classes=num_classes, norm_block=BatchNorm2d, **kwargs)


@register_gen_module('resnet32')
def resnet32(num_classes=10, **kwargs):
    """ 6n + 2 blocks Def of ResNet32 following https://github.com/ratschlab/repulsive_ensembles/blob/master/models/mnets_resnet.py """
    raise NotImplementedError('This has not been finished in the rewrite of the code. Come back later')
    return ResNet(BasicBlock, BasicSharedBlock, num_blocks=[5, 5, 5], planes=[16, 16, 32, 64], num_classes=num_classes, dang_mod=True, track_every_n=[3, 3, 2], norm_block=None, **kwargs)


# @TODO finish these
# def resnet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def resnet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def resnet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def resnet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


import math
import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution='uniform'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu',
                 bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'normal':
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode,
                                    nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
                                     nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module, mean=0., std=1., a=-2., b=2., bias=0.):
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

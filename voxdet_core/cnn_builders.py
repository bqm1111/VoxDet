import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# --- DCN wrapper ---

class DCNv2(nn.Module):
    """Deformable Conv2d wrapper compatible with mmcv's DCN config interface."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 deform_groups=1, **kwargs):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.offset_conv = nn.Conv2d(
            in_channels,
            deform_groups * 2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.dcn = DeformConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)


# --- Conv layer builder ---

CONV_LAYERS = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
    'Conv': nn.Conv2d,
    'DCN': DCNv2,
    'DCNv2': DCNv2,
}


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    cfg = cfg.copy()
    layer_type = cfg.pop('type', 'Conv2d')
    layer_cls = CONV_LAYERS.get(layer_type)
    if layer_cls is None:
        raise KeyError(f'Unrecognized conv type {layer_type}')
    # merge remaining cfg items into kwargs
    kwargs.update(cfg)
    return layer_cls(*args, **kwargs)


# --- Norm layer builder ---

NORM_LAYERS = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
}


def build_norm_layer(cfg, num_features, postfix=''):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    requires_grad = cfg.pop('requires_grad', True)
    layer_cls = NORM_LAYERS.get(layer_type)
    if layer_cls is None:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    abbr = layer_type.lower()
    name = abbr + str(postfix)

    if layer_type == 'GN':
        layer = layer_cls(num_channels=num_features, **cfg)
    elif layer_type == 'LN':
        layer = layer_cls(num_features, **cfg)
    else:
        layer = layer_cls(num_features, **cfg)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


# --- Upsample layer builder ---

UPSAMPLE_LAYERS = {
    'deconv': nn.ConvTranspose2d,
    'deconv3d': nn.ConvTranspose3d,
    'nearest': lambda **kw: nn.Upsample(scale_factor=kw.get('stride', 2), mode='nearest'),
    'bilinear': lambda **kw: nn.Upsample(scale_factor=kw.get('stride', 2), mode='bilinear', align_corners=False),
}


def build_upsample_layer(cfg, *args, **kwargs):
    if cfg is None:
        return nn.ConvTranspose2d(*args, **kwargs)
    cfg = cfg.copy()
    layer_type = cfg.pop('type', 'deconv')
    layer_cls = UPSAMPLE_LAYERS.get(layer_type)
    if layer_cls is None:
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    kwargs.update(cfg)
    return layer_cls(*args, **kwargs)


# --- Activation layer builder ---

ACTIVATION_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'Swish': nn.SiLU,
    'HSwish': nn.Hardswish,
    'HSigmoid': nn.Hardsigmoid,
}


def build_activation_layer(cfg):
    if cfg is None:
        return nn.ReLU(inplace=True)
    cfg = cfg.copy()
    layer_type = cfg.pop('type')
    layer_cls = ACTIVATION_LAYERS.get(layer_type)
    if layer_cls is None:
        raise KeyError(f'Unrecognized activation type {layer_type}')
    return layer_cls(**cfg)


# --- build_plugin_layer (stub for ResNet3D) ---

def build_plugin_layer(cfg, *args, **kwargs):
    raise NotImplementedError('build_plugin_layer is not supported in voxdet_core')


# --- Linear (re-export) ---

Linear = nn.Linear


# --- Dropout (re-export) ---

Dropout = nn.Dropout


# --- DropPath ---

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


def build_dropout(cfg, default_args=None):
    if cfg is None:
        return nn.Identity()
    cfg = cfg.copy()
    dropout_type = cfg.pop('type', 'Dropout')
    if dropout_type == 'DropPath':
        return DropPath(**cfg)
    return nn.Dropout(**cfg)


# --- ConvModule ---

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 order=('conv', 'norm', 'act'),
                 **kwargs):
        super().__init__()
        self.order = order
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        # Determine bias
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # Build conv
        conv_cfg = conv_cfg or {}
        conv_cfg_copy = conv_cfg.copy()
        conv_cfg_copy.pop('type', None)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_cfg_copy if not conv_cfg else {})

        # Rebuild conv properly
        conv_type = (conv_cfg or {}).get('type', 'Conv2d')
        conv_cls = CONV_LAYERS.get(conv_type, nn.Conv2d)
        self.conv = conv_cls(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        # Build norm
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.bn_name, self.norm = build_norm_layer(norm_cfg, norm_channels)
        else:
            self.norm = None

        # Build activation
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_.get('type') == 'ReLU' and inplace:
                act_cfg_['inplace'] = True
            self.activate = build_activation_layer(act_cfg_)
        else:
            self.activate = None

    def forward(self, x, activate=True, norm=True):
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and self.with_norm and norm:
                x = self.norm(x)
            elif layer_name == 'act' and self.with_activation and activate:
                x = self.activate(x)
        return x


# --- SELayer (Squeeze-and-Excitation) ---

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""

    def __init__(self, channels, ratio=16, act_cfg=(dict(type='ReLU'), dict(type='HSigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, dict(type='Sigmoid'))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        mid_channels = make_divisible(channels // ratio, 8)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1)
        self.act1 = build_activation_layer(act_cfg[0])
        self.conv2 = nn.Conv2d(mid_channels, channels, 1)
        self.act2 = build_activation_layer(act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return x * out


# --- make_divisible ---

def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value

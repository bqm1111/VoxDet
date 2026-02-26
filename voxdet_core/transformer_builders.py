import copy
import torch
import torch.nn as nn

from .base_module import BaseModule
from .registry import (
    TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE,
    FEEDFORWARD_NETWORK, build_from_cfg,
)
from .cnn_builders import build_activation_layer, build_dropout


class TransformerLayerSequence(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder.

    Builds N transformer layers from a config dict.
    """

    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        self.layers = nn.ModuleList()
        for cfg in transformerlayers:
            self.layers.append(build_from_cfg(cfg, TRANSFORMER_LAYER))
        self.num_layers = num_layers or len(self.layers)

    def forward(self, query, key=None, value=None, **kwargs):
        for layer in self.layers:
            query = layer(query, key, value, **kwargs)
        return query


@FEEDFORWARD_NETWORK.register_module()
class FFN(BaseModule):
    """Feed-Forward Network used in Transformers."""

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        assert num_fcs >= 2, f'num_fcs should be >= 2, got {num_fcs}'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                self.activate,
                nn.Dropout(ffn_drop),
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        if dropout_layer is not None:
            self.dropout_layer = build_dropout(dropout_layer)
        else:
            self.dropout_layer = torch.nn.Identity()

        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

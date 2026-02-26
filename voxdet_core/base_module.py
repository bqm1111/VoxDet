import torch.nn as nn


class BaseModule(nn.Module):
    """A thin wrapper around nn.Module that accepts and ignores init_cfg."""

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass


ModuleList = nn.ModuleList
Sequential = nn.Sequential

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

__all__ = ['TanhQ']


class TanhQ(Module):
    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def forward(self, x):
        if self.fake:
            return torch.tanh(x)
        x_value = F.hardtanh(x, -1, 1)
        x_graph = torch.tanh(x)
        return x_value.detach() + x_graph - x_graph.detach()

    def extra_repr(self):
        s_prefix = super(TanhQ, self).extra_repr()
        if self.fake:
            return '{}fake'.format(s_prefix)
        return ''


class SigmoidQ(Module):
    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def forward(self, x):
        if self.fake:
            return torch.sigmoid(x)
        x_value = F.hardtanh(1 / 4 * x + 1 / 2, 0, 1)
        x_graph = torch.sigmoid(x)
        return x_value.detach() + x_graph - x_graph.detach()

    def extra_repr(self):
        s_prefix = super(SigmoidQ, self).extra_repr()
        if self.fake:
            return 'fake'.format(s_prefix)
        return ''

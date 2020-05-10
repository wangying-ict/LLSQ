import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Conv2dBN']


class Conv2dBN(nn.Conv2d):
    """
    quantize weights after fold BN to conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(Conv2dBN, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self._bn = nn.BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if self._bn.training:
            conv_out = F.conv2d(input, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            # calculate mean and various
            fake_out = self._bn(conv_out)
            conv_out = conv_out.transpose(1, 0).contiguous()
            conv_out = conv_out.view(conv_out.size(0), -1)
            mu = conv_out.mean(dim=1)  # it is the same as mean calculated in _bn.
            var = torch.var(conv_out, dim=1, unbiased=False)
        else:
            mu = self._bn.running_mean
            var = self._bn.running_var
        if self._bn.affine:
            gamma = self._bn.weight
            beta = self._bn.bias
        else:
            gamma = torch.ones(self.out_channels).to(var.device)
            beta = torch.zeros(self.out_channels).to(var.device)

        A = gamma.div(torch.sqrt(var + self._bn.eps))
        A_expand = A.expand_as(self.weight.transpose(0, -1)).transpose(0, -1)

        weight_fold = self.weight * A_expand
        if self.bias is None:
            bias_fold = (- mu) * A + beta
        else:
            bias_fold = (self.bias - mu) * A + beta
        out = F.conv2d(input, weight_fold, bias_fold, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

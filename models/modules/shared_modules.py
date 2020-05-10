import torch.nn as nn
import torch.nn.functional as F
import ipdb

__all__ = ['Conv2dShare', 'BatchNorm2dShare', 'ReLUshare', 'MaxPool2dShare', 'ActShare']


class Conv2dShare(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, share_num=2):
        super(Conv2dShare, self).__init__()
        self.share_num = share_num
        self.convs = nn.ModuleList()
        for i in range(share_num):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups, bias=bias))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.convs[i](input[i]))
        return ret


class ActShare(nn.Module):
    def __init__(self):
        super(ActShare, self).__init__()

    def forward(self, input):
        return input


class BatchNorm2dShare(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, share_num=2):
        super(BatchNorm2dShare, self).__init__()
        self.share_num = share_num
        self.bns = nn.ModuleList()
        for i in range(share_num):
            self.bns.append(nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.bns[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(BatchNorm2dShare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)


class ReLUshare(nn.Module):
    def __init__(self, inplace=False, share_num=2):
        super(ReLUshare, self).__init__()
        self.share_num = share_num
        self.relu_list = nn.ModuleList()
        for i in range(share_num):
            self.relu_list.append(nn.ReLU(inplace))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.relu_list[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(ReLUshare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)


class MaxPool2dShare(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, share_num=2):
        super(MaxPool2dShare, self).__init__()
        self.share_num = share_num
        self.pools = nn.ModuleList()
        for i in range(share_num):
            self.pools.append(nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.pools[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(MaxPool2dShare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)

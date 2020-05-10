"""
Codes for implementing TTQ ternary
Copied from https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Codes/utils/TTQ.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import Qmodes
import ipdb

__all__ = ['TTQ_CNN', 'TTQ_Linear', 'Conv2dTBQ', 'LinearTBQ']

ratio = 0.1


class Function_binary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale, shift, positive=True):
        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        # thresh = thresh_factor * torch.max(torch.abs(w_reshape), dim=0)[0]
        if positive:
            indices = (w_reshape > shift).to(weight.device).float()
        else:
            indices = (w_reshape < shift).to(weight.device).float()

        binary_weight = scale * indices

        ctx.save_for_backward(indices, scale)

        return binary_weight.transpose(0, 1).reshape(weight.shape)

    @staticmethod
    def backward(ctx, grad_binary_weight):
        grad_reshape = grad_binary_weight.reshape([grad_binary_weight.shape[0], -1]).transpose(0, 1)
        indices, scale = ctx.saved_tensors
        pruned_indices = torch.ones(indices.shape).to(indices.device) - indices
        grad_scale = torch.mean(grad_reshape * indices, dim=0)
        grad_shift = -1 * torch.mean(grad_reshape * pruned_indices, dim=0)

        grad_fp_weight = scale * grad_reshape * indices + \
                         grad_reshape * pruned_indices

        return grad_fp_weight.transpose(0, 1).reshape(grad_binary_weight.shape), \
               grad_scale, grad_shift, None


class Function_binary_fc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, scale, shift, positive=True):
        if positive:
            indices = (weight > shift).to(weight.device).float()
        else:
            indices = (weight < shift).to(weight.device).float()

        binary_weight = scale * indices

        ctx.save_for_backward(indices, scale)

        return binary_weight

    @staticmethod
    def backward(ctx, grad_binary_weight):
        # grad_reshape = grad_binary_weight.transpose(0, 1)
        indices, scale = ctx.saved_tensors
        pruned_indices = torch.ones(indices.shape).to(indices.device) - indices
        grad_scale = torch.mean(grad_binary_weight * indices)
        grad_shift = -1 * torch.mean(grad_binary_weight * pruned_indices)

        grad_fp_weight = scale * grad_binary_weight * indices + \
                         grad_binary_weight * pruned_indices

        return grad_fp_weight, grad_scale, grad_shift, None


# class FunctionTBQ(torch.autograd.Function):
#     # kernel-wise quantization
#     # incremental tbq.
#     # Bad Results
#     @staticmethod
#     def forward(ctx, weight, pos, neg, ratio=0.5, positive=True):
#         w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
#         thresh = torch.zeros(weight.shape[0]).to(weight.device)
#
#         pos_indices = (w_reshape > thresh).to(weight.device).float()
#         neg_indices = (w_reshape < -thresh).to(weight.device).float()
#         # ratio: 0 => 0.1 => 0.2 => ... => 0.9 => 1.0
#         increase_mask = (torch.rand_like(neg_indices).to(weight.device) > ratio).float()
#         if positive:
#             neg_indices = neg_indices * increase_mask
#         else:
#             pos_indices = pos_indices * increase_mask
#
#         ternary_weight = pos * pos_indices + neg * neg_indices
#
#         ctx.save_for_backward(pos_indices, neg_indices, pos, neg)
#
#         return ternary_weight.transpose(0, 1).reshape(weight.shape)
#
#     @staticmethod
#     def backward(ctx, grad_ternary_weight):
#         grad_reshape = grad_ternary_weight.reshape([grad_ternary_weight.shape[0], -1]).transpose(0, 1)
#         pos_indices, neg_indices, pos, neg = ctx.saved_tensors
#         pruned_indices = torch.ones(pos_indices.shape).to(pos.device) - pos_indices - neg_indices
#
#         grad_pos = torch.mean(grad_reshape * pos_indices, dim=0)
#         grad_neg = torch.mean(grad_reshape * neg_indices, dim=0)
#
#         grad_fp_weight = pos * grad_reshape * pos_indices + \
#                          grad_reshape * pruned_indices + \
#                          neg * grad_reshape * neg_indices
#
#         return grad_fp_weight.transpose(0, 1).reshape(grad_ternary_weight.shape), \
#                grad_pos, grad_neg, None, None
class LinearTBQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearTBQ, self).__init__(in_features, out_features, bias=bias)
        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(torch.rand([]))
        self.pos_shift = nn.Parameter(torch.zeros([]))
        self.neg_shift = nn.Parameter(torch.zeros([]))
        self.register_buffer('init_state', torch.zeros(1))

        #  TODO: Split in_channel according to the size of crossbar.
        pos_num_crossbar = int(in_features / 128 / 2)
        split_channel = int(pos_num_crossbar * 128)
        # split across channel
        if split_channel == 0:
            split_channel = int((in_features + 1) / 2)
        self.split_channel = split_channel
        self.split_channel = in_features - split_channel
        print('split/in_features: {}/{}'.format(self.split_channel, in_features))

    def forward(self, x):
        if self.training and self.init_state == 0:
            pos_indices = (self.weight > 0).to(self.weight.device).float()
            neg_indices = (self.weight < 0).to(self.weight.device).float()
            pos_init = (self.weight * pos_indices).mean()
            neg_init = (self.weight * neg_indices).mean()
            self.pos.data.copy_(pos_init)
            self.neg.data.copy_(neg_init)
            self.init_state.fill_(1)

        weight_pos = self.weight[:, :self.split_channel]
        weight_neg = self.weight[:, self.split_channel:]
        binary_weight_pos = Function_binary_fc.apply(weight_pos, self.pos, self.pos_shift, True)
        binary_weight_neg = Function_binary_fc.apply(weight_neg, self.neg, self.neg_shift, False)
        binary_weight = torch.cat([binary_weight_pos, binary_weight_neg], dim=1)
        return F.linear(x, binary_weight, self.bias)


class Conv2dTBQ(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, mode=Qmodes.kernel_wise):
        super(Conv2dTBQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mode = mode
        if mode == Qmodes.kernel_wise:
            self.pos = nn.Parameter(torch.rand(out_channels))
            self.pos_shift = nn.Parameter(torch.zeros(out_channels))
            self.neg = nn.Parameter(-torch.rand(out_channels))
            self.neg_shift = nn.Parameter(torch.zeros(out_channels))
        else:
            self.pos = nn.Parameter(torch.rand([]))
            self.neg = nn.Parameter(-torch.rand([]))
        self.register_buffer('init_state', torch.zeros(1))

        #  TODO: Split in_channel according to the size of crossbar.
        pos_num_crossbar = int(in_channels * (kernel_size ** 2) / 128 / 2)
        split_channel = int(pos_num_crossbar * 128 / (kernel_size ** 2))
        # split across channel
        if split_channel == 0:
            split_channel = int((in_channels + 1) / 2)
        self.split_channel = split_channel
        print('split channel / in channel: {}/{}'.format(split_channel, in_channels))

    def forward(self, x):
        if self.mode == Qmodes.kernel_wise:
            if self.training and self.init_state == 0:
                # init running scale
                w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
                pos_indices = (w_reshape > 0).to(w_reshape.device).float()
                neg_indices = (w_reshape < 0).to(w_reshape.device).float()
                pos_init = (w_reshape * pos_indices).mean(dim=0)
                neg_init = (w_reshape * neg_indices).mean(dim=0)
                self.pos.data.copy_(pos_init)
                self.neg.data.copy_(neg_init)
                self.init_state.fill_(1)

            weight_pos = self.weight[:, :self.split_channel, :, :]
            weight_neg = self.weight[:, self.split_channel:, :, :]
            binary_weight_pos = Function_binary.apply(weight_pos, self.pos, self.pos_shift, True)
            binary_weight_neg = Function_binary.apply(weight_neg, self.neg, self.neg_shift, False)
            binary_weight = torch.cat([binary_weight_pos, binary_weight_neg], dim=1)
        else:
            raise NotImplementedError
        return F.conv2d(x, binary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dTBQ, self).extra_repr()
        return '{}, {}'.format(
            s_prefix, self.mode)


class Function_ternary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, pos, neg, thresh_factor):
        thresh = thresh_factor * torch.max(torch.abs(weight))

        pos_indices = (weight > thresh).type(torch.cuda.FloatTensor)
        neg_indices = (weight < -thresh).type(torch.cuda.FloatTensor)

        ternary_weight = pos * pos_indices + neg * neg_indices

        ctx.save_for_backward(pos_indices, neg_indices, pos, neg)

        return ternary_weight

    @staticmethod
    def backward(ctx, grad_ternary_weight):
        pos_indices, neg_indices, pos, neg = ctx.saved_tensors
        pruned_indices = torch.ones(pos_indices.shape).cuda() - pos_indices - neg_indices

        grad_pos = torch.mean(grad_ternary_weight * pos_indices)
        grad_neg = torch.mean(grad_ternary_weight * neg_indices)

        grad_fp_weight = pos * grad_ternary_weight * pos_indices + \
                         grad_ternary_weight * pruned_indices + \
                         neg * grad_ternary_weight * neg_indices

        return grad_fp_weight, grad_pos, grad_neg, None


class Function_ternary_kernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, pos, neg, thresh_factor):
        w_reshape = weight.reshape([weight.shape[0], -1]).transpose(0, 1)
        thresh = thresh_factor * torch.max(torch.abs(w_reshape), dim=0)[0]

        pos_indices = (w_reshape > thresh).to(weight.device).float()
        neg_indices = (w_reshape < -thresh).to(weight.device).float()

        ternary_weight = pos * pos_indices + neg * neg_indices

        ctx.save_for_backward(pos_indices, neg_indices, pos, neg)

        return ternary_weight.transpose(0, 1).reshape(weight.shape)

    @staticmethod
    def backward(ctx, grad_ternary_weight):
        grad_reshape = grad_ternary_weight.reshape([grad_ternary_weight.shape[0], -1]).transpose(0, 1)
        pos_indices, neg_indices, pos, neg = ctx.saved_tensors
        pruned_indices = torch.ones(pos_indices.shape).to(pos.device) - pos_indices - neg_indices

        grad_pos = torch.mean(grad_reshape * pos_indices, dim=0)
        grad_neg = torch.mean(grad_reshape * neg_indices, dim=0)

        grad_fp_weight = pos * grad_reshape * pos_indices + \
                         grad_reshape * pruned_indices + \
                         neg * grad_reshape * neg_indices

        return grad_fp_weight.transpose(0, 1).reshape(grad_ternary_weight.shape), \
               grad_pos, grad_neg, None


class TTQ_CNN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, thresh_factor=0.05, mode=Qmodes.kernel_wise):
        super(TTQ_CNN, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.mode = mode
        if mode == Qmodes.kernel_wise:
            self.pos = nn.Parameter(torch.rand(out_channels))
            self.neg = nn.Parameter(-torch.rand(out_channels))
        else:
            self.pos = nn.Parameter(torch.rand([]))
            self.neg = nn.Parameter(-torch.rand([]))
        self.thresh_factor = thresh_factor
        self.ternary_weight = None
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.mode == Qmodes.kernel_wise:
            if self.training and self.init_state == 0:
                # init running scale
                w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
                pos_indices = (w_reshape > 0).to(w_reshape.device).float()
                neg_indices = (w_reshape < 0).to(w_reshape.device).float()
                pos_init = (w_reshape * pos_indices).mean(dim=0)
                neg_init = (w_reshape * neg_indices).mean(dim=0)
                self.pos.data.copy_(pos_init)
                self.neg.data.copy_(neg_init)
                self.init_state.fill_(1)
            self.ternary_weight = Function_ternary_kernel.apply(self.weight, self.pos, self.neg, self.thresh_factor)
        else:
            if self.training and self.init_state == 0:
                # init running scale
                pos_indices = (self.weight > 0).to(self.weight.device).float()
                neg_indices = (self.weight < 0).to(self.weight.device).float()
                pos_init = (self.weight * pos_indices).mean()
                neg_init = (self.weight * neg_indices).mean()
                self.pos.data.copy_(pos_init)
                self.neg.data.copy_(neg_init)
                self.init_state.fill_(1)
            self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)

        return F.conv2d(x, self.ternary_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(TTQ_CNN, self).extra_repr()
        return '{}, {}'.format(
            s_prefix, self.mode)


class TTQ_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, thresh_factor=0.05):
        super(TTQ_Linear, self).__init__(in_features, out_features, bias=bias)

        self.pos = nn.Parameter(torch.rand([]))
        self.neg = nn.Parameter(-torch.rand([]))
        self.thresh_factor = thresh_factor

        self.ternary_weight = None
        # self.register_buffer('init_state', torch.ones(1))

    def forward(self, x):
        # if self.training and self.init_state == 0:
        #     # init running scale
        #     pos_indices = (self.weight > 0).to(self.weight.device).float()
        #     neg_indices = (self.weight < 0).to(self.weight.device).float()
        #     pos_init = (self.weight * pos_indices).mean()
        #     neg_init = (self.weight * neg_indices).mean()
        #     self.pos.data.copy_(pos_init)
        #     self.neg.data.copy_(neg_init)
        #     self.init_state.fill_(1)
        self.ternary_weight = Function_ternary.apply(self.weight, self.pos, self.neg, self.thresh_factor)

        return F.linear(x, self.ternary_weight, self.bias)

    def extra_repr(self):
        s_prefix = super(TTQ_Linear, self).extra_repr()
        return '{}'.format(
            s_prefix)


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()
        self.conv1 = TTQ_CNN(3, 32, 5, 1, 2)

    def forward(self, x):
        return self.conv1(x)


def measure_net_stats(layer):
    ternary_weight = layer.ternary_weight.data
    pos = layer.pos.data
    neg = layer.neg.data
    n_pos = torch.sum(ternary_weight > 0).type(torch.FloatTensor)
    n_neg = torch.sum(ternary_weight < 0).type(torch.FloatTensor)
    n_prune = torch.sum(ternary_weight == 0).type(torch.FloatTensor)
    n_weight = ternary_weight.numel()

    return pos, neg, n_pos / n_weight, n_neg / n_weight, n_prune / n_weight


if __name__ == '__main__':
    net = testNet()
    inputs = torch.rand([10, 3, 32, 32]).cuda()
    targets = torch.rand([10, 32, 32, 32]).cuda()

    net.cuda()
    outputs = net(inputs)
    losses = nn.MSELoss()(outputs, targets)
    losses.backward()
    print(outputs.shape)

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.modules import Qmodes, _Conv2dQ, _LinearQ, log_shift
import numpy as np

import ipdb

__all__ = ['Conv2dBWN', 'LinearBWN', 'Conv2dBWNS', 'LinearBWNS', 'Conv2dBNBWNS', 'FunSign']


class Conv2dBNBWNS(_Conv2dQ):
    """
        quantize weights after fold BN to conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 nbits=4, mode=Qmodes.kernel_wise,
                 ):
        super(Conv2dBNBWNS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           nbits=nbits, mode=mode)
        self._bn = nn.BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1
            # if self.q_mode is Qmodes.kernel_wise:
            #     raise NotImplementedError

    def forward(self, x):
        if self._bn.training:
            if self.alpha is not None and self.init_state != 0:
                w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
                alpha = self.alpha.detach()
                pre_quantized_weight = w_reshape / alpha
                quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
                w_q = quantized_weight.transpose(0, 1).reshape(self.weight.shape)
            else:
                w_q = self.weight
            conv_out = F.conv2d(x, w_q, self.bias, self.stride,
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
        if self.alpha is None:
            return F.conv2d(x, weight_fold, bias_fold, self.stride,
                            self.padding, self.dilation, self.groups)
        w_reshape = weight_fold.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(torch.ones(1))
            if self.q_mode == Qmodes.layer_wise:
                alpha_fp = torch.mean(torch.abs(w_reshape.data))
                alpha_s = log_shift(alpha_fp)
                if alpha_s >= 1:
                    alpha_s /= 2
                print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            else:
                alpha_fp = torch.mean(torch.abs(w_reshape.data), dim=0)
                alpha_s = log_shift(alpha_fp)
                for i in range(len(alpha_s)):
                    if alpha_s[i] >= 1:
                        alpha_s /= 2
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)

        alpha = self.alpha.detach()
        pre_quantized_weight = w_reshape / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        w_q = quantized_weight.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2dBWNS(_Conv2dQ):
    """
        'S' represents shift. In this case, the alpha must be 1/2^n
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.kernel_wise, ):
        super(Conv2dBWNS, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1
            # if self.q_mode is Qmodes.kernel_wise:
            #     raise NotImplementedError

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(torch.ones(1))
            if self.q_mode == Qmodes.layer_wise:
                alpha_fp = torch.mean(torch.abs(w_reshape.data))
                alpha_s = log_shift(alpha_fp)
                if alpha_s >= 1:
                    alpha_s /= 2
                print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            else:
                alpha_fp = torch.mean(torch.abs(w_reshape.data), dim=0)
                alpha_s = log_shift(alpha_fp)
                for i in range(len(alpha_s)):
                    if alpha_s[i] >= 1:
                        alpha_s /= 2
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)

        alpha = self.alpha.detach()
        pre_quantized_weight = w_reshape / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        w_q = quantized_weight.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # todo: no bias
        # return F.conv2d(x, w_q, None, self.stride,
        #                 self.padding, self.dilation, self.groups)


class Conv2dBWN(_Conv2dQ):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.kernel_wise, ):
        super(Conv2dBWN, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1
            if self.q_mode is Qmodes.kernel_wise:
                raise NotImplementedError

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(torch.ones(1))
            self.alpha.data.copy_(torch.mean(torch.abs(self.weight.data)))
            # alpha = torch.mean(torch.abs(self.weight.data))
            self.init_state.fill_(1)

        alpha = self.alpha.detach()
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearBWNS(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearBWNS, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1

    def save_inner_data(self, save, prefix, name, tensor):
        if not save:
            return
        print('saving {}_{} shape: {}'.format(prefix, name, tensor.size()))
        np.save('{}_{}'.format(prefix, name), tensor.detach().cpu().numpy())

    def forward(self, x, save=False):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        if self.training and self.init_state == 0:
            alpha_fp = torch.mean(torch.abs(self.weight.data))
            alpha_s = log_shift(alpha_fp)
            if alpha_s >= 1:
                alpha_s /= 2
            print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)
        alpha = self.alpha.detach()
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        self.save_inner_data(save, 'fc', 'alpha', alpha)
        self.save_inner_data(save, 'fc', 'weight', quantized_weight)
        self.save_inner_data(save, 'fc', 'bias', self.bias)
        return F.linear(x, quantized_weight, self.bias)
        # todo: no bias
        # return F.linear(x, quantized_weight)


class LinearBWN(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearBWN, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(torch.mean(torch.abs(self.weight.data)))
            self.init_state.fill_(1)
        alpha = self.alpha.detach()
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        return F.linear(x, quantized_weight, self.bias)


class Conv2dBWV(_Conv2dQ):
    """
        BWN variant, learned alpha
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.kernel_wise, ):
        super(Conv2dBWV, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        if self.nbits > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits = 1
            if self.q_mode is Qmodes.kernel_wise:
                raise NotImplementedError

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(torch.ones(1))
            self.alpha.data.copy_(torch.mean(torch.abs(self.weight.data)))
            # alpha = torch.mean(torch.abs(self.weight.data))
            self.init_state.fill_(1)

        alpha = self.alpha.detach()
        pre_quantized_weight = self.weight / alpha
        quantized_weight = alpha * FunSign.apply(pre_quantized_weight)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class FunSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return torch.sign(weight)

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None

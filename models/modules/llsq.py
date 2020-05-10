"""
@inproceedings{
    zhao2020linear,
    title={Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware},
    author={Xiandong Zhao and Ying Wang and Xuyi Cai and Cheng Liu and Lei Zhang},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=H1lBj2VFPS}
}
"""
import torch
import torch.nn.functional as F
import ipdb
from models.modules import _ActQ, log_shift, ln_error, update_running_scale, _Conv2dQ, Qmodes, _LinearQ, round_cus

__all__ = ['ActLLSQS', 'ActLLSQ', 'Conv2dLLSQ', 'LinearLLSQ']


class FunLLSQ(torch.autograd.Function):
    # TODO:
    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp, Qmode, is_l2, is_act=True):
        ctx.other = Qn, Qp, Qmode, is_l2, is_act
        q_x = (x / alpha).round().clamp(Qn, Qp)
        x_q = q_x * alpha
        ctx.save_for_backward(x, alpha)
        return x_q

    @staticmethod
    def backward(ctx, grad_x):

        return grad_x, grad_alpha, None, None, None, None, None


class ActLLSQ(_ActQ):
    def __init__(self, nbits=4, signed=False, is_l2=True):
        super(ActLLSQ, self).__init__(nbits=nbits, signed=signed)
        self.add_param('is_l2', is_l2)

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.signed:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            # empirical value
            if self.nbits >= 4:
                init_value = (Qp + 1)
            elif self.nbits == 3:
                init_value = 2 * Qp
            else:
                # TODO
                init_value = 2 * Qp
            self.alpha.data.fill_(x.detach().abs().max() / init_value)

        # scale = self.alpha.detach()
        # TODO
        # if self.scale_bits > 0:
        #     scale, _ = truncation(scale, nbits=self.scale_bits)
        # error, x_clip, y = ln_error(x, self.nbits, scale, is_act=True, l2=self.is_l2)
        y = FunLLSQ.apply(x, self.alpha, Qn, Qp, Qmodes.layer_wise, self.kwargs_q['is_l2'], True)
        # output = y.detach() + x_clip - x_clip.detach()
        return y


class LinearLLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4, is_l2=True):
        super(LinearLLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)
        self.add_param('is_l2', is_l2)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.transpose(0, 1)
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            self.alpha.data.fill_(w_reshape.detach().abs().max() / (Qp + 1))
        w_reshape_q = FunLLSQ.apply(w_reshape, self.alpha, Qn, Qp, Qmodes.layer_wise, self.kwargs_q['is_l2'], False)
        w_q = w_reshape_q.transpose(0, 1)
        return F.linear(x, w_q, self.bias)


class Conv2dLLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise, is_l2=True):
        super(Conv2dLLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        self.add_param('is_l2', is_l2)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            if self.q_mode == Qmodes.layer_wise:
                self.alpha.data.copy_(w_reshape.detach().abs().max() / (Qp + 1))
            else:
                self.alpha.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / Qp)
            self.init_state.fill_(1)
        w_reshape_q = FunLLSQ.apply(w_reshape, self.alpha, Qn, Qp, self.q_mode, self.kwargs_q['is_l2'], False)
        w_q = w_reshape_q.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ActLLSQS(_ActQ):
    def __init__(self, nbits=4, signed=False, floor=False, custom=False):
        super(ActLLSQS, self).__init__(nbits=nbits, signed=signed)
        self.add_param('floor', floor)
        self.add_param('custom', custom)

    def forward(self, x):
        import ipdb
        if self.alpha is None or x.max() < 1e-6:
            return x
        if self.training and self.init_state == 0:
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if self.signed:
                alpha_fp = x.detach().abs().max() / 2 ** (self.nbits - 1)
            else:
                alpha_fp = x.detach().abs().max() / 2 ** self.nbits
            alpha_s = log_shift(alpha_fp)
            print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)
        alpha = self.alpha.detach()
        if self.signed:
            x_clip = (x / alpha).clamp(- 2 ** (self.nbits - 1), 2 ** (self.nbits - 1) - 1)
        else:
            x_clip = (x / alpha).clamp(0, 2 ** self.nbits - 1)
        if self.kwargs_q['floor']:
            x_round = x_clip.floor()
        elif self.kwargs_q['custom']:
            x_round = round_cus(x_clip)
        else:
            x_round = x_clip.round()
        x_round = x_round * alpha
        x_clip = x_clip * alpha
        x_q = x_clip - x_clip.detach() + x_round.detach()
        return x_q


class ActDNQ(_ActQ):
    def __init__(self, nbits=4, signed=False):
        super(ActDNQ, self).__init__(nbits=nbits, signed=signed)

    def forward(self, x):
        if self.alpha is None or x.max() < 1e-6:
            assert ValueError
            return x
        if self.training and self.init_state == 0:
            # Please select a init_rate for activation.
            # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
            if self.signed:
                alpha_fp = x.detach().abs().max() / 2 ** (self.nbits - 1)
            else:
                alpha_fp = x.detach().abs().max() / 2 ** self.nbits
            alpha_s = log_shift(alpha_fp)
            print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)
        alpha = self.alpha.detach()
        if self.signed:
            x_clip = (x / alpha).clamp(- 2 ** (self.nbits - 1), 2 ** (self.nbits - 1) - 1)
        else:
            x_clip = (x / alpha).clamp(0, 2 ** self.nbits - 1)
        x_round = x_clip.round()
        x_round = x_round * alpha
        x_clip = x_clip * alpha
        x_q = x_clip - x_clip.detach() + x_round.detach()
        return x_q


class LinearDNQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearDNQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.transpose(0, 1)
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            alpha_fp = w_reshape.detach().abs().max() / (Qp + 1)
            alpha_s = log_shift(alpha_fp)
            print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            self.alpha.data.copy_(alpha_s)
        alpha = self.alpha.detach()
        w_q = (w_reshape / alpha).clamp(Qn, Qp).round() * alpha
        w_q = w_q.transpose(0, 1)
        return F.linear(x, w_q, self.bias)


class Conv2dDNQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise):
        super(Conv2dDNQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            if self.q_mode == Qmodes.layer_wise:
                alpha_fp = w_reshape.detach().abs().max() / (Qp + 1)
                alpha_s = log_shift(alpha_fp)
                print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            else:
                alpha_fp = w_reshape.detach().abs().max(dim=0)[0] / Qp
                ipdb.set_trace()
                alpha_s = log_shift(alpha_fp)
                print('-----')
            self.alpha.data.copy_(alpha_s)
            self.init_state.fill_(1)
        alpha = self.alpha.detach()
        w_reshape_q = (w_reshape / alpha).round().clamp(Qn, Qp) * alpha
        w_q = w_reshape_q.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

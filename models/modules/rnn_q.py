import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import LSTMCell

from models.modules import log_shift, FunSign
from .activation import TanhQ, SigmoidQ
from .eltwise import EltwiseAdd, EltwiseMult
from .llsq import ActLLSQS
import ipdb
import numpy as np

__all__ = ['LSTMCellQ']


class LSTMCellQ(LSTMCell):
    r"""
      Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
                hx, cx = rnn(input[i], (hx, cx))
                output.append(hx)

     .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o \tanh(c') \\
        \end{array}
    """

    def __init__(self, input_size, hidden_size, bias=True, nbits_w=-1, nbits_a=-1, round_cus=False):
        super(LSTMCellQ, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias)
        # Treat f,i,o,c_ex as one single object:
        # self.fc_gate_x = nn.Linear(input_size, hidden_size * 4, bias=bias)
        # self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        # Apply activations separately:
        self.act_f = SigmoidQ(fake=(nbits_a <= 0))
        self.act_i = SigmoidQ(fake=(nbits_a <= 0))
        self.act_o = SigmoidQ(fake=(nbits_a <= 0))
        self.act_g = TanhQ(fake=(nbits_a <= 0))
        # Calculate cell:
        self.eltwisemult_cell_forget = EltwiseMult()
        self.eltwisemult_cell_input = EltwiseMult()
        self.eltwiseadd_cell = EltwiseAdd()
        # Calculate hidden:
        self.act_h = TanhQ(fake=(nbits_a <= 0))
        self.eltwisemult_hidden = EltwiseMult()

        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.actq1 = ActLLSQS(nbits=nbits_a, signed=True, custom=round_cus)
        self.actq2 = ActLLSQS(nbits=nbits_a, signed=True, custom=round_cus)
        self.actq3 = ActLLSQS(nbits=nbits_a, signed=True, custom=round_cus)
        self.actq4 = ActLLSQS(nbits=nbits_a, signed=True, custom=round_cus)
        if self.nbits_w > 0:
            print('Only support 1 or -1, change the nbits to 1')
            self.nbits_w = 1
        if self.nbits_w < 0:
            self.register_parameter('alpha', None)
            return
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def save_inner_data(self, save, prefix, name, loop_id, tensor):
        if not save:
            return
        print('saving {}_{}_{} shape: {}'.format(prefix, name, loop_id, tensor.size()))
        np.save('{}_{}_{}'.format(prefix, name, loop_id), tensor.detach().cpu().numpy())

    def forward(self, x, hx=None, prefix='', loop_id=-1, save=False):
        self.check_forward_input(x)
        if hx is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        self.check_forward_hidden(x, hx[0], '[0]')
        self.check_forward_hidden(x, hx[1], '[1]')
        h_prev, c_prev = hx
        x_h_prev = torch.cat((x, h_prev), dim=1)
        x_h_prev_q = self.actq1(x_h_prev)
        self.save_inner_data(save, prefix, 'x_h_prev_q', loop_id, x_h_prev_q)
        weight_ih_hh = torch.cat((self.weight_ih, self.weight_hh), dim=1)
        bias_ih_hh = self.bias_ih + self.bias_hh
        if self.alpha is None:  # don't quantize weight and bias
            fc_gate = F.linear(x_h_prev_q, weight_ih_hh, bias_ih_hh)
        else:
            if self.training and self.init_state == 0:
                alpha_fp = torch.mean(torch.abs(weight_ih_hh))
                alpha_s = log_shift(alpha_fp)
                if alpha_s >= 1:
                    alpha_s /= 2
                print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
                self.alpha.data.copy_(alpha_s)
                self.init_state.fill_(1)

            alpha = self.alpha.detach()
            self.save_inner_data(save, prefix, 'alpha', 0, alpha)
            weight_ih_hh_q = alpha * FunSign.apply(weight_ih_hh / alpha)
            self.save_inner_data(save, prefix, 'weight_ih_hh_q', 0, weight_ih_hh_q)
            # todo: quantize bias
            self.save_inner_data(save, prefix, 'bias_ih_hh', 0, bias_ih_hh)
            fc_gate = F.linear(x_h_prev_q, weight_ih_hh_q, bias_ih_hh)
            # todo: no bias
            fc_gate = F.linear(x_h_prev_q, weight_ih_hh_q)
        fc_gate_q = self.actq4(fc_gate)
        i, f, g, o = torch.chunk(fc_gate_q, 4, dim=1)
        i, f, g, o = self.actq3(self.act_i(i)), self.actq3(self.act_f(f)), \
                     self.actq4(self.act_g(g)), self.actq3(self.act_o(o))
        self.save_inner_data(save, prefix, 'i', loop_id, i)
        self.save_inner_data(save, prefix, 'f', loop_id, f)
        self.save_inner_data(save, prefix, 'g', loop_id, g)
        self.save_inner_data(save, prefix, 'o', loop_id, o)
        self.save_inner_data(save, prefix, 'c_prev', loop_id, c_prev)
        ci, cf = self.actq2(self.eltwisemult_cell_input(i, g)), self.actq2(self.eltwisemult_cell_forget(f, c_prev))
        self.save_inner_data(save, prefix, 'ci', loop_id, ci)
        self.save_inner_data(save, prefix, 'cf', loop_id, cf)

        c = self.actq2(self.eltwiseadd_cell(cf, ci))
        h = self.actq1(self.eltwisemult_hidden(o, self.actq2(self.act_h(c))))
        self.save_inner_data(save, prefix, 'c', loop_id, c)
        self.save_inner_data(save, prefix, 'h', loop_id, h)

        self.save_inner_data(save, prefix, 'alpha1', 0, self.actq1.alpha)
        self.save_inner_data(save, prefix, 'alpha2', 0, self.actq2.alpha)
        self.save_inner_data(save, prefix, 'alpha3', 0, self.actq3.alpha)
        self.save_inner_data(save, prefix, 'alpha4', 0, self.actq4.alpha)

        return h, c


class DistillerLSTMCell(nn.Module):
    """todo: remove
    A single LSTM block.
    The calculation of the output takes into account the input and the previous output and cell state:
    https://pytorch.org/docs/stable/nn.html#lstmcell
    Args:
        input_size (int): the size of the input
        hidden_size (int): the size of the hidden state / output
        bias (bool): use bias. default: True
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(DistillerLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.eltwiseadd_gate = EltwiseAdd()
        # Apply activations separately:
        self.act_f = nn.Sigmoid()
        self.act_i = nn.Sigmoid()
        self.act_o = nn.Sigmoid()
        self.act_g = nn.Tanh()
        # Calculate cell:
        self.eltwisemult_cell_forget = EltwiseMult()
        self.eltwisemult_cell_input = EltwiseMult()
        self.eltwiseadd_cell = EltwiseAdd()
        # Calculate hidden:
        self.act_h = nn.Tanh()
        self.eltwisemult_hidden = EltwiseMult()
        self.init_weights()

    def forward(self, x, h=None):
        """
        Implemented as defined in https://pytorch.org/docs/stable/nn.html#lstmcell.
        """
        x_bsz, x_device = x.size(1), x.device
        if h is None:
            h = self.init_hidden(x_bsz, device=x_device)

        h_prev, c_prev = h
        fc_gate = self.eltwiseadd_gate(self.fc_gate_x(x), self.fc_gate_h(h_prev))
        i, f, g, o = torch.chunk(fc_gate, 4, dim=1)
        i, f, g, o = self.act_i(i), self.act_f(f), self.act_g(g), self.act_o(o)
        cf, ci = self.eltwisemult_cell_forget(f, c_prev), self.eltwisemult_cell_input(i, g)
        c = self.eltwiseadd_cell(cf, ci)
        h = self.eltwisemult_hidden(o, self.act_h(c))
        return h, c

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, c_0

    def init_weights(self):
        initrange = 1 / np.sqrt(self.hidden_size)
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)

    def to_pytorch_impl(self):
        module = nn.LSTMCell(self.input_size, self.hidden_size, self.bias)
        module.weight_hh, module.weight_ih = \
            nn.Parameter(self.fc_gate_h.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_x.weight.clone().detach())
        if self.bias:
            module.bias_hh, module.bias_ih = \
                nn.Parameter(self.fc_gate_h.bias.clone().detach()), \
                nn.Parameter(self.fc_gate_x.bias.clone().detach())
        return module

    @staticmethod
    def from_pytorch_impl(lstmcell: nn.LSTMCell):
        module = DistillerLSTMCell(input_size=lstmcell.input_size, hidden_size=lstmcell.hidden_size, bias=lstmcell.bias)
        module.fc_gate_x.weight = nn.Parameter(lstmcell.weight_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(lstmcell.weight_hh.clone().detach())
        if lstmcell.bias:
            module.fc_gate_x.bias = nn.Parameter(lstmcell.bias_ih.clone().detach())
            module.fc_gate_h.bias = nn.Parameter(lstmcell.bias_hh.clone().detach())

        return module

    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.input_size, self.hidden_size)

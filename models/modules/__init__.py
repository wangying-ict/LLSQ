from ._quan_base import *
from .quantize import *
from .eltwise import *
from .concat import *
from .upsample import *
from .yolo3 import *
from .bit_pruning import *
from .ttq import *
from .cluster_quant import *
from .shared_modules import *
from .bwn import *
from .lsq import *
from .llsq import *
from .svd import *
from .convbn_fold import *
from .rnn_q import *
from .activation import *


class QuantizationFactory(object):
    def __init__(self, quan_type, nbits_w, nbits_a, q_mode=Qmodes.layer_wise, signed=False, floor=False):
        """
        Default quantization parameters
        :param quan_type: 'lsq', ...
        :param nbits_w: 4,3,2,1
        :param nbits_a: 4,3,2,1
        :param mode: Qmodes.layer_wise, kernel_wise
        :param sign: False, True
        """
        super(QuantizationFactory, self).__init__()
        self.quan_type = quan_type
        self.nbits_a = nbits_a
        self.nbits_w = nbits_w
        self.q_mode = q_mode
        self.signed = signed
        self.floor = floor

    def product_Conv2dQ(self, in_channels, out_channels, kernel_size, stride=1,
                        padding=0, dilation=1, groups=1, bias=True,
                        quan_type=None, nbits=None, mode=None):
        if quan_type is None:
            quan_type = self.quan_type
        if nbits is None:
            nbits = self.nbits_w
        if mode is None:
            mode = self.q_mode
        if quan_type == 'lsq':
            return Conv2dLSQ(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                nbits=nbits, mode=mode)
        elif quan_type == 'llsq':
            return Conv2dLLSQ(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                nbits=nbits, mode=mode)
        elif quan_type == 'bwn':
            return Conv2dBWN(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                nbits=nbits, mode=mode
            )
        elif quan_type == 'bwns':
            return Conv2dBWNS(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                nbits=nbits, mode=mode
            )
        else:
            assert NotImplementedError

    def product_ActQ(self, quan_type=None, nbits=None, signed=None, floor=None):
        if quan_type is None:
            quan_type = self.quan_type
        if nbits is None:
            nbits = self.nbits_a
        if signed is None:
            signed = self.signed
        if floor is None:
            floor = self.floor
        if quan_type == 'lsq':
            return ActLSQ(nbits=nbits, signed=signed)
        elif quan_type == 'llsqs':
            return ActLLSQS(nbits=nbits, signed=signed, floor=floor)
        elif quan_type == 'llsq':
            return ActLLSQ(nbits=nbits, signed=signed)
        else:
            assert NotImplementedError

    def product_LinearQ(self, in_features, out_features, bias=True, quan_type=None, nbits=None):
        if quan_type is None:
            quan_type = self.quan_type
        if nbits is None:
            nbits = self.nbits_w
        if quan_type == 'lsq':
            return LinearLSQ(in_features, out_features, bias, nbits=nbits)
        elif quan_type == 'llsq':
            return LinearLLSQ(in_features, out_features, bias, nbits=nbits)
        elif quan_type == 'bwn':
            return LinearBWN(in_features, out_features, bias, nbits=nbits)
        elif quan_type == 'bwns':
            return LinearBWNS(in_features, out_features, bias, nbits=nbits)
        else:
            assert NotImplementedError

"""
    Copied from https://github.com/NervanaSystems/distiller/blob/master/jupyter/truncated_svd.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TSVDLinear', 'truncated_svd']


# See Faster-RCNN: https://github.com/rbgirshick/py-faster-rcnn/blob/master/tools/compress_net.py
# Replaced numpy operations with pytorch operations (so that we can leverage the GPU).
def truncated_svd(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    U, s, V = torch.svd(W, some=True)

    Ul = U[:, :l]
    sl = s[:l]
    V = V.t()
    Vl = V[:l, :]

    SV = torch.mm(torch.diag(sl), Vl)
    return Ul, SV


class TSVDLinear(nn.Linear):
    # truncated SVD linear
    def __init__(self, in_features, out_features, bias=True, preserve_ratio=0.5):
        super(TSVDLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.U = None
        self.SV = None
        self.preserve_ratio = preserve_ratio

    def forward(self, x):
        if self.U is None:
            k = int(self.weight.size(0) * self.preserve_ratio)
            self.U, self.SV = truncated_svd(self.weight, l=k)
            print('U shape: {}, SV shape: {}'.format(self.U.shape, self.SV.shape))
        x = F.linear(x, self.SV)
        x = F.linear(x, self.U, self.bias)
        out = x
        return out


class TruncatedSVD(nn.Module):
    def __init__(self, replaced_gemm, gemm_weights, preserve_ratio):
        super().__init__()
        self.replaced_gemm = replaced_gemm
        print("W = {}".format(gemm_weights.shape))
        self.U, self.SV = truncated_svd(gemm_weights.data, int(preserve_ratio * gemm_weights.size(0)))
        print("U = {}".format(self.U.shape))

        self.fc_u = nn.Linear(self.U.size(1), self.U.size(0)).cuda()
        self.fc_u.weight.data = self.U

        print("SV = {}".format(self.SV.shape))
        self.fc_sv = nn.Linear(self.SV.size(1), self.SV.size(0)).cuda()
        self.fc_sv.weight.data = self.SV  # .t()

    def forward(self, x):
        x = self.fc_sv.forward(x)
        x = self.fc_u.forward(x)
        return x

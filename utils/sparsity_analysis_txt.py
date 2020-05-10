'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import numpy as np
import argparse
import os
import math
import torch
import pandas
import ipdb

parser = argparse.ArgumentParser(description='Sparsity Analysis')
parser.add_argument('dir', metavar='DIR',
                    help='path to txt')
parser.add_argument('-c', '--complement', default=False, action='store_true',
                    help='use twos complement representation')


def main():
    args = parser.parse_args()
    assert os.path.exists(args.dir), '{} does not exist'.format(args.dir)
    if args.complement:
        print('Use twos complement representation')
    else:
        print('Use true form')
    fs = os.listdir(args.dir)
    sparse_weight = {}
    for f in fs:
        try:
            sparse_weight[f] = np.loadtxt(os.path.join(args.dir, f))
        except UnicodeDecodeError:
            sparse_weight[f] = np.load(os.path.join(args.dir, f))
    col_keys = ['Layer', 'Weight sparsity', 'Bit sparsity']
    data = []
    total_cnt = 0
    total_weight_cnt = 0
    total_bit_cnt = 0

    for k, v in sparse_weight.items():
        cnt_sum = v.size
        total_cnt += cnt_sum
        total_weight_cnt += (abs(v) > 0).sum().astype(np.float)
        bit_cnt = count_bit(v, complement=args.complement)
        total_bit_cnt += bit_cnt.sum().astype(np.float)
        weight_sparsity = 1 - (abs(v) > 0).sum().astype(np.float) / cnt_sum
        bit_sparsity = bit_sparse(bit_cnt, args.complement)
        data.append([k, '{:.3f}'.format(weight_sparsity), '{:.3f}'.format(bit_sparsity)])
    if args.complement:
        bit_width = 4
    else:
        bit_width = 3
    data.append(['total', '{:.3f}'.format(1 - total_weight_cnt / total_cnt),
                 '{:.3f}'.format(1 - total_bit_cnt / total_cnt / bit_width)])
    df = pandas.DataFrame(data=data, columns=col_keys)
    print(df)


def count_bit(w_int, complement=False):
    if complement:
        raise NotImplementedError
        # w_int = torch.where(w_int < 0, 256 + w_int, w_int).int().to(w_int.device)
        # bit_cnt = torch.zeros(w_int.shape).int().to(w_int.device)
        # for i in range(4):
        #     bit_cnt += w_int % 2
        #     w_int /= 2
    else:
        w_int = abs(w_int)
        bit_cnt = np.zeros(w_int.shape)
        for i in range(3):
            bit_cnt += w_int % 2
            w_int /= 2
    return bit_cnt


def bit_sparse(bit_cnt, complement=False):
    if complement:
        return 1 - bit_cnt.sum().astype(np.float) / (4 * bit_cnt.size)
    else:
        return 1 - bit_cnt.sum().astype(np.float) / (4 * bit_cnt.size)


if __name__ == '__main__':
    main()

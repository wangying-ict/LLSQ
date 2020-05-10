'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import math
import torch
import pandas
import numpy as np
import ipdb

from models.modules.bit_pruning import count_bit, truncation, bit_sparse

parser = argparse.ArgumentParser(description='Sparsity Analysis for activation')
parser.add_argument('dir', metavar='DIR',
                    help='path to pth')
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
    activations = {}
    for f in fs:
        if '.npy' in f:
            activations[f] = torch.from_numpy(np.load(os.path.join(args.dir, f)))
    col_keys = ['Layer', 'Activation sparsity', 'Bit sparsity']

    data = []
    # total_cnt = 0
    total_conv_cnt = 0
    total_weight_cnt = 0
    total_weight_conv_cnt = 0
    total_bit_cnt = 0
    total_bit_conv_cnt = 0

    instance_bs_dict = {}

    batch_size = -1
    for k, v in activations.items():
        batch_size = v.shape[0]
        v_reshape = v.view(batch_size, -1)
        il = torch.log2(v_reshape.max(1)[0].sort()[0][int(batch_size / 3)]) + 1
        il = math.ceil(il - 1e-5)
        radix_position = 8 - il
        print(radix_position)
        radix_position = 7
        _, value_int = truncation(v, radix_position)
        cnt_sum = v.view(-1).shape[0]
        # total_cnt += cnt_sum
        total_weight_cnt += (value_int.float().abs() > 0).sum().float()
        bit_cnt = count_bit(value_int, complement=args.complement)
        total_bit_cnt += bit_cnt.sum().float()
        value_sparsity = 1 - (v_reshape.float().abs() > 0).sum().float() / cnt_sum
        bit_sparsity = bit_sparse(bit_cnt, args.complement)
        instance_bs = []
        for i in range(batch_size):
            instance_bs.append(bit_sparse(bit_cnt[i], args.complement).item())
        instance_bs_dict[k] = instance_bs

        total_conv_cnt += cnt_sum
        total_weight_conv_cnt += (value_int.float().abs() > 0).sum().float()
        total_bit_conv_cnt += bit_cnt.sum().float()
        data.append([k, '{:.3f}'.format(value_sparsity), '{:.3f}'.format(bit_sparsity)])

    pandas.set_option('display.width', 5000)
    df = pandas.DataFrame(data=data, columns=col_keys)
    print(df)
    instance_data = []
    instance_keys = ['instance_id']
    for k, v in instance_bs_dict.items():
        instance_keys.append(k)
    for i in range(batch_size):
        in_data = [i]
        for layer_id in instance_keys[1:]:
            in_data.append('{:.3f}'.format(instance_bs_dict[layer_id][i]))
        instance_data.append(in_data)
    instance_df = pandas.DataFrame(data=instance_data, columns=instance_keys)
    # print(instance_df)
    instance_df.to_csv(os.path.join(args.dir, 'act_bs_analysis.csv'), index=None)
    print('act_bs_analysis.csv has been saved in {}'.format(args.dir))


if __name__ == '__main__':
    main()

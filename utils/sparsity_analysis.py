'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import math
import torch
import pandas
import ipdb

from models.modules.bit_pruning import count_bit, truncation, bit_sparse, bit_pruning_int_truncation

flops_files = ['alexnet_flops.txt', 'resnet18_flops.txt', 'resnet50_flops.txt', 'mobilenet_distiller_flops.txt']

parser = argparse.ArgumentParser(description='Sparsity Analysis')
parser.add_argument('pth', metavar='FILE',
                    help='path to pth')
parser.add_argument('-c', '--complement', default=False, action='store_true',
                    help='use twos complement representation')
parser.add_argument('-p', '--bit-pruning', default=False, action='store_true',
                    help='bit pruning by using method2')
parser.add_argument('-k', '--array-size', type=int, default=8, help='PE array size')


def main():
    args = parser.parse_args()
    assert os.path.exists(args.pth), '{} does not exist'.format(args.pth)
    if args.complement:
        print('Use twos complement representation')
    else:
        print('Use true form')
    sparse_weight = torch.load(args.pth, map_location='cpu')
    try:
        sparse_weight = sparse_weight['state_dict']
    except:
        pass
    col_keys = ['Layer', 'Weight sparsity', 'Bit sparsity', '0', '1', '2', '3', '4', '5', '6', '7', 'ave_cycle', '#MAC']
    data = []
    total_cnt = 0
    total_conv_cnt = 0
    total_weight_cnt = 0
    total_weight_conv_cnt = 0
    total_bit_cnt = 0
    total_bit_conv_cnt = 0
    bit_distribute_sum = [0 for i in range(8)]
    bit_distribute_sum_conv = [0 for i in range(8)]
    total_mac_conv = 0
    total_mac = 0
    ave_cycle_conv = 0
    ave_cycle_total = 0
    has_int = False
    for k, v in sparse_weight.items():
        if 'weight_int' in k:
            has_int = True
    if 'alexnet' in args.pth:
        file_name = flops_files[0]
    elif 'resnet18' in args.pth:
        file_name = flops_files[1]
    elif 'resnet50' in args.pth:
        file_name = flops_files[2]
    elif 'mobile' in args.pth:
        file_name = flops_files[3]
    else:
        file_name = None
    mac_str = []
    if file_name is not None:
        with open(os.path.join('utils', file_name), 'r') as rf:
            for line in rf:
                mac_str.append(line.strip('\n'))
    for k, v in sparse_weight.items():
        if has_int:
            if 'weight_int' in k:
                if file_name is not None:
                    mac = get_mac(mac_str, k)
                else:
                    mac = 0
                cnt_sum = v.view(-1).shape[0]
                total_cnt += cnt_sum
                total_weight_cnt += (v.float().abs() > 0).sum().float()
                bit_cnt = count_bit(v, complement=args.complement)
                weight_sparsity = 1 - (v.float().abs() > 0).sum().float() / cnt_sum
                bit_sparsity = bit_sparse(bit_cnt, args.complement)
                bit_distribute = []
                for i in range(8):
                    special_bit_cnt = (bit_cnt > (i - 1)).sum().item() - (bit_cnt > i).sum().item()
                    bit_distribute_sum[i] += special_bit_cnt
                    if len(v.shape) == 4:
                        bit_distribute_sum_conv[i] += special_bit_cnt
                    bit_distribute.append(special_bit_cnt / bit_cnt.view(-1).shape[0])
                total_bit_cnt += bit_cnt.sum().float()
                ave_cycle = get_ave_cycle(bit_distribute, args.array_size)
                total_mac += mac
                ave_cycle_total += mac * ave_cycle
                if len(v.shape) == 4:
                    total_conv_cnt += cnt_sum
                    total_weight_conv_cnt += (v.float().abs() > 0).sum().float()
                    total_bit_conv_cnt += bit_cnt.sum().float()
                    total_mac_conv += mac
                    ave_cycle_conv += mac * ave_cycle
                data.append(
                    [k, '{:.2f}'.format(weight_sparsity * 100), '{:.2f}'.format(bit_sparsity * 100)] + [
                        "{:.2f}".format(i * 100) for i in bit_distribute] +
                    ['{:.3f}'.format(get_ave_cycle(bit_distribute, args.array_size)), '{:.3f}'.format(mac)])
        elif 'weight' in k and (len(v.shape) == 4 or len(v.shape) == 2):
            if file_name is not None:
                mac = get_mac(mac_str, k)
            else:
                mac = 0
            il = torch.log2(torch.max(v.max(), v.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            radix_position = 8 - il
            _, weight_int = truncation(v, radix_position)
            cnt_sum = weight_int.view(-1).shape[0]
            total_cnt += cnt_sum
            total_weight_cnt += (weight_int.float().abs() > 0).sum().float()
            if args.bit_pruning:
                weight_int = bit_pruning_int_truncation(weight_int)
            bit_cnt = count_bit(weight_int, args.complement)
            bit_distribute = []
            for i in range(8):
                special_bit_cnt = (bit_cnt > (i - 1)).sum().item() - (bit_cnt > i).sum().item()
                bit_distribute_sum[i] += special_bit_cnt
                if len(v.shape) == 4:
                    bit_distribute_sum_conv[i] += special_bit_cnt
                bit_distribute.append(special_bit_cnt / bit_cnt.view(-1).shape[0])
            total_bit_cnt += bit_cnt.sum().float()
            weight_sparsity = 1 - (weight_int.float().abs() > 0).sum().float() / cnt_sum
            bit_sparsity = bit_sparse(bit_cnt, args.complement)

            total_mac += mac
            ave_cycle = get_ave_cycle(bit_distribute, args.array_size)
            ave_cycle_total += mac * ave_cycle

            if len(v.shape) == 4:
                total_conv_cnt += cnt_sum
                total_weight_conv_cnt += (weight_int.float().abs() > 0).sum().float()
                total_bit_conv_cnt += bit_cnt.sum().float()

                total_mac_conv += mac
                ave_cycle_conv += mac * ave_cycle
            data.append(
                [k, '{:.2f}'.format(weight_sparsity * 100), '{:.2f}'.format(bit_sparsity * 100)] + [
                    "{:.2f}".format(i * 100) for i in bit_distribute] + [
                    '{:.3f}'.format(ave_cycle), '{:.3f}'.format(mac)])
    if args.complement:
        bit_width = 8
    else:
        bit_width = 7
    tmp_sum = sum(bit_distribute_sum_conv)
    bit_distribute_sum_conv = [float(i) / tmp_sum for i in bit_distribute_sum_conv]
    bit_distribute_sum_conv = ["{:.2f}".format(i * 100) for i in bit_distribute_sum_conv]
    tmp_sum = sum(bit_distribute_sum)
    bit_distribute_sum = [float(i) / tmp_sum for i in bit_distribute_sum]
    bit_distribute_sum = ["{:.2f}".format(i * 100) for i in bit_distribute_sum]
    data.append(['total_conv', '{:.2f}'.format((1 - total_weight_conv_cnt / total_conv_cnt) * 100),
                 '{:.2f}'.format(
                     (1 - total_bit_conv_cnt / total_conv_cnt / bit_width) * 100)] + bit_distribute_sum_conv + [
                    ave_cycle_conv, total_mac_conv
                ])
    data.append(['total', '{:.2f}'.format((1 - total_weight_cnt / total_cnt) * 100),
                 '{:.2f}'.format(100 * (1 - total_bit_cnt / total_cnt / bit_width))] + bit_distribute_sum + [
                    ave_cycle_total, total_mac
                ])
    pandas.set_option('display.width', 5000)
    df = pandas.DataFrame(data=data, columns=col_keys)
    print(df)
    df.to_csv('{}.csv'.format(args.pth), index=None)
    print('{} have been saved'.format(args.pth))
    # simulate the inference cycle
    # suppose p6, p7 = 0
    # p0, p1, p2, p3, p4 != 0
    # p1 = p1 / ( 1 - p0)
    # p2 = p2 / ( 1 - p0)
    # p3 = p3 / ( 1 - p0)
    # p4 = p4 / ( 1 - p0)
    # simulated cycle = 4 - 2 p1 ^ n - (p1 + p2)^n - (p1 + p2 + p3) ^ n

    # if 'alexnet' in args.pth:
    #     bit_distribute_float = [float(i) for i in bit_distribute_sum_conv]
    # else:
    #     bit_distribute_float = [float(i) for i in bit_distribute_sum]

    # col_keys_cycle = ['parallel number', 'cycle']

    # df2 = pandas.DataFrame(data=data_cycle, columns=col_keys_cycle)
    # print(df2)
    # df2.to_csv('{}.csv'.format(args.pth), index=None)
    # print('{} have been saved'.format(args.pth))


def get_ave_cycle(bit_distribution, k=8, fraction=1 / 3):
    k = int(k)
    p0 = bit_distribution[0]
    p1 = bit_distribution[1]
    p2 = bit_distribution[2]
    p3 = bit_distribution[3]
    p4 = bit_distribution[4]
    p5 = bit_distribution[5]
    # fraction = 1  # 1/3 of zero weights are involved in bit serial unit.
    total = 1 - p0 * (1 - fraction)
    p0 = p0 * fraction / total
    p1 = p1 / total
    p2 = p2 / total
    p3 = p3 / total
    p4 = p4 / total
    p5 = p5 / total

    P0 = p0 ** k
    P1 = (p0 + p1) ** k - P0
    P2 = (p0 + p1 + p2) ** k - P0 - P1
    P3 = (p0 + p1 + p2 + p3) ** k - P0 - P1 - P2
    P4 = (p0 + p1 + p2 + p3 + p4) ** k - P0 - P1 - P2 - P3
    P5 = 1 - P0 - P1 - P2 - P3 - P4
    cycle_ave = P0 * 0 + P1 * 1 + P2 * 2 + P3 * 3 + P4 * 4 + P5 * 5
    cycle_ave = cycle_ave * total  # the zero weight can be skipped by weight sparse code.
    # cycle = 4 - p1 ** n - (p1 + p2) ** n - (p1 + p2 + p3) ** n
    return cycle_ave


def get_mac(mac_str, k):
    keys = k.split('.')[:-1]
    if keys[0] == 'module':
        keys = keys[1:]
    current_point = 0
    # print(keys)
    for key in keys:
        current_point = find_str(key, mac_str, current_point + 1)
    start = mac_str[current_point].find('GMac') + 6
    end = mac_str[current_point].find('MACs') - 2
    # print(mac_str[current_point])
    mac = float(mac_str[current_point][start: end]) / 100
    return mac


def find_str(key, str_list, current_p):
    key = '(' + key + ')'
    for i in range(current_p, len(str_list)):
        if key in str_list[i]:
            return i
    raise RuntimeError


if __name__ == '__main__':
    main()

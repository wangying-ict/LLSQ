# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu
# Adopted from https://github.com/mit-han-lab/haq-release/blob/master/lib/utils/quantize_utils.py
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import ipdb

__all__ = ['Conv2dClusterQ', 'Conv2dShareQ', 'ActShareQ']


class FuncKmeansSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, centroids, labels):
        weight = reconstruct_weight_from_k_means_result(centroids, labels)
        ctx.save_for_backward(centroids, labels)
        return weight

    @staticmethod
    def backward(ctx, grad_weight):
        centroids, labels = ctx.saved_tensors
        grad_weight_ = torch.zeros_like(grad_weight)
        grad_centroids = torch.zeros_like(centroids)
        num_centroids = centroids.size(0)
        for j in range(num_centroids):
            mask_cl = (labels == j).float()
            grad_weight_ += (grad_weight * mask_cl).sum() / mask_cl.sum() * mask_cl
            grad_centroids[j] += (grad_weight * mask_cl).sum() / mask_cl.sum()

        return grad_weight_, grad_centroids, None


class FuncKmeansActSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, centroids):
        centroid_sort, index = centroids.sort()
        centroid_thresh = [-10000] + [(centroid_sort[i] + centroid_sort[i + 1]) / 2
                                      for i in range(len(centroid_sort) - 1)] + [10000]
        label = torch.zeros_like(activation)
        for i in range(len(centroid_sort)):
            case = (centroid_thresh[i] <= activation) * (activation < centroid_thresh[i + 1])
            activation = torch.where(case, torch.zeros_like(activation).fill_(centroid_sort[i]), activation)
            label = torch.where(case, torch.zeros_like(activation).fill_(index[i]), label)
        ctx.save_for_backward(centroids, label)
        return activation

    @staticmethod
    def backward(ctx, grad_act):
        centroids, labels = ctx.saved_tensors
        # grad_act_ = torch.zeros_like(grad_act)
        grad_centroids = torch.zeros_like(centroids)
        num_centroids = centroids.size(0)
        for j in range(num_centroids):
            mask_cl = (labels == j).float()
            # grad_act_ += (grad_act * mask_cl).sum() / mask_cl.sum() * mask_cl
            grad_centroids[j] += (grad_act * mask_cl).sum() / mask_cl.sum()

        return grad_act, grad_centroids


class Conv2dClusterQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4, fix_zero=True):
        super(Conv2dClusterQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                             padding=padding, dilation=dilation, groups=groups, bias=bias)
        if nbits < 0:
            self.register_buffer('centroids', None)
            self.register_buffer('labels', None)
            return
        self.nbits = nbits
        self.mode = 'cpu'
        self.centroids = nn.Parameter(torch.zeros(2 ** self.nbits))
        self.register_buffer('labels', torch.zeros_like(self.weight))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        w = self.weight.data
        if self.init_state == 0:
            if self.training and self.mode == 'cpu':
                with torch.no_grad():
                    centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** self.nbits)
                    self.centroids.copy_(centroids)
                    self.labels.copy_(labels)
                    self.weight.data.copy_(reconstruct_weight_from_k_means_result(centroids, labels))
            else:
                raise NotImplementedError
            self.init_state.fill_(1)
        weight = FuncKmeansSTE.apply(self.weight, self.centroids, self.labels)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dClusterQ, self).extra_repr()
        if self.nbits is None:
            return '{}, fake'.format(s_prefix)
        return '{}, nbits={}'.format(
            s_prefix, self.nbits)


class ActShareQ(nn.Module):
    def __init__(self, nbits=4, share_num=2):
        super(ActShareQ, self).__init__()
        if nbits < 0:
            self.nbits = nbits
            self.register_buffer('centroids', None)
            return
        self.nbits = nbits
        self.share_num = share_num
        self.mode = 'cpu'
        self.centroids = nn.Parameter(torch.zeros(2 ** self.nbits))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.nbits < 0:
            return input
        if self.init_state == 0 and self.training:
            if self.mode == 'cpu':
                with torch.no_grad():
                    input_cat = []
                    for i in range(self.share_num):
                        input_cat.append(input[i])
                    input_cat = torch.cat(input_cat)
                    centroids, labels = k_means_cpu(input_cat.cpu().numpy(), 2 ** self.nbits)
                    self.centroids.copy_(centroids)
                    # input = reconstruct_weight_from_k_means_result(centroids, labels)
            else:
                raise NotImplementedError
            self.init_state.fill_(1)
        input_cat = []
        for i in range(self.share_num):
            input_cat.append(input[i])
        input_cat = torch.cat(input_cat)
        input_q = FuncKmeansActSTE.apply(input_cat, self.centroids)
        split = int(input_cat.size(0) / self.share_num)
        ret = []
        for i in range(self.share_num):
            ret.append(input_q[split * i:split * (i + 1), :, :, :])
        return ret

    def extra_repr(self):
        s_prefix = super(ActShareQ, self).extra_repr()
        return '{},bits={}, share_num={}'.format(
            s_prefix, self.nbits, self.share_num)


class Conv2dShareQ(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4, share_num=2):
        super(Conv2dShareQ, self).__init__()
        self.share_num = share_num
        self.convs = nn.ModuleList()
        for i in range(share_num):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias))
        if nbits < 0:
            self.nbits = nbits
            self.register_buffer('centroids', None)
            self.register_buffer('labels', None)
            return
        self.nbits = nbits
        self.mode = 'cpu'
        self.centroids = nn.Parameter(torch.zeros(2 ** self.nbits))
        weight = []
        for i in range(share_num):
            weight.append(self.convs[i].weight)
        self.register_buffer('labels', torch.zeros_like(torch.cat(weight)))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.nbits < 0:
            ret = []
            for i in range(self.share_num):
                ret.append(F.conv2d(input[i], self.convs[i].weight, self.convs[i].bias, self.convs[i].stride,
                                    self.conv[i].padding, self.convs[i].dilation, self.convs[i].groups))
            return ret
        if self.init_state == 0 and self.training:
            if self.mode == 'cpu':
                with torch.no_grad():
                    weight = []
                    for i in range(self.share_num):
                        weight.append(self.convs[i].weight.data)
                    weight = torch.cat(weight)
                    centroids, labels = k_means_cpu(weight.cpu().numpy(), 2 ** self.nbits)
                    self.centroids.copy_(centroids)
                    self.labels.copy_(labels)
                    wq = reconstruct_weight_from_k_means_result(centroids, labels)
                    split = int(weight.size(0) / self.share_num)
                    for i in range(self.share_num):
                        wqi = wq[i * split: (i + 1) * split, :, :, :]
                        self.convs[i].weight.data.copy_(wqi)
            else:
                raise NotImplementedError
            self.init_state.fill_(1)
        weight = []
        for i in range(self.share_num):
            weight.append(self.convs[i].weight.data)
        weight = torch.cat(weight)
        weight_q = FuncKmeansSTE.apply(weight, self.centroids, self.labels)
        ret = []
        split = int(weight.size(0) / self.share_num)
        for i in range(self.share_num):
            wqi = weight_q[i * split: (i + 1) * split, :, :, :]
            self.convs[i].weight.data.copy_(wqi)
            ret.append(
                F.conv2d(input[0], wqi, self.convs[i].bias, self.convs[i].stride,
                         self.convs[i].padding, self.convs[i].dilation, self.convs[i].groups)
            )
        return ret

    def extra_repr(self):
        s_prefix = super(Conv2dShareQ, self).extra_repr()
        return '{}, nbits={}, share_num={}'.format(
            s_prefix, self.nbits, self.share_num)


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).cuda().view(-1), torch.from_numpy(labels).int().cuda()


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


def kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=False):
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        new_weight_data = layer.weight.data.clone()
        new_weight_data.zero_()
        this_cl_list = centroid_label_dict[i]
        num_centroids = this_cl_list[0][0].numel()
        if num_centroids > 2 ** 6 and free_high_bit:
            # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
            continue
        for j in range(num_centroids):
            mask_cl = (this_cl_list[0][1] == j).float()
            new_weight_data += (layer.weight.data * mask_cl).sum() / mask_cl.sum() * mask_cl
        layer.weight.data = new_weight_data

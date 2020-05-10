#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# WeightedEltwiseAdd is added by Joey.Z for qfi models.

import torch.nn as nn
import ipdb

__all__ = ['EltwiseAdd', 'EltwiseMult', 'WeightedEltwiseAdd']


class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class WeightedEltwiseAdd(nn.Module):
    def __init__(self, inplace=False, weight=None):
        super(WeightedEltwiseAdd, self).__init__()
        if weight is None:
            weight = [1, 1]
        self.inplace = inplace
        self.weight = weight

    def forward(self, *input):
        assert len(input) == len(self.weight), 'Weight length should be the same as input in WeightedEltwiseAdd'
        res = self.weight[0] * input[0]
        if self.inplace:
            for i, t in enumerate(input[1:]):
                res += self.weight[i + 1] * t
        else:
            for i, t in enumerate(input[1:]):
                res = res + self.weight[i + 1] * t
        return res

    def extra_repr(self):
        return '{}'.format(self.weight)


class EltwiseMult(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseMult, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res

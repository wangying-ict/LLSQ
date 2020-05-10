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

from math import floor
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.modules import Conv2dBP, LinearBP, Conv2dBPv2, LinearBPv2

model_urls = {
    'mobilenet': 'https://download.pytorch.org/models/mobilenet-distiller-sparsity.pth',
}
__all__ = ['mobilenet_distiller', 'mobilenet_025', 'mobilenet_050', 'mobilenet_075',
           'mobilenet_distiller_bp_exp', 'mobilenet_distiller_bp_exp_v2']


class MobileNet(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8):
        super(MobileNet, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                nn.Conv2d(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(self.channels[5], 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x


class MobileNet_BP_EXP(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8, nbits_w=8, increase_factor=1 / 3):
        super(MobileNet_BP_EXP, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                Conv2dBP(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False,
                         nbits=nbits_w, increase_factor=increase_factor),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7),
        )
        self.fc = LinearBP(self.channels[5], 1000, nbits=nbits_w, increase_factor=increase_factor / 2)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x


class MobileNet_BP_EXP_v2(nn.Module):
    def __init__(self, channel_multiplier=1.0, min_channels=8, nbits_w=8, log=False):
        super(MobileNet_BP_EXP_v2, self).__init__()

        if channel_multiplier <= 0:
            raise ValueError('channel_multiplier must be >= 0')

        def conv_bn_relu(n_ifm, n_ofm, kernel_size, stride=1, padding=0, groups=1):
            return [
                Conv2dBPv2(n_ifm, n_ofm, kernel_size, stride=stride, padding=padding, groups=groups, bias=False,
                           nbits=nbits_w, log=log),
                nn.BatchNorm2d(n_ofm),
                nn.ReLU(inplace=True)
            ]

        def depthwise_conv(n_ifm, n_ofm, stride):
            return nn.Sequential(
                *conv_bn_relu(n_ifm, n_ifm, 3, stride=stride, padding=1, groups=n_ifm),
                *conv_bn_relu(n_ifm, n_ofm, 1, stride=1)
            )

        base_channels = [32, 64, 128, 256, 512, 1024]
        self.channels = [max(floor(n * channel_multiplier), min_channels) for n in base_channels]

        self.model = nn.Sequential(
            nn.Sequential(*conv_bn_relu(3, self.channels[0], 3, stride=2, padding=1)),
            depthwise_conv(self.channels[0], self.channels[1], 1),
            depthwise_conv(self.channels[1], self.channels[2], 2),
            depthwise_conv(self.channels[2], self.channels[2], 1),
            depthwise_conv(self.channels[2], self.channels[3], 2),
            depthwise_conv(self.channels[3], self.channels[3], 1),
            depthwise_conv(self.channels[3], self.channels[4], 2),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[4], 1),
            depthwise_conv(self.channels[4], self.channels[5], 2),
            depthwise_conv(self.channels[5], self.channels[5], 1),
            nn.AvgPool2d(7),
        )
        self.fc = LinearBPv2(self.channels[5], 1000, nbits=nbits_w, log=log)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.channels[-1])
        x = self.fc(x)
        return x


def mobilenet_025():
    return MobileNet(channel_multiplier=0.25)


def mobilenet_050():
    return MobileNet(channel_multiplier=0.5)


def mobilenet_075():
    return MobileNet(channel_multiplier=0.75)


def mobilenet_distiller(pretrained=False, **kwargs):
    model = MobileNet(**kwargs)
    if pretrained:
        sd = model_zoo.load_url(model_urls['mobilenet'], map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
    return model


def mobilenet_distiller_bp_exp(pretrained=False, **kwargs):
    model = MobileNet_BP_EXP(**kwargs)
    if pretrained:
        sd = model_zoo.load_url(model_urls['mobilenet'], map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model


def mobilenet_distiller_bp_exp_v2(pretrained=False, **kwargs):
    model = MobileNet_BP_EXP_v2(**kwargs)
    if pretrained:
        sd = model_zoo.load_url(model_urls['mobilenet'], map_location='cpu')
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model

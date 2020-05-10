"""
    VGG small of PyTorch.
    Create by Joey.Z.
    Reference: https://github.com/Microsoft/LQ-Nets/blob/master/cifar10-vgg-small.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.cifar10 import load_fake_quantized_state_dict
from models.modules import ActQ, Conv2dQ, LinearQ, PACT, Conv2dQv2, ActQv2, LinearQv2
from models.modules import Qmodes

"""
    cifar10_vgg{}_q: don't quantize the first and last layer just like LQ-Net
    cifar10_vgg{}_qfn(q-full-naive): quantize the whole model including the first and last layer using naive method.
    cifar10_vgg{}_qfi(q-full-improve): quantize the whole model including the first and last layer using improved method.
"""

__all__ = [
    'VGGQ', 'cifar10_vggsmall_q', 'cifar10_vggsmall_qfn', 'cifar10_vggsmall_qfnv2', 'cifar10_vggsmall_qfi',
    'cifar10_vggsmall_qfn_pact', 'cifar10_vggsmall_qv2'
]

model_urls = {
    'vgg7': 'https://fake/models/cifar10_vggsmall-zxd-fiefefe.pth',
}


def cifar10_vggsmall_qv2(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQv2('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7']), 'cifar10_vggsmall_qv2_map.json')
    return model


def cifar10_vggsmall_q(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQ('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7']), 'cifar10_vggsmall_q_map.json')
    return model


def cifar10_vggsmall_qfn(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQFN('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7'], map_location='cpu'),
                                       'cifar10_vggsmall_qfn_map.json')
    return model


def cifar10_vggsmall_qfnv2(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQFNv2('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7'], map_location='cpu'),
                                       'cifar10_vggsmall_qfnv2_map.json')
    return model


def cifar10_vggsmall_qfn_pact(pretrained=False, **kwargs):
    model = VGGQFN_PACT('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7'], map_location='cpu'),
                                       'cifar10_vggsmall_qfn_pact_map.json')
    return model


def cifar10_vggsmall_qfi(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQFI('VGG7Q', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['vgg7'], map_location='cpu'),
                                       'cifar10_vggsmall_qfi_map.json')
    return model


cfg = {
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG7Q': [128, 128, 'M', 256, 256, 'M', 512],
}


class VGGQv2(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQv2, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w,
                                          nbits_a=nbits_a, q_mode=q_mode)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            nn.Linear(512 * scale, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3

        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # do not quantize first layer
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQv2(nbits=nbits_a, l2=self.l2)]
                in_channels = x
            else:
                layers += [Conv2dQv2(in_channels, x, kernel_size=3, padding=1, bias=False,
                                     nbits=nbits_w, mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQv2(nbits=nbits_a, l2=self.l2)]
                in_channels = x
        return nn.Sequential(*layers)


class VGGQ(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQ, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w,
                                          nbits_a=nbits_a, q_mode=q_mode)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            nn.Linear(512 * scale, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3

        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # do not quantize first layer
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
                in_channels = x
            else:
                layers += [Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode,
                                   l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
                in_channels = x
        return nn.Sequential(*layers)


class VGGQFN(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQFN, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a, q_mode=q_mode)
        # self.last_actq = ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else nbits_a * 2, l2=self.l2)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            LinearQ(512 * scale, 10, nbits=nbits_w, l2=l2),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.last_actq(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3
        first_bits = -1 if max(nbits_a, nbits_w) <= 0 else 8
        layers += [ActQ(nbits=first_bits, signed=True, l2=self.l2)]
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w,
                                   mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
                in_channels = x
            elif i == 7:  # last layer
                layers += [Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode,
                                   l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
            else:
                layers += [Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
                in_channels = x
        return nn.Sequential(*layers)


class VGGQFN_PACT(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQFN_PACT, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a, q_mode=q_mode)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            PACT(nbits=nbits_a, inplace=False),
            LinearQ(512 * scale, 10, nbits=nbits_w, l2=l2), )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.last_actq(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3
        # change to actq+convq by Joey.Z on May 28 2019
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [
                    ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True, l2=self.l2),
                    Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                            nbits=nbits_w,
                            mode=q_mode, l2=self.l2),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True), ]
                in_channels = x
            elif i == 7:  # last layer
                layers += [PACT(nbits=nbits_a),
                           Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode,
                                   l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
            else:
                layers += [PACT(nbits=nbits_a),
                           Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
        return nn.Sequential(*layers)


class VGGQFNv2(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQFNv2, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a, q_mode=q_mode)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            ActQv2(nbits=nbits_a, l2=l2),
            LinearQv2(512 * scale, 10, nbits=nbits_w, l2=l2),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.last_actq(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3
        # change to actq+convq by Joey.Z on May 28 2019
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [
                    ActQv2(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True, l2=self.l2),
                    Conv2dQv2(in_channels, x, kernel_size=3, padding=1, bias=False,
                              nbits=nbits_w,
                              mode=q_mode, l2=self.l2),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True), ]
                in_channels = x
            elif i == 7:  # last layer
                layers += [ActQv2(nbits=nbits_a, l2=self.l2),
                           Conv2dQv2(in_channels, x, kernel_size=3, padding=1, bias=False,
                                     nbits=nbits_w, mode=q_mode,
                                     l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
            else:
                layers += [ActQv2(nbits=nbits_a, l2=self.l2),
                           Conv2dQv2(in_channels, x, kernel_size=3, padding=1, bias=False,
                                     nbits=nbits_w, mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
        return nn.Sequential(*layers)


class VGGQFI(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(VGGQFI, self).__init__()
        self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a, q_mode=q_mode)
        self.last_features = nn.Sequential(
            Conv2dQ(512, 512, kernel_size=3, padding=1, bias=False, nbits=nbits_w, mode=q_mode, l2=self.l2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        scale = 1
        if vgg_name == 'VGG7Q':
            scale = 16
        self.expand_classifier = nn.Sequential(
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, expand=True, l2=self.l2),
            LinearQ(512 * scale * 2, 10, nbits=nbits_w, l2=self.l2),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.last_features(out)
        out = out.view(out.size(0), -1)
        out = self.expand_classifier(out)
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a, q_mode):
        layers = []
        in_channels = 3
        first_bits = -1 if max(nbits_a, nbits_w) <= 0 else 8
        layers += [ActQ(nbits=first_bits, signed=True)]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2dQ(in_channels, x, kernel_size=3, padding=1, bias=False,
                                   nbits=nbits_w, mode=q_mode, l2=self.l2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ActQ(nbits=nbits_a, l2=self.l2)]
                in_channels = x
        return nn.Sequential(*layers)

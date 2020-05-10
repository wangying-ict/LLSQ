import math
import sys

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import models.modules as my_nn
from models.modules import ActQ, Conv2dQ, LinearQ, Qmodes, ActQv2, Conv2dQv2, LinearQv2
from models.imagenet import load_fake_quantized_state_dict
from models import load_pre_state_dict
import ipdb

# TODO: re-structure
__all__ = ['ResNetQ',
           'resnet18_lsq',
           'resnet18_llsq',
           'resnet18_q', 'resnet18_qfn', 'resnet18_qfi',
           'resnet34_q',
           'resnet50_q', 'resnet50_qfn',
           'resnet18_qv2', 'resnet18_qfnv2',
           'resnet34_qv2', 'resnet34_qfnv2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18_lsq(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _ResNetQ(_BasicBlockQ, [2, 2, 2, 2], qf=quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['resnet18']),
                            '{}_map.json'.format(model_name))
    return model


def resnet18_qv2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQv2(BasicBlockQv2, [2, 2, 2, 2], **kwargs)
    model_name = 'resnet18_lsq'
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['resnet18']),
                            '{}_map.json'.format(model_name))
    return model


def resnet18_qfnv2(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFNv2(BasicBlockQv2, [2, 2, 2, 2], **kwargs)
    model_name = resnet18_lsq
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['resnet18']),
                            '{}_map.json'.format(model_name))
    return model


def resnet18_llsq(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _ResNetQ(_BasicBlockQ, [2, 2, 2, 2], qf=quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['resnet18']),
                            '{}_map.json'.format(model_name))
    return model


class _ResNetQ(nn.Module):

    def __init__(self, block, layers, num_classes=1000, qf=None):
        self.inplanes = 64
        super(_ResNetQ, self).__init__()
        # We don't quantize first layer
        self.qf = qf
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.qf.product_Conv2dQ(self.inplanes, planes * block.expansion,
                                        kernel_size=1, stride=stride, bias=False, ),
                nn.BatchNorm2d(planes * block.expansion),
                self.qf.product_ActQ(signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, qf=self.qf))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, qf=self.qf))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class _BasicBlockQ(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, qf=None):
        super(_BasicBlockQ, self).__init__()
        self.conv1 = nn.Sequential(_convq3x3(inplanes, planes, stride, qf=qf),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True),
                                   qf.product_ActQ())
        self.conv2 = nn.Sequential(_convq3x3(planes, planes, qf=qf),
                                   nn.BatchNorm2d(planes),
                                   qf.product_ActQ(signed=True))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.out_actq = qf.product_ActQ()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out


def _convq3x3(in_planes, out_planes, stride=1, qf=None):
    """3x3 convolution with padding"""
    return qf.product_Conv2dQ(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)


def convq3x3(in_planes, out_planes, stride=1, nbits_w=4, q_mode=Qmodes.kernel_wise, l2=True):
    """3x3 convolution with padding"""
    return Conv2dQ(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, nbits=nbits_w, mode=q_mode, l2=l2)


def convqv2_3x3(in_planes, out_planes, stride=1, nbits_w=4, q_mode=Qmodes.kernel_wise, l2=True):
    """3x3 convolution with padding"""
    return Conv2dQv2(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, nbits=nbits_w, mode=q_mode, l2=l2)


class BasicBlockQ(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise,
                 l2=True):
        super(BasicBlockQ, self).__init__()
        self.conv1 = nn.Sequential(convq3x3(inplanes, planes, stride, nbits_w=nbits_w, q_mode=q_mode, l2=l2),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True),
                                   ActQ(nbits=nbits_a, l2=l2))
        self.conv2 = nn.Sequential(convq3x3(planes, planes, nbits_w=nbits_w, q_mode=q_mode, l2=l2),
                                   nn.BatchNorm2d(planes),
                                   ActQ(nbits=nbits_a, l2=l2))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.out_actq = ActQ(nbits=nbits_a, l2=l2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out


class BasicBlockQv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise,
                 l2=True):
        super(BasicBlockQv2, self).__init__()
        self.conv1 = nn.Sequential(convqv2_3x3(inplanes, planes, stride, nbits_w=nbits_w, q_mode=q_mode, l2=l2),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True),
                                   ActQv2(nbits=nbits_a, l2=l2))
        self.conv2 = nn.Sequential(convqv2_3x3(planes, planes, nbits_w=nbits_w, q_mode=q_mode, l2=l2),
                                   nn.BatchNorm2d(planes),
                                   ActQv2(nbits=nbits_a, l2=l2))
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        self.out_actq = ActQv2(nbits=nbits_a, l2=l2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out


class BottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise,
                 l2=True):
        super(BottleneckQ, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2dQ(inplanes, planes, kernel_size=1, bias=False, nbits=nbits_w, mode=q_mode, l2=l2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2))
        self.conv2 = nn.Sequential(
            Conv2dQ(planes, planes, kernel_size=3, stride=stride,
                    padding=1, bias=False, nbits=nbits_w, mode=q_mode, l2=l2),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2))
        self.conv3 = nn.Sequential(
            Conv2dQ(planes, planes * 4, kernel_size=1, bias=False, nbits=nbits_w, mode=q_mode, l2=l2),
            nn.BatchNorm2d(planes * 4),
            ActQ(nbits=nbits_a, signed=True, l2=l2))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.out_actq = ActQ(nbits=nbits_a, l2=l2)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.out_actq(out)
        return out


class ResNetQ(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True,
                 **kwargs):
        self.inplanes = 64
        super(ResNetQ, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.l2 = l2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        nbits=self.nbits_w, mode=self.q_mode, l2=self.l2),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(nbits=self.nbits_a, signed=True, l2=self.l2),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQv2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True,
                 **kwargs):
        self.inplanes = 64
        super(ResNetQv2, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.l2 = l2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQv2(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          nbits=self.nbits_w, mode=self.q_mode, l2=self.l2),
                nn.BatchNorm2d(planes * block.expansion),
                ActQv2(nbits=self.nbits_a, signed=True, l2=self.l2),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQFNv2(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True,
                 **kwargs):
        self.inplanes = 64
        super(ResNetQFNv2, self).__init__()
        # We don't quantize first layer
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.l2 = l2
        self.conv1 = nn.Sequential(
            ActQv2(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQv2(3, 64, kernel_size=7, stride=2, padding=3, bias=False, nbits=nbits_w, mode=q_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1), )  # del ActQ as LQ-Net
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1),
                                     ActQv2(nbits=nbits_a))  # del ActQ as LQ-Net
        self.fc = LinearQv2(512 * block.expansion, num_classes, nbits=nbits_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQv2(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          nbits=self.nbits_w, mode=self.q_mode, l2=self.l2),
                nn.BatchNorm2d(planes * block.expansion),
                ActQv2(nbits=self.nbits_a, signed=True, l2=self.l2),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, q_mode=self.q_mode, l2=self.l2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetQFI(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise):
        self.inplanes = 64
        super(ResNetQFI, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.conv1 = nn.Sequential(
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQ(3, 64, kernel_size=7, stride=2, padding=3, bias=False, nbits=nbits_w, mode=q_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     ActQ(nbits=nbits_a))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.expand_actq = ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8,
                                expand=True)
        self.expand_fc = LinearQ(512 * block.expansion * 2, num_classes, nbits=nbits_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        nbits=self.nbits_w, mode=self.q_mode),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(nbits=self.nbits_a, signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, q_mode=self.q_mode))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, q_mode=self.q_mode))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.expand_actq(x)
        x = self.expand_fc(x)
        return x


class ResNetQFN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise):
        self.inplanes = 64
        super(ResNetQFN, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.conv1 = nn.Sequential(
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQ(3, 64, kernel_size=7, stride=2, padding=3, bias=False, nbits=nbits_w, mode=q_mode),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # del ActQ as LQ-Net
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     ActQ(nbits=nbits_a))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7, stride=1),
                                     ActQ(nbits=nbits_a))  # del ActQ as LQ-Net
        self.fc = LinearQ(512 * block.expansion, num_classes, nbits=nbits_w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dQ(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        nbits=self.nbits_w, mode=self.q_mode),
                nn.BatchNorm2d(planes * block.expansion),
                ActQ(nbits=self.nbits_a, signed=True),  # different with pre-trained model
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits_w=self.nbits_w, nbits_a=self.nbits_a, q_mode=self.q_mode))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits_w=self.nbits_w,
                                nbits_a=self.nbits_a, q_mode=self.q_mode))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_q(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_q_map.json')
    return model


def resnet18_qfn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFN(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qfn_map.json')
    return model


def resnet18_qfi(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFI(BasicBlockQ, [2, 2, 2, 2], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet18']), 'resnet18_qfi_map.json')
    return model


def resnet34_q(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BasicBlockQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_q_map.json')
    return model


def resnet34_qv2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQv2(BasicBlockQv2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qv2_map.json')
    return model


def resnet34_qfnv2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFNv2(BasicBlockQv2, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qfnv2_map.json')
    return model


def resnet34_qfi(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFI(BasicBlockQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet34']), 'resnet34_qfi_map.json')
    return model


def resnet50_q(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet50']), 'resnet50_q_map.json')
    return model


def resnet50_qfn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQFN(BottleneckQ, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['resnet50']), 'resnet50_qfn_map.json')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQ(BottleneckQ, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

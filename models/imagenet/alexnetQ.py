import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import sys

from models.modules import ActQ, Conv2dQ, LinearQ, PACT, ActQv2, Conv2dQv2, LinearQv2, DropoutScale
from models.imagenet import load_fake_quantized_state_dict
from models.modules import Qmodes
import models.modules as my_nn
from models import load_pre_state_dict
import ipdb

# TODO: Re-structure the code.
__all__ = ['_AlexNetQ', 'AlexNetQ', 'alexnet_q', 'alexnet_qfn', 'alexnet_qfi',
           'alexnet_q_pact', 'alexnet_qv2', 'alexnet_qfnv2',
           'alexnet_lsq', 'alexnet_llsq']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def alexnet_llsq(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Please use [dataset]_[architecture]_[quan_type] as the function name

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _AlexNetQ(qf=quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['alexnet']),
                            '{}_map.json'.format(model_name))
    return model


def alexnet_lsq(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Please use [dataset]_[architecture]_[quan_type] as the function name

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _AlexNetQ(qf=quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['alexnet']),
                            '{}_map.json'.format(model_name))
    return model


class _AlexNetQ(nn.Module):

    def __init__(self, num_classes=1000, qf=None):
        """
        :param num_classes:
        :param qf: quantization factory
        """
        super(_AlexNetQ, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qf.product_ActQ(),
            qf.product_Conv2dQ(64, 192, kernel_size=5, padding=2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qf.product_ActQ(),
            qf.product_Conv2dQ(192, 384, kernel_size=3, padding=1),
            # conv3
            nn.ReLU(inplace=True),
            qf.product_ActQ(),
            qf.product_Conv2dQ(384, 256, kernel_size=3, padding=1),
            # conv4
            nn.ReLU(inplace=True),
            qf.product_ActQ(),
            qf.product_Conv2dQ(256, 256, kernel_size=3, padding=1),
            # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            qf.product_ActQ(),
        )
        self.classifier = nn.Sequential(
            DropoutScale(),
            qf.product_LinearQ(256 * 6 * 6, 4096),  # fc6
            nn.ReLU(inplace=True),
            qf.product_ActQ(),
            DropoutScale(),
            qf.product_LinearQ(4096, 4096),  # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetQ(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQ, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv3
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv4
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # As the experiment result shows, there is no difference between layer wise with kernel wise.
            LinearQ(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            nn.Dropout(),
            LinearQ(4096, 4096, nbits=nbits_w, mode=q_mode.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetQv2(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQv2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),
            # conv3
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),
            # conv4
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),
            # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQv2(nbits=nbits_a, l2=l2),
        )
        self.classifier = nn.Sequential(
            DropoutScale(),
            # As the experiment result shows, there is no difference between layer wise with kernel wise.
            LinearQv2(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            DropoutScale(),
            LinearQv2(4096, 4096, nbits=nbits_w, mode=q_mode.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetQPACT(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQPACT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            PACT(nbits=nbits_a),
            Conv2dQ(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            PACT(nbits=nbits_a),
            Conv2dQ(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv3
            nn.ReLU(inplace=True),
            PACT(nbits=nbits_a),
            Conv2dQ(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv4
            nn.ReLU(inplace=True),
            PACT(nbits=nbits_a),
            Conv2dQ(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            PACT(nbits=nbits_a),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # As the experiment result shows, there is no difference between layer wise with kernel wise.
            LinearQ(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            PACT(nbits=nbits_a),
            nn.Dropout(),
            LinearQ(4096, 4096, nbits=nbits_w, mode=q_mode.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetQFN(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQFN, self).__init__()
        self.features = nn.Sequential(
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True, l2=l2),
            Conv2dQ(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv3
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv4
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            Conv2dQ(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a, l2=l2),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            LinearQ(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),
            # nn.Dropout(),
            LinearQ(4096, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a, l2=l2),  # key layer
            LinearQ(4096, num_classes, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetQFNv2(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise, l2=True):
        super(AlexNetQFNv2, self).__init__()
        self.features = nn.Sequential(
            ActQv2(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True, l2=l2),
            Conv2dQv2(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode, l2=l2),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv3
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv4
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            Conv2dQv2(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode, l2=l2),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            ActQv2(nbits=nbits_a, l2=l2),
            DropoutScale(),
            LinearQv2(256 * 6 * 6, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc6
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),
            DropoutScale(),
            LinearQv2(4096, 4096, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc7
            nn.ReLU(inplace=True),
            ActQv2(nbits=nbits_a, l2=l2),  # key layer
            LinearQv2(4096, num_classes, nbits=nbits_w, mode=Qmodes.layer_wise, l2=l2),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# # TODO: find best {nbits => Expansion coefficient}
# expand_weight = {
#     4: [0.8, 0.2]  # [16/17, 1/17]
# }


class AlexNetQFI(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=4, nbits_a=4, q_mode=Qmodes.kernel_wise):
        super(AlexNetQFI, self).__init__()
        self.nbits_w = nbits_w
        self.nbits_a = nbits_a
        self.q_mode = q_mode
        self.features = nn.Sequential(
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
            Conv2dQ(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, mode=q_mode),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a),
            Conv2dQ(64, 192, kernel_size=5, padding=2, nbits=nbits_w, mode=q_mode),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a),
            Conv2dQ(192, 384, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode),  # conv3
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a),
            Conv2dQ(384, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode),  # conv4
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a),
            Conv2dQ(256, 256, kernel_size=3, padding=1, nbits=nbits_w, mode=q_mode),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ActQ(nbits=nbits_a),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            LinearQ(256 * 6 * 6, 4096, nbits=nbits_w),  # fc6
            nn.ReLU(inplace=True),
            ActQ(nbits=nbits_a),
            # nn.Dropout(),
            LinearQ(4096, 4096, nbits=nbits_w),  # fc7
            nn.ReLU(inplace=True),
            ActQ(nbits=-1 if max(nbits_a, nbits_w) <= 0 else 8, expand=True),
        )
        # self.shared_fc = LinearQ(4096, num_classes, nbits=nbits_w)
        # self.last_add = EltwiseAdd(inplace=True)
        self.expand_fc = LinearQ(4096 * 2, num_classes, nbits=nbits_w)  # fc8

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        out = self.expand_fc(x)
        # x_high, x_low = self.classifier(x)
        # out_high = self.shared_fc(x_high)
        # out_low = self.shared_fc(x_low)
        # out = self.last_add(out_high, out_low)
        return out


def alexnet_q(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQ(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_q_map.json')
    return model


def alexnet_qv2(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQv2(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_qv2_map.json')
    return model


def alexnet_qfnv2(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQFNv2(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_qfnv2_map.json')
    return model


def alexnet_q_pact(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQPACT(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_q_pact_map.json')
    return model


def alexnet_qfn(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQFN(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_qfn_map.json')
    return model


def alexnet_qfi(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetQFI(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_qfi_map.json')
    return model

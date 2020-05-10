import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.imagenet import load_fake_quantized_state_dict
from models.modules import Conv2dBP, LinearBP, Conv2dBPv2

__all__ = ['AlexNetBP', 'alexnet_bp', 'alexnet_bp_no_fc', 'alexnet_bp_no_fc_v2']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-distiller-sparse.pth',
    # 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    # 'alexnet': 'https://download.pytorch.org/models/alexnet_bp.pth',
}


class AlexNetBP(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=8, increase_factor=1 / 3):
        super(AlexNetBP, self).__init__()
        self.features = nn.Sequential(
            Conv2dBP(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, increase_factor=increase_factor / 2),
            # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBP(64, 192, kernel_size=5, padding=2, nbits=nbits_w, increase_factor=increase_factor),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBP(192, 384, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv3
            nn.ReLU(inplace=True),
            Conv2dBP(384, 256, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv4
            nn.ReLU(inplace=True),
            Conv2dBP(256, 256, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LinearBP(256 * 6 * 6, 4096, nbits=nbits_w, increase_factor=increase_factor),  # fc6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearBP(4096, 4096, nbits=nbits_w, increase_factor=increase_factor),  # fc7
            nn.ReLU(inplace=True),
            LinearBP(4096, num_classes, nbits=nbits_w, increase_factor=increase_factor / 2),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetBPNOFC(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=8, increase_factor=1 / 3, **kwargs):
        super(AlexNetBPNOFC, self).__init__()
        self.features = nn.Sequential(
            Conv2dBP(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, increase_factor=increase_factor / 2),
            # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBP(64, 192, kernel_size=5, padding=2, nbits=nbits_w, increase_factor=increase_factor),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBP(192, 384, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv3
            nn.ReLU(inplace=True),
            Conv2dBP(384, 256, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv4
            nn.ReLU(inplace=True),
            Conv2dBP(256, 256, kernel_size=3, padding=1, nbits=nbits_w, increase_factor=increase_factor),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LinearBP(256 * 6 * 6, 4096, nbits=nbits_w, increase_factor=increase_factor, no_bp=True),  # fc6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearBP(4096, 4096, nbits=nbits_w, increase_factor=increase_factor, no_bp=True),  # fc7
            nn.ReLU(inplace=True),
            LinearBP(4096, num_classes, nbits=nbits_w, increase_factor=increase_factor / 2, no_bp=True),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetBPNOFCv2(nn.Module):

    def __init__(self, num_classes=1000, nbits_w=8, increase_factor=1 / 3, log=False):
        super(AlexNetBPNOFCv2, self).__init__()
        self.features = nn.Sequential(
            Conv2dBPv2(3, 64, kernel_size=11, stride=4, padding=2, nbits=nbits_w, log=log),
            # conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBPv2(64, 192, kernel_size=5, padding=2, nbits=nbits_w, log=log),  # conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBPv2(192, 384, kernel_size=3, padding=1, nbits=nbits_w, log=log),  # conv3
            nn.ReLU(inplace=True),
            Conv2dBPv2(384, 256, kernel_size=3, padding=1, nbits=nbits_w, log=log),  # conv4
            nn.ReLU(inplace=True),
            Conv2dBPv2(256, 256, kernel_size=3, padding=1, nbits=nbits_w, log=log),  # conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            LinearBP(256 * 6 * 6, 4096, nbits=nbits_w, increase_factor=increase_factor, no_bp=True),  # fc6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearBP(4096, 4096, nbits=nbits_w, increase_factor=increase_factor, no_bp=True),  # fc7
            nn.ReLU(inplace=True),
            LinearBP(4096, num_classes, nbits=nbits_w, increase_factor=increase_factor / 2, no_bp=True),  # fc8
        )

    def forward(self, x):
        x = self.features(x)
        if len(x) == 2:
            x[0] = x[0].view(x[0].size(0), 256 * 6 * 6)
        else:
            x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_bp(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetBP(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_bp_map.json')
    return model


def alexnet_bp_no_fc(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetBPNOFC(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_bp_no_fc_map.json')
    return model


def alexnet_bp_no_fc_v2(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetBPNOFCv2(**kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, model_zoo.load_url(model_urls['alexnet'], map_location='cpu'),
                                       'alexnet_bp_no_fc_v2_map.json')
    return model

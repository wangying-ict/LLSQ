'''VGG11/13/16/19 in Pytorch.'''
import sys

import torch.utils.model_zoo as model_zoo

import models.modules as my_nn
from models import load_pre_state_dict
from models.cifar10 import _VGGQ

__all__ = [
    'cifar10_vggsmall_llsq'
]

# model name: [dataset]-[architecture]-[acc]-zxd.pth
model_urls = {
    'vgg7': 'https://fake/models/cifar10-vgg7-zxd-8943fa3.pth',
}


def cifar10_vggsmall_llsq(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Please use [dataset]_[architecture]_[quan_type] as the function name

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    model_name = sys._getframe().f_code.co_name
    quan_type = model_name.split('_')[-1]
    quan_factory = my_nn.QuantizationFactory(quan_type, **kwargs)
    model = _VGGQ('VGG7', quan_factory)
    if pretrained:
        load_pre_state_dict(model, model_zoo.load_url(model_urls['vgg7']),
                            '{}_map.json'.format(model_name))
    return model

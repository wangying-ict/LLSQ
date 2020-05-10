import torch
from torchvision.models import resnet
import models.imagenet as imagenet_extra_models
import wrapper
from utils.PytorchToCaffe import pytorch_to_caffe
import ipdb

if __name__ == '__main__':
    name = 'resnet18'
    model = imagenet_extra_models.seq_resnet18(pretrained=True)
    # model = resnet.resnet18(pretrained=True)
    wrapper.fuse_bn_recursively(model)
    # resnet18.load_state_dict()
    model.eval()
    input = torch.ones([1, 3, 224, 224])
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(model, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    # pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))

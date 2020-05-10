# https://github.com/sovrasov/flops-counter.pytorch
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from models.imagenet import mobilenet_han
import ipdb

with torch.cuda.device(0):
    net = mobilenet_han()
    with open('mobilenet_han_flops.txt', 'w') as f:
        flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                                  ost=f)
    print('Flops:  ' + flops)
    print('Params: ' + params)

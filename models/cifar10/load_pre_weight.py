import json
import torch
from collections import OrderedDict


def load_fake_quantized_state_dict(model, original_state_dict, key_map=None):
    if not isinstance(key_map, OrderedDict):
        with open('models/cifar10/{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        if 'expand_' in k and model.state_dict()[k].shape != original_state_dict[v].shape:
            ori_weight = original_state_dict[v]
            new_weight = torch.cat((ori_weight, ori_weight * 2 ** 4), dim=1)
            model.state_dict()[k].copy_(new_weight)
        else:
            model.state_dict()[k].copy_(original_state_dict[v])

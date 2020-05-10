import os
import json
import argparse
import ipdb


def rename():
    with open(args.layer_rename_map, 'r') as rf:
        name_map = json.load(rf)
    for filename in os.listdir(args.dir):
        for old, new in name_map.items():
            if old in filename:
                os.rename(os.path.join(args.dir, filename), os.path.join(args.dir, filename.replace(old, new)))
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Fast Pose: Prepare prototxt for Simulator')
    parser.add_argument('dir', help='txt dir')
    parser.add_argument('--layer-rename-map', default='alexnet_map.json')
    args = parser.parse_args()
    rename()

import random
import os
import argparse
import ipdb
import shutil

parser = argparse.ArgumentParser(description='Sparsity Analysis')
parser.add_argument('src', metavar='DIR',
                    help='the source directory')
parser.add_argument('dst', metavar='DIR', help='destination directory')
parser.add_argument('-s', '--split', type=int, default=5,
                    help='the split ratio, default: (1/5)')

"""
    root/dog/xxxx.jpgs
    root/dog/xxxx.jpgs
"""


def main():
    args = parser.parse_args()
    assert os.path.exists(args.src), '{} does not exists'.format(args.src)
    if not os.path.exists(args.dst):
        print('mkdir {}'.format(args.dst))
        os.makedirs(args.dst)
    assert os.path.exists(args.dst), '{} does not exists'.format(args.dst)
    dirs = os.listdir(args.src)
    for cls_dir in dirs:
        imgs = os.listdir(os.path.join(args.src, cls_dir))
        for i, img in enumerate(imgs):
            if not os.path.exists(os.path.join(args.dst, cls_dir)):
                print('mkdir {}'.format(os.path.join(args.dst, cls_dir)))
                os.makedirs(os.path.join(args.dst, cls_dir))
            if i % args.split == 0:
                shutil.move(os.path.join(args.src, cls_dir, img), os.path.join(args.dst, cls_dir, img))


if __name__ == '__main__':
    main()

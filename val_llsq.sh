#!/usr/bin/env bash
python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_llsq -j 10 --pretrained -b 128 --log-name $2 \
    --lr 2e-4 --wd 1e-4 --warmup-multiplier 2.5 --warmup-epoch 2 \
    --gpu $1 --epochs 60 --lr-scheduler  CosineAnnealingLR  --qw 3 --qa 3 --debug


python examples/classifier_imagenet/main.py ~/datasets/data.imagenet \
    -a alexnet_llsq -j 10 --pretrained -b 128 --log-name $2 \
    --lr 2e-4 --wd 1e-4 --warmup-multiplier 2.5 --warmup-epoch 2 \
    --gpu $1 --epochs 60 --lr-scheduler  CosineAnnealingLR  --qw 3 --qa 3 --debug
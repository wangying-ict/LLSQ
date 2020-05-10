import argparse
import json
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

import models.imagenet as imagenet_extra_models
import wrapper
from examples import add_weight_decay, gen_key_map, accuracy, set_bn_eval
from models.modules import q_modes, FunctionBitPruningSTE
from models.modules.bit_pruning import count_bit, bit_sparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.extend(sorted(name for name in imagenet_extra_models.__dict__
                          if name.islower() and not name.startswith("__")
                          and callable(imagenet_extra_models.__dict__[name])))
q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
str_q_mode_map = {'layer_wise': q_modes.layer_wise,
                  'kernel_wise': q_modes.kernel_wise}
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--log-name', default='log', type=str)
parser.add_argument('--gen-map', action='store_true', default=False,
                    help='generate key map for quantized model')
parser.add_argument('--original-model', default='', type=str,
                    help='original model')
parser.add_argument('--qw', default=8, type=int,
                    help='quantized weight bit')
parser.add_argument('--q-mode', choices=q_modes_choice, default='kernel_wise',
                    help='Quantization modes: ' +
                         ' | '.join(q_modes_choice) +
                         ' (default: kernel_wise)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='save running scale in tensorboard')
parser.add_argument('--freeze-bn', action='store_true', default=False, help='Freeze BN')
parser.add_argument('--extract-inner-data', action='store_true', default=False,
                    help='Extract inner feature map and weights')
parser.add_argument('--increase-factor', default=0.33, type=float, help='increase factor for bit pruning')
parser.add_argument('-c', '--complement', default=False, action='store_true',
                    help='use twos complement representation')
parser.add_argument('--quan-log', default=False, action='store_true',
                    help='use log-like quantization for bit pruning')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.gen_map:
        args.qw = -1
        args.qa = -1
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    try:
        model = models.__dict__[args.arch](pretrained=args.pretrained)
        args.qw = -1
        args.qa = -1
    except KeyError:
        if 'bp' in args.arch:
            model = imagenet_extra_models.__dict__[args.arch](pretrained=args.pretrained,
                                                              nbits_w=args.qw,
                                                              log=args.quan_log,
                                                              increase_factor=args.increase_factor
                                                              )
        else:
            model = imagenet_extra_models.__dict__[args.arch](pretrained=args.pretrained)
    print('model:\n=========\n{}\n=========='.format(model))
    if args.gen_map:
        try:
            ori_model = models.__dict__[args.original_model]()
        except KeyError:
            ori_model = imagenet_extra_models.__dict__[args.original_model]()
        print('original model:\n=========\n{}\n=========='.format(ori_model))

        key_map = gen_key_map(model.state_dict(), ori_model.state_dict())

        with open('models/imagenet/{}_map.json'.format(args.arch), 'w') as wf:
            json.dump(key_map, wf)
        print('Generate key map done')
        return

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    params = add_weight_decay(model, weight_decay=args.weight_decay, skip_keys=['expand_'])
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            # if not args.quant_bias_scale:
            #     args.start_epoch = checkpoint['epoch']
            #     best_acc1 = checkpoint['best_acc1']
            #     if args.gpu is not None:
            #         # best_acc1 may be from a checkpoint from a different GPU
            #         best_acc1 = best_acc1.to(args.gpu)
            # try:
            #     model.load_state_dict(checkpoint['state_dict'])
            #     # ValueError: loaded state dict has a different number of parameter groups
            #     # different version
            #     # optimizer.load_state_dict(checkpoint['optimizer'])
            # except RuntimeError:
            #     print('Fine-tune qfi_wide model using qfn_relaxed weights.')
            #     key_map = gen_key_map(model.state_dict(), checkpoint['state_dict'])
            #     load_fake_quantized_state_dict(model, checkpoint['state_dict'], key_map)
            #     args.start_epoch = 0
            #     best_acc1 = 0
            #     optimizer = torch.optim.SGD(params, args.lr,
            #                                 momentum=args.momentum)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    args.log_name = 'logger/{}_{}'.format(args.arch, args.log_name)
    writer = SummaryWriter(args.log_name)
    with open('{}/{}.txt'.format(args.log_name, args.arch), 'w') as wf:
        wf.write(str(model))
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=3,
                                              after_scheduler=scheduler_cosine)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        scheduler_warmup.step()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        writer.add_scalar('val/acc1', acc1, epoch)
        writer.add_scalar('val/acc5', acc5, epoch)
        # writer.add_scalar('val/bs', bs, epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, prefix='{}/{}'.format(args.log_name, args.arch))
        if epoch % 10 == 0:
            save_checkpoint_backup(model.state_dict(), prefix='{}/{}_{}'.format(args.log_name, args.arch, epoch))
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    # TODO: one epoch lr config
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if args.freeze_bn:
        model.apply(set_bn_eval)
    end = time.time()
    base_step = epoch * len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], base_step + i)
        writer.add_scalar('train/acc1', top1.avg, base_step + i)
        writer.add_scalar('train/acc5', top5.avg, base_step + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
            if args.debug:
                for k, v in model.state_dict().items():
                    if 'weight_int' in k:
                        # writer.add_histogram('train/{}'.format(k), v.float(), base_step + i)
                        bit_cnt = count_bit(v)
                        bs = bit_sparse(bit_cnt)
                        writer.add_scalar('train/bit_sparse/{}'.format(k), bs, base_step + i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    # =============bit sparsity =================
    # total_cnt = 0
    # total_conv_cnt = 0
    # total_weight_cnt = 0
    # total_weight_conv_cnt = 0
    # total_bit_cnt = 0
    # total_bit_conv_cnt = 0
    # for k, v in model.state_dict().items():
    #     if 'weight_int' in k:
    #         cnt_sum = v.view(-1).shape[0]
    #         total_cnt += cnt_sum
    #         total_weight_cnt += (v.float().abs() > 0).sum().float()
    #         bit_cnt = count_bit(v, complement=args.complement)
    #         total_bit_cnt += bit_cnt.sum().float()
    #         # weight_sparsity = 1 - (v.float().abs() > 0).sum().float() / cnt_sum
    #         # bit_sparsity = bit_sparse(bit_cnt, args.complement)
    #         if len(v.shape) == 4:
    #             total_conv_cnt += cnt_sum
    #             total_weight_conv_cnt += (v.float().abs() > 0).sum().float()
    #             total_bit_conv_cnt += bit_cnt.sum().float()
    # if args.complement:
    #     bit_width = 8
    # else:
    #     bit_width = 7
    # total_conv_bs = 1 - total_bit_conv_cnt / total_conv_cnt / bit_width
    # total_bs = 1 - total_bit_cnt / total_cnt / bit_width
    # if 'alexnet_bp_no_fc' in args.arch:
    #     return_bs = total_conv_bs
    # else:
    #     return_bs = total_bs
    # =============bit sparsity end =================
    if args.extract_inner_data:
        # wrapper.save_inner_hooks(model)
        wrapper.debug_graph_hooks(model)
        print('extract inner feature map and weight')
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
            if args.extract_inner_data:
                for k, v in model.state_dict().items():
                    if 'num_batches_tracked' in k:
                        continue
                    nparray = v.detach().cpu().float().numpy()
                    if 'weight' in k:
                        radix_key = k.replace('weight', 'radix_position')
                        try:
                            radix_position = model.state_dict()[k.replace('weight', 'radix_position')]
                            v_bp = FunctionBitPruningSTE.apply(v, radix_position)
                            nparray = v_bp.detach().cpu().float().numpy()
                        except KeyError:
                            print('warning: {} does not exist.'.format(radix_key))
                            pass
                        np_save = nparray.reshape(nparray.shape[0], -1)
                    elif 'bias' in k:
                        np_save = nparray.reshape(1, -1)
                    else:
                        print(k)
                        np_save = nparray.reshape(-1, nparray.shape[-1])
                    np.savetxt('{}.txt'.format(k), np_save, delimiter=' ', fmt='%.8f')
                    # np.save('{}'.format(k), v.cpu().float().numpy())
                print('extract inner data done, return')
                break
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint_backup(state, prefix, filename='_checkpoint.pth'):
    torch.save(state, prefix + filename)


def save_checkpoint(state, is_best, prefix, filename='_checkpoint.pth.tar'):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + '_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

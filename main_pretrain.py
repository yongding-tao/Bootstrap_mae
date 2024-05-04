# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10

import timm

from util.ema import EMA
from util.pos_embed import interpolate_pos_embed

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    # EMA
    parser.add_argument('--use_ema', action='store_true',
                        help='use EMA algorithm to train model weight')
    parser.add_argument('--half_life', default=0.99, type=float,
                        help='the half-life of EMA for model weight')
    
    # bootstrap
    parser.add_argument('--bootstrap_k', default=0, type=int,
                        help='iteration k of bootstrap, k = 0 means the original mae not save the feature. k = 1 means the original mae but save features.')
    parser.add_argument('--feature_depth', default=8, type=int,
                        help='the depth of the feature to save')
    parser.add_argument('--last_model_checkpoint', default="./MAE-1/pretrain/output_dir/checkpoint-199.pth", type=str,
                        help='the last model to load')
    parser.add_argument('--only_load_encoder', action='store_true',
                        help='only the encoder weight of last model')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_freq', default=20, type=int,
                        help='The frequency of saving checkpoints')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # for CIFAR10
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for ImageNet
            ])
    
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train = CIFAR10(root=args.data_path,
                      train=True,
                      download=True,
                      transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, bootstrap_k=args.bootstrap_k) # debug not write bootstrap_k=arg.bootstrap_k

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # bootstrap : load last model
    if args.bootstrap_k > 1:
        last_model = models_mae.__dict__["B"+args.model]()
        
        checkpoint = torch.load(args.last_model_checkpoint, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.last_model_checkpoint)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        if not args.only_load_encoder or args.bootstrap_k > 2:
            msg1 = model.load_state_dict(checkpoint_model, strict=False) # initialize with last model weight
            print(msg1)
        else: # only load encoder of MAE-1 for the MAE-2
            # 初始化一个字典来存储encoder部分的权重
            encoder_weights = {}

            # 根据模型中的模块和预训练权重中的键名筛选出encoder部分的权重
            # 这里我们根据 'patch_embed.', 'pos_embed', 'blocks.', 'norm.' 等模块前缀来筛选
            for key, value in checkpoint_model.items():
                if key.startswith('patch_embed.'):
                    # 去除 'patch_embed.' 前缀
                    new_key = key[len('patch_embed.'):]
                    encoder_weights[new_key] = value
                elif key == 'pos_embed':
                    encoder_weights['pos_embed'] = value
                elif key.startswith('blocks.'):
                    # 去除 'blocks.' 前缀
                    new_key = key[len('blocks.'):]
                    # 修改键名以适应 Block 模块的键名
                    layer_index, sub_key = new_key.split('.', 1)
                    layer_index = int(layer_index)
                    if layer_index not in encoder_weights:
                        encoder_weights[layer_index] = {}
                    encoder_weights[layer_index][sub_key] = value
                elif key.startswith('norm.'):
                    # 去除 'norm.' 前缀
                    new_key = key[len('norm.'):]
                    encoder_weights[f'norm.{new_key}'] = value

            # 加载encoder部分的权重到模型中
            model.patch_embed.load_state_dict({k: encoder_weights[k] for k in encoder_weights if k in model.patch_embed.state_dict()})
            model.pos_embed.data.copy_(encoder_weights['pos_embed'])
            model.norm.load_state_dict({k: encoder_weights[f'norm.{k}'] for k in model.norm.state_dict()})

            # 遍历模型的blocks部分，逐个加载块（Block）的权重
            for i, block in enumerate(model.blocks):
                # 提取每个Block的权重
                block_state_dict = encoder_weights.get(i, {})
                # 确保字典不为空
                if block_state_dict:
                    # 加载权重
                    block.load_state_dict(block_state_dict)
        msg2 = last_model.load_state_dict(checkpoint_model, strict=False)
        print(msg2)
        
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        interpolate_pos_embed(last_model, checkpoint_model)
        
        # # 在加载模型到设备之前记录显存使用情况
        # initial_memory_allocated = torch.cuda.memory_allocated(device)
        # initial_memory_reserved = torch.cuda.memory_reserved(device)

        # 将模型加载到 GPU 设备
        last_model.to(device)

        # # 在加载模型到设备之后记录显存使用情况
        # final_memory_allocated = torch.cuda.memory_allocated(device)
        # final_memory_reserved = torch.cuda.memory_reserved(device)

        # # 计算加载模型过程中显存变化
        # allocated_difference = final_memory_allocated - initial_memory_allocated
        # reserved_difference = final_memory_reserved - initial_memory_reserved

        # print(f"显存分配变化: {allocated_difference / 1024**2:.2f} MB")
        # print(f"显存保留变化: {reserved_difference / 1024**2:.2f} MB")
        
        # last_model.to(device)
        last_model.eval()
        
    ema = None
    if args.use_ema:
        # EMA init
        ema = EMA(model, args.half_life)
        ema.register()


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if args.bootstrap_k > 1:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                last_model=last_model,
                ema=ema
            )
        else:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                ema=ema
            )
        
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            if args.use_ema:
                ema.apply_shadow()
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            if args.use_ema:
                ema.restore()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# -*- coding: utf-8 -*-
'''
This code is for C2C pretraining and CLS finetuning of ViT.
It is based on the original work by @kentaroy47, @arutema47
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from models.vit_designing_clear import ViT_designing_clear

from utils import progress_bar
from models.convmixer import ConvMixer
from randomaug import RandAugment

from einops.layers.torch import Rearrange
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from random import choices
import random

import piqa
import math

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='256')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
parser.add_argument('--wdecay', type=float, default=0, help='weight decay parameter of Adam')
parser.add_argument('--task', default='cifar10', help='datasets and transformations')
parser.add_argument('--watermark', type=str, default=None, help='control head number')

args = parser.parse_args()

# take in args
import wandb

if args.watermark != None:
    watermark = f"{args.watermark}"
else:
    watermark = f"{args.net}_lr{args.lr}"
if args.amp:
    watermark += "_useamp"

wandb.init(project="c2c_code",
           name=watermark)
wandb.config.update(args)

if args.aug:
    import albumentations
bs = int(args.bs)
imsize = int(args.size)

use_amp = args.amp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
size = imsize

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
])

transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
])

# Add RandAugment with N, M(hyperparameter)
if args.aug:
    N = 2;
    M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

if args.task in ['cls', 'clear']:
    print("dataset: cifar100")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=0) #batch_size 100->10

    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
               'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
               'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
               'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
               'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
               'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
               'squirrel', 'streetcar', 'sunflower','sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
               'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')

# Model
print('==> Building model..')
if args.net == 'vit_designing_clear': 
    net = ViT_designing_clear( # base
        image_size=size,
        patch_size=args.patch,
        num_classes=100,  
        dim=int(args.dimhead),
        depth=12,  
        heads=12, 
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1,
        task=args.task,
        args=args,
    )

if device == 'cuda':
    print("make parallel...")
    net = torch.nn.DataParallel(net)  # make parallel
    net = net.to(device)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion_class = nn.CrossEntropyLoss()
criterion_denoise = nn.MSELoss()

def psnr(mse):
    PIXEL_MAX = 1.0 #255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) #PSNR구하는 코드

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wdecay) # for weight_decay experiment
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3 * 1e-5,
                                               factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau"

##### Training
scaler = torch.amp.GradScaler(enabled=use_amp)

cifar_img_size = transforms.Resize(32)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    psnr_score = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.task in ['cls']:
                outputs = net(inputs)
                loss = criterion_class(outputs, targets)        
            elif args.task in ['clear']:
                clear_im = inputs
                outputs = net(clear_im)
                loss = criterion_denoise(outputs, clear_im)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if args.task in ['cls']:
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        elif args.task in ['clear']:
            psnr_score = psnr(loss)
            print("training... psnr: {}".format(psnr_score))
        
        train_acc = 100. * correct / total

        if args.task in ['clear']:
            train_acc = psnr_score
    return train_loss / (batch_idx + 1), train_acc


##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_loss_cls = 0
    test_loss_deno = 0
    correct = 0
    psnr_score = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.task in ['cls']:
                outputs = net(inputs)
                loss = criterion_class(outputs, targets)
            elif args.task in ['clear']:
                clear_im = inputs
                outputs = net(clear_im)
                loss = criterion_denoise(outputs, clear_im)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            if args.task in ['cls']:
                correct += predicted.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            elif args.task in ['clear']:
                psnr_score = psnr(loss)
                print("testing... psnr: {}".format(psnr_score))

    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    cls_acc = acc
    if args.task in ['clear']:
        cls_acc = acc
        acc = psnr_score
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-ckpt.t7'.format(args.patch))
        best_acc = acc


    if (epoch + 1) % 50 == 0:
        state = {"model": net.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict()}
        torch.save(state, './checkpoint/' + args.net + '_' + args.task + '_epoch{}_'.format(epoch) + args.watermark + '.t7')


    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")

    return test_loss, acc


list_loss = []
list_acc = []


state_dict_temp = net.state_dict()

if args.watermark == 'lr_1e-4_cls_finetuned':
    state_dict = torch.load('./checkpoint/vit_designing_clear_cls_epoch99_lr_1e-4_cls.t7')['model'] # from pretrained lr_1e-4_cls
    state_dict_temp.update(state_dict)
    net.load_state_dict(state_dict_temp, strict=False)
elif args.watermark == 'lr_1e-4_clear_finetuned':
    state_dict = torch.load('./checkpoint/vit_designing_clear_clear_epoch99_lr_1e-4_clear.t7')['model'] # from pretrained 1clear_lr1e-4
    state_dict_temp.update(state_dict)
    net.load_state_dict(state_dict_temp, strict=False)

wandb.watch(net)

for epoch in range(start_epoch, args.n_epochs):
    if epoch == 0:
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        torch.save(state, './checkpoint/' + args.net + '_' + args.task + '_initial.t7')
    start = time.time()
    trainloss, train_acc = train(epoch) # training
    
    val_loss, acc = test(epoch)

    if args.cos:
        scheduler.step(epoch - 1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    wandb.log({'epoch': epoch, 'val_loss': val_loss, "val_acc(or psnr)": acc,  'train_loss': trainloss, 'train_acc': train_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)

# writeout wandb
wandb.save("wandb_{}.h5".format(args.net))
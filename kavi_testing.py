import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from utils import evaluate_standard, evaluate_standard_rp, evaluate_pgd, evaluate_pgd_rp
from utils import clamp, get_loaders, get_limit
from train import get_args
from model.resnet import ResNet18, ResNet50

def hi():

    args = get_args()
    print("*****")
    path = os.path.join('ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    if args.network == 'ResNet18':
        net = ResNet18
    elif args.network == 'ResNet50':
        net = ResNet50
    else:
        print('Wrong network:', args.network)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                    worker=args.worker)

    model = net(num_classes=10, mnist=True, rp=args.rp, rp_block=args.rp_block, rp_out_channel=args.rp_out_channel,
                    normalize=dataset_normalization).cuda()

    for name, param in model.named_parameters():
        print(name)
       

if __name__ == '__main__':
      hi()

    




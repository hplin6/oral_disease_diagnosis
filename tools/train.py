# ------------------------------------------------------------------------------
# Copy from https://github.com/HRNet/HRNet-Image-Classification
# Modified by us
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger


def parse_args(subdir):
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default="experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
                        #default = "experiments/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')

    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    parser.add_argument('--subDir',
                        help='sub directory',
                        type=str,
                        default=subdir)

    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def visualize(model, input_size=(3, 512, 512)):
    '''Visualize the input size though the layers of the model'''
    x = torch.zeros(input_size).unsqueeze_(dim=0)
    print(x.size())
    for layer in list(model.features) + list(model.classifier):
        x = layer(x)
        print(x.size())

def main_train(subdir):
    args = parse_args(subdir)

    logger, console, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    print(gpus)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #print(model)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
            best_model = True
            
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
    traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    valdir = os.path.join(config.DATASET.ROOT, config.DATASET.VALID_SET)
    print(traindir,valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0],scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((int(config.MODEL.IMAGE_SIZE[0]),int(config.MODEL.IMAGE_SIZE[1]))),
            #transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    import time
    localtime = time.localtime(time.time())
    runtime = "{}-{}-{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour)
    print("runtime=",runtime)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch, writer_dict)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
            if epoch >= 0:
                modelname= config.MODEL.NAME + '-{}_{}_{}'.format(subdir,runtime,epoch+1)
                final_model_state_file = os.path.join(final_output_dir,modelname+".pth")
                logger.info('saving final model state to {}'.format(
                    final_model_state_file))
                torch.save(model.module.state_dict(), final_model_state_file)
                writer_dict['writer'].close()
        else:
            best_model = False

    logger.removeHandler(console)#not print again

if __name__ == '__main__':
    train_method="cross_valid"
    if train_method=="cross_valid":
        subdirs=["part1","part2","part3","part4","part5"]
        for subdir in subdirs:
            main_train(subdir)
    elif train_method=="traind_valid":
        main_train("v4")

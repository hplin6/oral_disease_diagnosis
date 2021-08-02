# ------------------------------------------------------------------------------
# Copy from https://github.com/HRNet/HRNet-Image-Classification
# Modified by us
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate,validate_speed
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args(subdir):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        #required=True,
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
                        default='save_weights/cls_hrnet-v4_4-18-19.h5.pth')  # my1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````et_4-18-19_16.h5.pth')  # #original
                        #default = 'save_weights/cls_hrnet-v4_7-29-6.h5.pth')  # my2
    args = parser.parse_args()
    update_config(config, args)

    return args

def test_main(subdir):
    args = parse_args(subdir)

    logger, console, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))    

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Data loading code
    valdir = os.path.join(config.DATASET.ROOT,
                          config.DATASET.TEST_SET)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0])),
            # transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1,#config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    # evaluate on validation set
    validate(config, valid_loader, model, criterion, None, b_test=True)

if __name__ == '__main__':
    subdir="v4-random-oversampling"
    subdir="v4-center-oversampling"
    test_main(subdir)

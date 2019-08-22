import argparse
import yaml
import logging
import random

import numpy as np
import torch

from easydict import EasyDict

from lib.dataset import VOCDataset
from lib.dataset import MakeDataLoader
from lib.execute import Execute
from lib.transform import Resize, Compose
from lib.transform import ToTensor, ToGridCellOffset
from lib.transform import RandomHorizontalFlip
from lib.transform import RandomCrop, ColorJitter
from lib.transform import RandomBlur, RandomShift
from lib._parser_config import parser_config

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='train object-detection of single stage.')
parser.add_argument('--config', type=str, default='cfgs/yolo.yaml',
                    help='configuration file')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.config) as rptr:
        config = EasyDict(yaml.load(rptr))
    config = parser_config(config)

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    rhf = RandomHorizontalFlip(p=0.5)
    rc_ = RandomCrop(ratio=0.75)
    cj_ = ColorJitter(brightness=0.4, saturation=0.4, hue=0.4)
    rb_ = RandomBlur(p=0.5, r=(2, 3))
    rsf = RandomShift(p=0.5, ratio=0.15)
    rs_ = Resize(size=(448, 448))
    tt_ = ToTensor()
    gco = ToGridCellOffset((448, 448), (7, 7))
    img_trans = Compose([rhf, rc_, cj_, rb_, rsf, rs_, tt_])
    box_trans = Compose([rhf, rc_, rsf, rs_, gco])

    dataloader = MakeDataLoader(
        dataset=VOCDataset(config, phase='train',
                           img_transform=img_trans,
                           box_transform=box_trans),
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True)

    exe = Execute(config=config, dataloader=dataloader)

    exe.train()

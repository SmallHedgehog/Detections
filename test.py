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
from lib._parser_config import parser_config

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s:%(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='test object-detection of single stage.')
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

    rs_ = Resize(size=(448, 448))
    tt_ = ToTensor()
    img_trans = Compose([rs_, tt_])

    dataloader = MakeDataLoader(
        dataset=VOCDataset(config, phase='test',
                           img_transform=img_trans),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False)

    exe = Execute(config=config, dataloader=dataloader, phase='test')

    exe.test()

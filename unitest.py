import logging as log
import argparse

import yaml
import numpy as np
import cv2
import torch

from easydict import EasyDict

from lib.models.network import Yolo
from lib.dataset.parser.VOCParser import VOCParser
from lib.dataset.VOCDataset import VOCDataset
from lib.transform import RandomHorizontalFlip, Compose

log.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=log.DEBUG)

parser = argparse.ArgumentParser(description='unitest')
parser.add_argument('--config', type=str, default='cfgs/yolo.yaml')
args = parser.parse_args()

with open(args.config) as rptr:
    config = EasyDict(yaml.load(rptr))


def test_yolo():
    x = torch.randn(1, 3, 448, 448)
    y = Yolo(num_boxes=3, num_classes=20, weights_file=None)(x)

    log.info('Size: {}, {}'.format(y.size(), y.view(49, -1).size()))

def test_VOCParser():
    file = 'H:/dataset/VOC07/VOCdevkit/VOC2007/Annotations/000001.xml'
    ps = VOCParser()
    ps.parser(parser_file=file)

    log.info('Boxes: {}'.format(ps.__repr__()))

def test_VOCDatest():
    voc = VOCDataset(config, phase='train')
    img, boxes = voc[0]
    print(img.size, boxes)

def test_RandomHorizontalFlip():
    rhf = RandomHorizontalFlip()
    img_trans = Compose([rhf])
    box_trans = Compose([rhf])
    voc = VOCDataset(config, phase='train', img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[0]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # test_yolo()
    # test_VOCParser()
    # test_VOCDatest()
    test_RandomHorizontalFlip()

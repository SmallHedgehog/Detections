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
from lib.dataset.parser import Box
from lib.transform import RandomHorizontalFlip, Compose
from lib.transform import RandomCrop, ColorJitter
from lib.transform import RandomBlur, RandomShift
from lib.transform import Resize, ToTensor
from lib.transform import ToGridCellOffset
from lib.loss import Yolov1Loss
from lib.execute import Execute

log.basicConfig(
    format='[%(levelname)s] %(asctime)s:%(pathname)s:%(lineno)s:%(message)s', level=log.DEBUG)

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
    voc = VOCDataset(config, phase='train',
                     img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[0]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_RandomCrop():
    rc = RandomCrop(ratio=0.7)
    img_trans = Compose([rc])
    box_trans = Compose([rc])
    voc = VOCDataset(config, phase='train',
                     img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[1]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_ColorJitter():
    cj = ColorJitter(brightness=0.4, saturation=0.2, hue=0.2)
    img_trans = Compose([cj])
    voc = VOCDataset(config, phase='train', img_transform=img_trans)
    img, boxes = voc[2]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_RandomBlur():
    rb = RandomBlur(p=0.5, r=(2, 3))
    img_trans = Compose([rb])
    voc = VOCDataset(config, phase='train', img_transform=img_trans)
    img, boxes = voc[2]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_RandomShift():
    rs = RandomShift(p=0.5, ratio=0.1)
    img_trans = Compose([rs])
    box_trans = Compose([rs])
    voc = VOCDataset(config, phase='train', img_transform=img_trans,
                     box_transform=box_trans)
    img, boxes = voc[0]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_Resize():
    resize = Resize(size=(448, 448))
    img_trans = Compose([resize])
    box_trans = Compose([resize])
    voc = VOCDataset(config, phase='train', img_transform=img_trans,
                     box_transform=box_trans)
    img, boxes = voc[2]

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print(img.shape)
    for box in boxes:
        pt1, pt2 = box.points()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_ToTensor():
    toTensor = ToTensor()
    img_trans = Compose([toTensor])
    box_trans = Compose([toTensor])
    voc = VOCDataset(config, phase='train', img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[4]
    print(img.size(), type(boxes), boxes, boxes.size())

    img = img.permute(1, 2, 0)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print(img.shape)
    img_h, img_w = img.shape[:2]
    for box in list(boxes):
        class_idx, cx, cy, w, h = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        pt1, pt2 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_transform():
    rhf = RandomHorizontalFlip(p=0.5)
    rc  = RandomCrop(ratio=0.8)
    cj  = ColorJitter(brightness=0.4, saturation=0.4, hue=0.4)
    rb  = RandomBlur(p=0.5, r=(2, 3))
    rs  = RandomShift(p=0.5, ratio=0.1)
    rs_ = Resize(size=(448, 448))
    tt  = ToTensor()
    img_trans = Compose([rhf, rc, cj, rb, rs, rs_, tt])
    box_trans = Compose([rhf, rc, rs, rs_, tt])
    voc = VOCDataset(config, phase='train', img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[4]
    print(img.size(), type(boxes), boxes, boxes.size())

    img = img.permute(1, 2, 0)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    print(img.shape)
    img_h, img_w = img.shape[:2]
    for box in list(boxes):
        class_idx, cx, cy, w, h = int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
        pt1, pt2 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_ToGridCellOffset():
    rs_ = Resize(size=(448, 448))
    tt  = ToTensor()
    gco = ToGridCellOffset(img_size=(448, 448), grid_size=(7, 7))
    img_trans = Compose([rs_, tt])
    box_trans = Compose([rs_, gco])
    voc = VOCDataset(config, phase='train', img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[4]
    print(img.size(), type(boxes), len(boxes))
    print(boxes[0].size(), boxes[1].size())
    print(boxes[0])
    print(boxes[1])
    img = img.permute(1, 2, 0)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes[1]:
        i, j, _, o_x, o_y, w, h = int(box[0]), int(box[1]), int(box[2]), float(box[3]), float(box[4]), float(
            box[5]), float(box[6])
        w, h = w * 448, h * 448
        x, y = (i + o_x) * (448 / 7.) - w / 2, (j + o_y) * (448 / 7.) - h / 2
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        # Center point
        cv2.line(img, (int(x + w / 2), int(y + h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 10)

    # Draw grid
    inter_ = 448 // 7
    for i in range(7):
        cv2.line(img, (0, inter_ * i), (448, inter_ * i), (0, 0, 255), 2)
    for j in range(7):
        cv2.line(img, (inter_ * j, 0), (inter_ * j, 448), (0, 0, 225), 2)

    cv2.imshow('transform', img)
    cv2.waitKey(0)

def test_Yolov1Loss():
    v1_loss = Yolov1Loss(weight_coord=1., weight_noobject=1., num_boxes=2, num_classes=20, grid_size=(7, 7))
    print(v1_loss)
    img_size, grid_size = (448, 448), (7, 7)
    rs_ = Resize(size=img_size)
    tt  = ToTensor()
    gco = ToGridCellOffset(img_size=img_size, grid_size=grid_size)
    img_trans = Compose([rs_, tt])
    box_trans = Compose([rs_, gco])
    voc = VOCDataset(config, phase='train', img_transform=img_trans, box_transform=box_trans)
    img, boxes = voc[4]
    # print(img.size(), type(boxes), boxes[0].size(), boxes[1].size())
    img = img.unsqueeze(0)
    grid = boxes[0].unsqueeze(0)
    box = boxes[1].unsqueeze(0)
    # print(img.size(), type(boxes), grid.size(), box.size())

    net = Yolo(num_boxes=3, num_classes=20, grid_size=grid_size)
    # net = net.cuda()
    # img = img.cuda()
    out = net(img)
    # print(out.size())
    # print(out.view(out.size(0), grid_size[0], grid_size[1], -1).size())
    loss = v1_loss(out, (grid, box))
    print(loss)

def test_Execute():
    exe = Execute(config)


if __name__ == '__main__':
    # test_yolo()
    # test_VOCParser()
    # test_VOCDatest()
    # test_RandomHorizontalFlip()
    # test_RandomCrop()
    # test_ColorJitter()
    # test_RandomBlur()
    # test_RandomShift()
    # test_Resize()
    # test_ToTensor()
    # test_transform()
    # test_ToGridCellOffset()
    # test_Yolov1Loss()
    test_Execute()

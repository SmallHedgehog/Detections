import os.path as osp
import logging as log

from PIL import Image

from ._basicdataset import BasicDataset
from .parser.VOCParser import VOCParser

__all__ = ['VOCDataset']


class VOCDataset(BasicDataset):
    """ Dataset for file organization like PASCAL VOC.

    Args:
        config (dict): Configuration infomations
        phase (str): Training or testing, 'train' or 'test', default 'train'
        img_transform (torchvision.transforms.Compose): Tansforms to perform on the images
        box_transform (torchvision.transforms.Compose): Tansforms to perform on the boxes
    """
    def __init__(self, config, phase='train', img_transform=None, box_transform=None):
        super(VOCDataset, self).__init__()
        assert phase in ('train', 'test')

        self.root_dir = config.DATA_DIR
        self.phase = phase
        self.img_trans = img_transform
        self.box_trans = box_transform
        self.parser = VOCParser()

        self.label2class = {}
        with open(config.LABEL_NAME_FILE) as rptr:
            for idx, line in enumerate(rptr.readlines()):
                self.label2class[line.strip()] = idx

        if self.phase == 'train':
            file = osp.join(config.DATA_DIR, 'train.txt')
        else:
            file = osp.join(config.DATA_DIR, 'test.txt')
        with open(file) as rptr:
            self.file_name_sets = [line.rstrip() for line in rptr.readlines()]

        log.info('Dataset loaded, PHASE: {}, SIZE: {}'.format(self.phase, len(self.file_name_sets)))

    def __len__(self):
        return len(self.file_name_sets)

    def __getitem__(self, index):
        _, boxes = self.parser.parser(
            osp.join(self.root_dir, 'Annotations', self.file_name_sets[index] + '.xml'))
        for idx in range(len(boxes)):
            boxes[idx].class_idx = self.label2class[boxes[idx].class_name]
        img = Image.open(osp.join(self.root_dir, 'JPEGImages', self.file_name_sets[index] + '.jpg'))

        # Transform
        if self.img_trans is not None:
            img = self.img_trans(img)
        if self.box_trans is not None:
            boxes = self.box_trans(boxes)

        return img, boxes

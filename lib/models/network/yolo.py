import numpy as np
import torch

from itertools import product

from ..backbone import yolo as yolo_backbone
from ..head import yolo as yolo_head
from ._net import BasicNet

__all__ = ['Yolo']


class Yolo(BasicNet):
    def __init__(self, num_boxes=3, num_classes=20, grid_size=(7, 7), weights_file=None):
        super(Yolo, self).__init__()

        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size

        self.backbone = yolo_backbone.Yolo()
        self.head = yolo_head.Yolo(num_boxes=num_boxes, num_classes=num_classes, grid_size=grid_size)

        if weights_file is not None:
            self.load_weights(weights_file)
        else:
            self.init_weights(slope=0.1)

    def predict(self, x):
        B, _, H, W = x.size()
        device = x.device
        out = self.forward(x)
        out = out.view(x.size(0), self.grid_size[0], self.grid_size[1], -1)
        # Confidence
        confidence = out[:, :, :, :self.num_boxes].sigmoid()
        confidence, conf_index = torch.max(confidence.view(-1, self.num_boxes), dim=1)
        # Bounding boxes
        boxes = out[:, :, :, self.num_boxes:(5 * self.num_boxes)].sigmoid()
        boxes = boxes.view(-1, self.num_boxes, 4)
        conf_index = (conf_index.unsqueeze(-1).repeat(1, 4)).unsqueeze(1)
        boxes = torch.gather(boxes, 1, conf_index).view(-1, 4)
        grid_yx = torch.Tensor(list(product(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]))))
        grid_yx = grid_yx.repeat(x.size(0), 2).to(device)
        boxes[:, 0] = (grid_yx[:, 1] + boxes[:, 0]) * (W / self.grid_size[0])
        boxes[:, 1] = (grid_yx[:, 0] + boxes[:, 1]) * (H / self.grid_size[1])
        boxes[:, 2] = boxes[:, 2] * W
        boxes[:, 3] = boxes[:, 3] * H
        # Class score
        class_score = out[:, :, :, (5 * self.num_boxes):].softmax(-1)
        class_score, class_idx = torch.max(class_score, -1)
        return confidence.view(B, -1), boxes.view(B, -1, 4), \
               (class_score.view(B, -1), class_idx.view(B, -1))

    def forward(self, x):
        x = self.backbone(x)

        return self.head(x)

    def __repr__(self):
        return self.__class__.__name__ + '(num_boxes={}, num_classes={}, grid_size={})' \
                ''.format(self.num_boxes, self.num_classes, self.grid_size)

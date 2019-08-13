from ..backbone import yolo as yolo_backbone
from ..head import yolo as yolo_head
from ._basicnet import BasicNet

__all__ = ['Yolo']


class Yolo(BasicNet):
    def __init__(self, num_boxes=3, num_classes=20, weights_file=None):
        super(Yolo, self).__init__()

        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.backbone = yolo_backbone.Yolo()
        self.head = yolo_head.Yolo(num_boxes=num_boxes, num_classes=num_classes)

        if weights_file is not None:
            self.load_weights(weights_file)
        else:
            self.init_weights(slope=0.1)

    def forward(self, x):
        x = self.backbone(x)

        return self.head(x)
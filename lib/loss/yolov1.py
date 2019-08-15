import torch.nn as nn

__all__ = ['Yolov1Loss']


class Yolov1Loss(nn.Module):
    """ Yolov1 loss. reference to https://arxiv.org/pdf/1506.02640.pdf.

    Args:
        weight_coord (float): weight of bounding box coordinates
        weight_noobject (float): weight of regions without target boxes
        weight_object (float): weight of regions with target boxes
        weight_class (float): weight of categorical predictions
        num_boxes (int): the number of box of each cell. Default value is 3
        num_classes (int): the number of category. Default value is 20
        grid_size (tuple 2): grid size. Default size(7 x7)
    """
    def __init__(self, weight_coord, weight_noobject, weight_object, weight_class,
                 num_boxes=3, num_classes=20, grid_size=(7, 7)):
        super(Yolov1Loss, self).__init__()
        self.weight_coord = weight_coord
        self.weight_noobject = weight_noobject
        self.weight_object = weight_object
        self.weight_class = weight_class
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size

    def forward(self, predication, target):
        """
        Args:
            predication (torch.FloatTensor): Tensor with shape
        (B x grid_size[0] x grid_size[1] x (num_boxes * 5 + num_classes)
            target (tuple 2): (grid mask with shape torch.int(B x grid_size[0] x grid_size[1])
        that indicates have object in cell of grid when cell's value is equal to 1,
        list[list] that indicates each example in B have a list of
        [grid_x, grid_y, class_idx, offset_x, offset_y, width, height])
        """
        pass

import torch.nn as nn
import torch

__all__ = ['Yolov1Loss']


class Yolov1Loss(nn.Module):
    """ Yolov1 loss. reference to https://arxiv.org/pdf/1506.02640.pdf.

    Args:
        weight_coord (float): weight of bounding box coordinates
        weight_noobject (float): weight of regions without target boxes
        num_boxes (int): the number of box of each cell. Default value is 3
        num_classes (int): the number of category. Default value is 20
        grid_size (tuple 2): grid size. Default size(7 x7)
    """
    def __init__(self, weight_coord, weight_noobject, num_boxes=2, num_classes=20, grid_size=(7, 7)):
        super(Yolov1Loss, self).__init__()
        self.weight_coord = weight_coord
        self.weight_noobject = weight_noobject
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.grid_size = grid_size

    def forward(self, predication, target):
        """
        Args:
            predication (torch.FloatTensor): Tensor with shape
        (B, grid_size[0] x grid_size[1] x (num_boxes * 5 + num_classes)
            target (tuple 2): (grid mask with shape torch.int(B, grid_size[0], grid_size[1])
        that indicates have object in cell of grid when cell's value is equal to 1,
        list[list] that indicates each example in B have a list of
        [grid_x, grid_y, class_idx, offset_x, offset_y, width, height])
        """
        pred = predication.view(predication.size(0), self.grid_size[0], self.grid_size[1], -1)
        grid, boxes = target

        loss = self._forward_one(pred[0], grid[0], boxes[0])
        for batch_idx in range(1, pred.size(0)):
            loss += self._forward_one(pred[batch_idx], grid[batch_idx], boxes[batch_idx])
        return loss / pred.size(0)

    def _forward_one(self, pred, grid, boxes):
        """
        Args:
            pred (torch.FloatTensor, 3-D): (grid_size[0], grid_size[1], (num_boxes * 5 + num_classes)
            grid (torch.FloatTensor, 2-D): (grid_size[0], grid_size[1])
            boxes (torch.FloatTensor, 2-D): (number of box, 7)
        """
        # Object index in grid
        object_index = (grid == 1).nonzero()
        # No object index in grid
        noobject_index = (grid != 1).nonzero()
        box_obj_indx = boxes[:, :2].long()
        # Find all object index in a cell of grid
        if object_index.size(0) != 0:
            loss = self._fetch_object_one(object_index[0], box_obj_indx, boxes, pred)
            # No object loss
            if noobject_index.size(0) != 0:
                loss += self._noobject_loss(pred[noobject_index[:, 0], noobject_index[:, 1]])
        else:
            # No object loss
            loss = self._noobject_loss(pred[noobject_index[:, 0], noobject_index[:, 1]])

        # Object loss
        for obj_idx in range(1, object_index.size(0)):
            loss += self._fetch_object_one(object_index[obj_idx], box_obj_indx, boxes, pred)

        return loss

    def _fetch_object_one(self, object_index, box_obj_indx, boxes, pred):
        object_index = object_index[torch.Tensor([1, 0]).long()]
        box_idxes = (((box_obj_indx == object_index).sum(dim=1)) == 2).nonzero()
        # Shape from [-1, 1] to [-1]
        box_idxes = box_idxes.view(-1)
        # Get all box of same cell
        boxes_of_cell = boxes[box_idxes]
        # Calculate IOUs
        x, y = object_index
        object_loss = self._object_loss(pred[y, x], boxes_of_cell)
        return object_loss

    def _calc_iou(self, pred_box, target_box):
        """ Compute IOU of given two boxes.

        Args:
            pred_box (torch.FloatTensor): [offset_x, offset_y, w, h]
            target_box (torch.FloatTensor): [grid_x, grid_y, class_idx, offset_x, offset_y, w, h]

        Returns:
            torch.FloatTensor: iou
        """
        x, y = target_box[:2]
        box_cx, box_cy = (x + target_box[3]) / self.grid_size[0], (y + target_box[4]) / self.grid_size[1]
        box_half_w, box_half_h = target_box[5] / 2, target_box[6] / 2
        pred_cx, pred_cy = (x + pred_box[0]) / self.grid_size[0], (y + pred_box[1]) / self.grid_size[1]
        pred_half_w, pred_half_h = pred_box[2] / 2, pred_box[3] / 2
        tb = torch.min(box_cx + box_half_w, pred_cx + pred_half_w) - \
             torch.max(box_cx - box_half_w, pred_cx - pred_half_w)
        lr = torch.min(box_cy + box_half_h, pred_cy + pred_half_h) - \
             torch.max(box_cy - box_half_h, pred_cy - pred_half_h)
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        iou = inter / (target_box[5] * target_box[6] + pred_box[2] * pred_box[3] - inter)
        return iou

    def _compute_ious(self, pred_detach, box):
        """ Compute IOUs.

        Args:
            pred_detach (torch.FloatTensor): Size (num_pred_boxes, 4) containing
        [offset_x, offset_y, w, h]
            box (torch.FloatTensor): Size (7). containing
        [grid_x, grid_y, class_idx, offset_x, offset_y, w, h]

        Returns:
            float: best iou of predications for a given bounding box
            int: best index of predications for a given bounding box
        """
        best_iou, best_idx = -1., -1
        for idx in range(pred_detach.size(0)):
            iou = self._calc_iou(pred_detach[idx], box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return best_iou, best_idx

    def _noobject_loss(self, cell_pred):
        """ Calculate no object cell confidence loss.

        Args:
            cell_pred (torch.FloatTensor, 2-D): (number of no object cells, num_boxes * 5 + num_classes)

        Returns:
            torch.FloatTensor: Size (1)
        """
        confidences = cell_pred[:, :self.num_boxes].sigmoid()
        return self.weight_noobject * (confidences ** 2).sum()

    def _object_loss(self, cell_pred, cell_boxes):
        """ Calculate IOUs. Using best IOU.

        Args:
            cell_pred (torch.FloatTensor, 1-D): (num_boxes * 5 + num_classes)
            cell_boxes (torch.FloatTensor, 2-D): (num_cell_boxes, 7)
        """
        # Confidence
        confidences = cell_pred[:self.num_boxes].sigmoid()
        # Bounding box
        pred_boxes = cell_pred[self.num_boxes:(5 * self.num_boxes)].view(-1, 4).sigmoid()
        # Class score
        class_scores = cell_pred[(5 * self.num_boxes):].softmax(0)

        BEST_IOU, BEST_pred_idx, BEST_box_idx = -1., -1, -1
        for box_idx in range(cell_boxes.size(0)):
            iou, idx = self._compute_ious(pred_boxes.detach(), cell_boxes[box_idx])
            if iou > BEST_IOU:
                BEST_IOU = iou
                BEST_pred_idx = idx
                BEST_box_idx = box_idx

        iou = self._calc_iou(pred_boxes[BEST_pred_idx], cell_boxes[BEST_box_idx])
        object_iou_loss = (confidences[BEST_pred_idx] - iou) ** 2
        object_cls_loss = (1. - class_scores[cell_boxes[BEST_box_idx][2].long()]) ** 2
        object_coord_offset_loss = ((pred_boxes[BEST_pred_idx][0] - cell_boxes[BEST_box_idx][3]) ** 2) + \
                                   ((pred_boxes[BEST_pred_idx][1] - cell_boxes[BEST_box_idx][4]) ** 2)
        object_coord_wh_loss = ((torch.sqrt(pred_boxes[BEST_pred_idx][2]) -
                                 torch.sqrt(cell_boxes[BEST_box_idx][5])) ** 2) + \
                               ((torch.sqrt(pred_boxes[BEST_pred_idx][3]) -
                                 torch.sqrt(cell_boxes[BEST_box_idx][6])) ** 2)
        return self.weight_coord * (object_coord_offset_loss + object_coord_wh_loss) + \
               object_iou_loss + object_cls_loss

    def __repr__(self):
        return self.__class__.__name__ + '(weight_coord={}, weight_noobject={}, ' \
                                         'num_boxes={}, num_classes={}, grid_size={})'.format(
            self.weight_coord, self.weight_noobject, self.num_boxes, self.num_classes, self.grid_size)

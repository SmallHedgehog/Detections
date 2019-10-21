import numpy as np
import torch

__all__ = ['NMS', 'SoftNMS']

class SoftNMS(object):
    """
    Args:
        nms_thresh: (Number [0-1]): For filter overlap bounding box.
    """
    def __init__(self, nms_thresh, use_classes=False):
        super(SoftNMS, self).__init__()
        self.nms_thresh = nms_thresh
        self.use_classes = use_classes

    def _xywh2xyxy(self, boxes):
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    def softnms(self, boxes, _type='linear', sigma=0.5, conf_thresh=0.5):
        self._xywh2xyxy(boxes)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        D, S = [], []
        while len(areas) > 0:
            max_indx = torch.argmax(boxes[:, 5])
            index = list(torch.arange(len(areas)))

            max_box = boxes[max_indx]
            D.append(max_box)
            this_area = areas[max_indx]

            del index[int(max_indx)]
            index = torch.LongTensor(index)
            boxes = boxes[index]
            areas = areas[index]

            lu_x = torch.max(max_box[0], boxes[:, 0])
            rb_x = torch.min(max_box[2], boxes[:, 2])
            lu_y = torch.max(max_box[1], boxes[:, 1])
            rb_y = torch.min(max_box[3], boxes[:, 3])

            w = torch.max(torch.FloatTensor([0.0]), rb_x - lu_x)
            h = torch.max(torch.FloatTensor([0.0]), rb_y - lu_y)
            inter = w * h
            iou = inter / (areas + this_area - inter)

            indx = torch.where(iou > self.nms_thresh)[0]
            if _type == 'linear':
                boxes[indx, 4] *= (1 - iou[indx])
            elif _type == 'gaussian':
                boxes[indx, 4] *= torch.exp(-torch.pow(iou[indx], 2) / sigma)
            else:
                raise ValueError

        ret = []
        for indx in range(len(D)):
            if D[indx][4] >= conf_thresh:
                ret.append(D[indx])
        return ret

    def __call__(self, boxes):
        """
        Args:
            boxes (torch.FloatTensor): The prediction of boxes, with shape
        (number of boxes, 6), i.g. boxes[0] = (x, y, w, h, confidence, class_idx)
        """
        self._xywh2xyxy(boxes)
        conf = boxes[:, 4]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = conf.argsort(descending=True)

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            lu_x = torch.max(boxes[i, 0], boxes[order[1:]][:, 0])
            rb_x = torch.min(boxes[i, 2], boxes[order[1:]][:, 2])
            lu_y = torch.max(boxes[i, 1], boxes[order[1:]][:, 1])
            rb_y = torch.min(boxes[i, 3], boxes[order[1:]][:, 3])

            w = torch.max(torch.FloatTensor([0.0]), rb_x - lu_x)
            h = torch.max(torch.FloatTensor([0.0]), rb_y - lu_y)
            inter = w * h
            iou = inter / (areas[order[1:]] + areas[i] - inter)

            if self.use_classes:
                indx = torch.where((iou <= self.nms_thresh) |
                               (boxes[i, 5] != boxes[order[1:]][:, 5]))[0]
            else:
                indx = torch.where(iou <= self.nms_thresh)[0]
            order = order[indx + 1]

        return keep


class NMS(object):
    """ Performs Non-maxinum suppression(NMS) on the bounding boxes,
    filtering boxes with a high overlap.

    Args:
        nms_thresh (Number [0-1]): For filter overlap bounding box.
        conf_thressh (Number [0-1]): Confidence threshold.
    """
    def __init__(self, nms_thresh, conf_thresh):
        super(NMS, self).__init__()
        self.nms_threshold = nms_thresh
        self.confidence_threshold = conf_thresh

    def __call__(self, confidence, boxes, class_infos):
        """
        Args:
            confidence (torch.FloatTensor): The confidence of the boxes, with shape (number of boxes)
            boxes (torch.FloatTensor): The prediction of boxes, with shape (number of boxes, 4)
            class_infos (tuple): (class score, class id)

        Returns:
            torch.FloatTensor: Pruned boxes
            tuple: Pruned (class score, class id)
        """
        if boxes.numel() == 0:
            return boxes, class_infos

        _, order = confidence.sort(0, descending=True)
        # keep_indexs = order
        center_xy = boxes[:, :2]
        wh = boxes[:, 2:]
        bboxes = torch.cat([center_xy - wh / 2, center_xy + wh / 2], 1)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 2]
        areas = (x2 - x1) * (y2 - y1)

        class_score, class_id = class_infos

        keep = []
        while order.numel() > 0:
            cur_idx = order[0]
            if confidence[cur_idx] > self.confidence_threshold:
                keep.append(order[0].item())
            else:
                break

            dx = (x2[order[1:]].clamp(min=x2[cur_idx].item()) -
                  x1[order[1:]].clamp(max=x1[cur_idx].item())).clamp(min=0)
            dy = (y2[order[1:]].clamp(min=y2[cur_idx].item()) -
                  y1[order[1:]].clamp(max=y1[cur_idx].item())).clamp(min=0)
            intersections = dx * dy
            unions = (areas[cur_idx] + areas[order[1:]] - intersections)
            ious = intersections / unions

            cur_class_id = class_id[cur_idx]
            oth_class_id = class_id[order[1:]]

            indexs = (1 - ((ious > self.nms_threshold) &
                           (oth_class_id == cur_class_id))).nonzero().squeeze()
            if indexs.numel() == 0:
                break
            order = order[indexs + 1]
        keep_indexs = torch.LongTensor(keep)
        if boxes.is_cuda:
            return boxes[keep_indexs].cpu(), (class_score[keep_indexs].cpu(), class_id[keep_indexs].cpu())
        return boxes[keep_indexs], (class_score[keep_indexs], class_id[keep_indexs])


if __name__ == '__main__':
    nms = SoftNMS(nms_thresh=0.3, use_classes=False)

    boxes = torch.FloatTensor([
        [50, 50, 10, 10, 0.9, 1],
        [48, 48, 10, 10, 0.8, 1],
        [50, 50, 10, 10, 0.8, 1],
        [50, 50, 10, 10, 0.9, 0]
    ])

    print(nms.softnms(boxes, _type='linear', sigma=0.5))

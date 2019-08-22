import torch

__all__ = ['NMS']


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

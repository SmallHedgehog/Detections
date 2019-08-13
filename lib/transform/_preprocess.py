import collections
import random
import logging as log

from PIL import Image

__all__ = ['RandomHorizontalFlip', 'RandomCrop']


class RandomHorizontalFlip(object):
    """ Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Defualt value is 0.5
        isflip (bool): Whether flip, flip if isflip is True
        w (Number): image width

    Note:
        Horizontally flip the given image and box respectively.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.isflip = False
        self.w = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return [self._rf_box(box) for box in data]
        elif isinstance(data, Image.Image):
            return self._rf_pil(data)
        else:
            log.error('Only works with <Box of lists> or <PIL image>, type:{}'.format(type(data)))

    def _rf_pil(self, img):
        """ Flip a image of :class:`Image.Image` randomly. """
        self._get_flip()
        self.w = img.size[0]
        if self.isflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _rf_box(self, box):
        """ Flip Box object. """
        if self.isflip and self.w is not None:
            box.horizontal_flip(self.w)
        return box

    def _get_flip(self):
        self.isflip = random.random() < self.p

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(object):
    """ Crop the given PIL Image at a random location.

    Args:
        ratio (Number [0-1]): Crop ratio of the given PIL Image
        crop (tuple 4): Crop area, (top_left_x, top_left_y, width, height)

    Note:
        Crop the given PIL Image and box.
    """
    def __init__(self, ratio):
        super(RandomCrop, self).__init__()
        self.ratio = ratio
        self.crop = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._rc_box(data)
        elif isinstance(data, Image.Image):
            return self._rc_pil(data)
        else:
            log.error('Only works with <Box of lists> or <PIL image>, type:{}'.format(type(data)))

    def _rc_pil(self, img):
        """ Crop the given PIL Image. """
        im_w, im_h = img.size
        self._get_crop(im_w, im_h)
        left, upper, w, h = self.crop
        return img.crop((left, upper, left + w, upper + h))

    def _rc_box(self, boxes):
        """ Modify Box object of list of boxes. """
        modified_boxes = []
        for idx in range(len(boxes)):
            if boxes[idx].crop(self.crop):
                modified_boxes.append(boxes[idx])
        return modified_boxes

    def _get_crop(self, im_w, im_h):
        dw, dh = int(im_w * self.ratio), int(im_h * self.ratio)
        crop_left_top_x = random.randint(0, im_w - dw)
        crop_left_top_y = random.randint(0, im_h - dh)
        self.crop = (crop_left_top_x, crop_left_top_y, dw, dh)

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={})'.format(self.ratio)

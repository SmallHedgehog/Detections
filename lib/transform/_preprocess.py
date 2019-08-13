import collections
import random
import logging as log

from PIL import Image

__all__ = ['RandomHorizontalFlip']


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

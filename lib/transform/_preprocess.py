import collections
import random
import logging as log

import numpy as np
import torch
import torchvision.transforms as transform

from PIL import Image, ImageFilter

__all__ = ['RandomHorizontalFlip', 'RandomCrop', 'ColorJitter', 'RandomBlur',
           'RandomShift', 'Resize', 'ToTensor', 'ToGridCellOffset']


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


class ColorJitter(transform.ColorJitter):
    """ Randomly change the brightness, contrast and saturation of an image,
    inherited :class:`torchvision.transforms.ColorJitter`
    """
    def __init__(self, brightness=0, constrast=0, saturation=0, hue=0):
        super(ColorJitter, self).__init__(brightness, constrast, saturation, hue)


class RandomBlur(object):
    """ Randomly blur the given PIL Image by gaussian blur.

    Args:
        p (float): Probability of the image being flipped. Defualt value is 0.5
        r (tuple): Gaussian kernel. Defualt value is (2,)
    """
    def __init__(self, p=0.5, r=(2,)):
        super(RandomBlur, self).__init__()
        self.p = p
        self.r = self._check_input(r)

    def _check_input(self, r):
        if not isinstance(r, collections.Sequence):
            log.error('`r` must be a sequence.')
            raise ValueError('`r` must be a sequence.')
        return r

    def __call__(self, img):
        if img is None:
            return None
        elif isinstance(img, Image.Image):
            return self._rb_pil(img)
        else:
            log.error('Only works with <PIL image>, type:{}'.format(type(img)))

    def _rb_pil(self, img):
        if random.random() < self.p:
            radius = random.choice(self.r)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, r={})'.format(self.p, self.r)


class RandomShift(object):
    """ Randomly shift the given PIL Image.

    Args:
        p (float): Probability of the image being shifted. Defualt value is 0.5
        ratio (Number [0-1]): Shift ratio of the given PIL Image. Defualt value is 0.1
    Note:
        Shift the given PIL Image and bounding box respectively.
    """
    def __init__(self, p=0.5, ratio=0.1):
        super(RandomShift, self).__init__()
        self.p = p
        self.ratio = ratio
        self.is_shift = False
        self.AOI = None  # Area of interest
        self.offset = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._rs_box(data)
        elif isinstance(data, Image.Image):
            return self._rs_pil(data)
        else:
            log.error('Only works with <Box of lists> or <PIL image>, type:{}'.format(type(data)))

    def _rs_pil(self, img):
        """ Shift the given PIL Image randomly. """
        self._get_shift()
        if self.is_shift:
            img_w, img_h = img.size
            range_x = int(img_w * self.ratio / 2)
            range_y = int(img_h * self.ratio / 2)
            offset_x = random.randint(-range_x, range_x)
            offset_y = random.randint(-range_y, range_y)
            # Shift PIL Image
            np_img = np.asarray(img)
            shift_img = np.zeros(np_img.shape, dtype=np_img.dtype)
            self.offset = (offset_x, offset_y)
            if offset_x >= 0 and offset_y >= 0:
                shift_img[offset_y:, offset_x:, :] = np_img[:img_h-offset_y, :img_w-offset_x, :]
                self.AOI = (offset_x, offset_y, img_w-offset_x, img_h-offset_y)
            elif offset_x >= 0 and offset_y < 0:
                shift_img[:img_h+offset_y, offset_x:, :] = np_img[-offset_y:img_h, :img_w-offset_x, :]
                self.AOI = (offset_x, 0, img_w-offset_x, img_h+offset_y)
            elif offset_x < 0 and offset_y >= 0:
                shift_img[offset_y:, :img_w+offset_x, :] = np_img[:img_h-offset_y, -offset_x:img_w, :]
                self.AOI = (0, offset_y, img_w+offset_x, img_h-offset_y)
            elif offset_x < 0 and offset_y < 0:
                shift_img[:img_h+offset_y, :img_w+offset_x, :] = np_img[-offset_y:img_h, -offset_x:img_w, :]
                self.AOI = (0, 0, img_w+offset_x, img_h+offset_y)
            img = Image.fromarray(shift_img)
        return img

    def _rs_box(self, boxes):
        """ Adjust the object of Box. """
        if self.is_shift and self.AOI is not None:
            adjust_boxes = []
            for box in boxes:
                if box.shift(self.AOI, self.offset):
                    adjust_boxes.append(box)
            return adjust_boxes
        return boxes

    def _get_shift(self):
        self.is_shift = random.random() < self.p

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, ratio={})'.format(self.p, self.ratio)


class Resize(object):
    """ Resize the given PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like (h, w),
    output size will be matched to this. If size is an int, smaller edge of the image
    will be matched to this number. i.e, if height > width, then image will be rescaled
    to (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``

    Note:
        Resize the given PIL Image and bounding box respectively.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        super(Resize, self).__init__()
        assert isinstance(size, int) or (isinstance(size, collections.Sequence) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.rescaled_ratio = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return [self._resize_box(box) for box in data]
        elif isinstance(data, Image.Image):
            return self._resize_pil(data)
        else:
            log.error('Only works with <Box of lists> or <PIL image>, type:{}'.format(type(data)))

    def _resize_pil(self, img):
        """ Resize the given PIL Image according to self.size.

        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        w, h = img.size
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                self.rescaled_ratio = None
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                self.rescaled_ratio = (ow / w, self.size / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                self.rescaled_ratio = (self.size / h, oh / h)
            return img.resize((ow, oh), self.interpolation)
        else:
            self.rescaled_ratio = (self.size[0] / w, self.size[1] / h)
            return img.resize(self.size, self.interpolation)

    def _resize_box(self, box):
        """ Resize object of Box. """
        if self.rescaled_ratio is not None:
            box.resize(self.rescaled_ratio)
        return box

    def __repr__(self):
        return self.__class__.__name__ + '(size={}, interpolation={})'.format(self.size, self.interpolation)


class ToTensor(object):
    """ Convert a ``PIL Image`` or `lib.dataset.parser.box.Box` to tensor.

    Converts a PIL Image in the range [0, 255] to a torch.FloatTensor of shape (C x H x W)
    in the range [0.0, 1.0]. Convert the list of box. Box objects to tensor of dimension
    [number of boxes in list, 5] containing [class_idx, center_x, center_y, width, height]
    of range [0.0, 1.0] except class_idx for every detection.

    Note:
        Convert the given PIL Image and the list of box.Box objects respectively.
    """
    def __init__(self):
        super(ToTensor, self).__init__()
        self.img_tensor = transform.ToTensor()
        self.img_size = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._tensor_box(data)
        elif isinstance(data, Image.Image):
            return self._tensor_pil(data)
        else:
            log.error('Only works with <Box of lists> or <PIL image>, type:{}'.format(type(data)))

    def _tensor_pil(self, img):
        self.img_size = img.size
        return self.img_tensor(img)

    def _tensor_box(self, boxes):
        """ Convert a list of object of box.Box to a tensor.

        Returns:
            torch.FloatTensor or None: If not None, return tensor of dimension
        [number of boxes in list, 5] of range [0.0, 1.0]
        """
        if self.img_size is not None:
            list_boxes = []
            for box in boxes:
                list_boxes.append(box.toTensor(self.img_size))
            return torch.from_numpy(np.array(list_boxes, dtype=np.float))
        else:
            log.error('The attribute img_size of {} is None'.format(self.__class__.__name__))
            raise ValueError('The attribute img_size of {} is None'.format(self.__class__.__name__))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToGridCellOffset(object):
    """ Gets the relative position of the center of the box relative to the grid cell, the
    relative position value is in range of [0.0, 1.0].

    Args:
        img_size (tuple 2): The given PIL Image size. (width, height)
        grid_size (tuple 2): Grid size. i.e grid is 7 x 7

    Note:
        Only works on box of class :class:`lib.dataset.parser.box.Box`.
    """
    def __init__(self, img_size, grid_size):
        super(ToGridCellOffset, self).__init__()
        self.img_size = self._check_input(img_size)
        self.grid_size = self._check_input(grid_size)

    def _check_input(self, _size):
        assert isinstance(_size, collections.Sequence) and (len(_size) == 2)
        return _size

    def __call__(self, boxes):
        if isinstance(boxes, collections.Sequence):
            return self._gco_box(boxes)
        else:
            log.error('Only works with <Box of lists>, type:{}'.format(type(boxes)))

    def _gco_box(self, boxes):
        """ Generate grid that each cell of grid indicate wether have object. Grid shape is
        (self.grid_size x self.grid_size). Generate bounding box list of dimension
        [len(boxes), 7] containing [grid_x, grid_y, class_idx, offset_x, offset_y, width, height].
            grid_x and grid_y: x coordinate and y coordinate of gird.
            offset_x and offset_y: the relative position of the center of the box relative to the
        grid cell, their value is in range of [0.0, 1.0].

        Returns:
            torch.IntTensor: Grid of size(self.grid_size x self.grid_size)
            torch.FloatTensor: Bounding box with size(len(boxes), 7)
        """
        grid, list_boxes = torch.zeros(self.grid_size, dtype=torch.long), []
        for box in boxes:
            infos_box = box.grid_cell_offset(self.img_size, self.grid_size)
            # Note
            grid[infos_box[1], infos_box[0]] = 1
            list_boxes.append(infos_box)
        return grid, torch.from_numpy(np.array(list_boxes, dtype=np.float))

    def __repr__(self):
        return self.__class__.__name__ + '(img_size={}, grid_size={})'.format(self.img_size, self.grid_size)

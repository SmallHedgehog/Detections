import numpy as np

__all__ = ['Compose']


class Compose(list):
    """ This is lightnet's own version of :class:`torchvision.transforms.Compose`.

    Note:
        The reason we have our own version is because this one offers more freedom to the user.
        For all intends and purposes this class is just a list.
        This `Compose` version allows the user to access elements through index, append items, extend it with another list, etc.
        When calling instances of this class, it behaves just like :class:`torchvision.transforms.Compose`.

    Note:
        I proposed to change :class:`torchvision.transforms.Compose` to something similar to this version,
        which would render this class useless. In the meanwhile, we use our own version
        and you can track `the issue`_ to see if and when this comes to torchvision.

    .. _the issue: https://github.com/pytorch/vision/issues/456
    """
    def __call__(self, data):
        for tf in self:
            data = tf(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ['
        for tf in self:
            format_string += '\n  {}'.format(tf.__class__.__name__)
        format_string += '\n]'
        return format_string


def pad_to_square(image, value=0):
    if isinstance(image, np.ndarray):
        W, H, _ = image.shape
    else:
        H, W = image.size
    diff = np.abs(H - W)
    pad1, pad2 = diff // 2, diff - diff // 2
    pad = (0, 0, pad1, pad2) if H <= W else (pad1, pad2, 0, 0)
    image = np.pad(image, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant', constant_values=(value, value))
    return image

def resize(image, size):
    return cv2.resize(image, (size, size))

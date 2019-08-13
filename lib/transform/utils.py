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

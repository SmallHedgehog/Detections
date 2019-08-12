__all__ = ['BasicParser']


class BasicParser(object):
    """ This is a abstraction class of parser file.
    There are 1 basic ways of useing this class:

    - Override the ``parser()`` function.
      This funcition return a list of :class:lib.dataset.parser.box.Box
    """
    def __init__(self):
        super(BasicParser, self).__init__()
        self.boxes = None

    def parser(self, parser_file):
        """ Abstraction function that can be overide in the derived class.

        Args:
            parser_file (string): path to file

        Returns:
            list: list of Box objects
        """
        raise NotImplementedError

    def __repr__(self):
        return '_'.join([box.__repr__() for box in self.boxes])

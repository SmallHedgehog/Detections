__all__ = ['BasicParser']


class BasicParser(object):
    """ This is a abstraction class of parser file.
    There are 1 basic ways of useing this class:

    - Override the ``parser()`` function.
      This funcition return a list of :class:lib.dataset.parser.box.Box

    Attributes:
        file_name (string): file name
        boxes (list): list of :class:lib.dataset.parser.box.Box objects
    """
    def __init__(self):
        super(BasicParser, self).__init__()
        self.file_name = ''
        self.boxes = []

    def parser(self, parser_file):
        """ Abstraction function that can be overide in the derived class.

        Args:
            parser_file (string): path to file
        """
        raise NotImplementedError

    def __repr__(self):
        return self.file_name + ' ' + '_'.join([box.__repr__() for box in self.boxes])

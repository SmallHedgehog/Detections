import logging as log
import xml.etree.ElementTree as ET

from .box import Box
from ._parser import BasicParser


class VOCParser(BasicParser):
    def __init__(self):
        super(VOCParser, self).__init__()

    def parser(self, parser_file):
        """ Parser XML file.

        Args:
            parser_file (string): path to file

        Returns:
            str: file name
            list: list of Box objects
        """
        try:
            with open(parser_file) as rptr:
                root = ET.fromstring(rptr.read())
                boxes, object_id = [], -1
                self.file_name = root.find('filename').text
                for obj in root.iter('object'):
                    class_name = obj.find('name').text
                    box_obj = obj.find('bndbox')

                    x_top_left = float(box_obj.find('xmin').text)
                    y_top_left = float(box_obj.find('ymin').text)
                    width = float(box_obj.find('xmax').text) - x_top_left
                    height = float(box_obj.find('ymax').text) - y_top_left

                    object_id += 1
                    boxes.append(Box(class_name, object_id, x_top_left, y_top_left, width, height))
                self.boxes = boxes

        except AttributeError as err:
            log.error('{}'.format(err))
            self.file_name, self.boxes = '', []
            raise Exception(err)
        except FileNotFoundError as err:
            log.error('{}'.format(err))
            self.file_name, self.boxes = '', []
            raise Exception(err)

        return self.file_name, self.boxes

__all__ = ['Box']


class Box(object):
    """ This is a generic bounding box representation.

    Attributes:
        class_name (string): Class name, default ''
        class_idx (int): Class index, default -1
        object_id (int): Object identifier for reid purpose, defualt 0
        x_top_left (Number): X pixel coordinate of the top left corner of the bounding box, default 0.
        y_top_left (Number): Y pixel coordinate of the top left corner of the bounding box, defautl 0.
        width (Number): Width of the bounding box in pixels, default 0.
        height (Number): Height of the bounding box in pixels, default 0.
    """

    def __init__(self, class_name='', object_id=0, x_top_left=0., y_top_left=0., width=0., height=0.):
        super(Box, self).__init__()

        self.class_name = class_name
        self.class_idx  = -1
        self.object_id  = object_id
        self.x_top_left = x_top_left
        self.y_top_left = y_top_left
        self.width = width
        self.height = height

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return self.serialize()

    def serialize(self):
        """ Serialize to string. """
        return ' '.join(
            map(lambda x: str(x),
                [self.class_name, self.object_id, self.x_top_left, self.y_top_left, self.width, self.height]))

    def deserialize(self, string):
        """ Deserialize from a string. """
        box_infos       = string.split(' ')
        self.class_name = box_infos[0]
        self.object_id  = int(box_infos[1])
        self.x_top_left = float(box_infos[2])
        self.y_top_left = float(box_infos[3])
        self.width      = float(box_infos[4])
        self.height     = float(box_infos[5])

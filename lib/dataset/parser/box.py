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

    def horizontal_flip(self, width):
        """ Horizontal flip a Box by change coordinates of x_top_left.

        Args:
            width (Number): image width
        """
        self.x_top_left = width - (self.x_top_left + self.width)

    def crop(self, crop):
        """ Adjust object of Box according to the croped area of the image.

        Args:
            crop (tuple 4): Crop area of an image, (top_left_x, top_left_y, width, height)

        Returns:
            bool: if False, the bounding box is not in the cropped area.
        """
        left, upper, w, h = crop
        if self.is_intersect(crop):
            x_right = self.x_top_left + self.width
            y_lower = self.y_top_left + self.height
            # X pixel coordinate and width of the bounding box in pixels
            if self.x_top_left < left:
                self.x_top_left = left
            if x_right > (left + w):
                self.width = left + w - self.x_top_left
            # Y pixel coordinate and height of the bounding box in pixels
            if self.y_top_left < upper:
                self.y_top_left = upper
            if y_lower > (upper + h):
                self.height = upper + h - self.y_top_left
            # To cropped image coordinate
            self.x_top_left -= left
            self.y_top_left -= upper
            return True
        return False

    def is_intersect(self, box):
        """ Determine whether two boxes intersect.

        Args:
            box: (tuple 4): A bounding box, (top_left_x, top_left_y, width, height)

        Returns:
            bool: if True, two box is intersected.
        """
        left, upper, w, h = box
        x_center_width = (self.x_top_left + self.width / 2) - (left + w / 2)
        y_center_width = (self.y_top_left + self.height / 2) - (upper + h / 2)
        x_width = self.width / 2 + w / 2
        y_width = self.height / 2 + h / 2
        if x_center_width <= x_width and y_center_width <= y_width:
            return True
        return False

    def shift(self, AOI, offset):
        """ Adjust the bounding box according to AOI(area of interest).

        Args:
            AOI (tuple 4): Area of interest. (top_left_x, top_left_y, width, height)
            offset (tuple 2): Shift offset. (x coordinate offset, y coordinate offset)

        Returns:
            bool: Determine if the bounding box is out of the AOI, if True the box
        is in the AOI.
        """
        x_offset, y_offset = offset
        aoi_x, aoi_y, aoi_w, aoi_h = AOI
        self.x_top_left += x_offset
        self.y_top_left += y_offset
        if (self.x_top_left + self.width) <= aoi_x or self.x_top_left >= (aoi_x + aoi_w):
            return False
        if (self.y_top_left + self.height) <= aoi_y or self.y_top_left >= (aoi_y + aoi_h):
            return False
        if self.x_top_left < aoi_x:
            self.width -= aoi_x - self.x_top_left
            self.x_top_left = aoi_x
        if (self.x_top_left + self.width) > (aoi_x + aoi_w):
            self.width = aoi_x + aoi_w - self.x_top_left
        if self.y_top_left < aoi_y:
            self.height -= aoi_y - self.y_top_left
            self.y_top_left = aoi_y
        if (self.y_top_left + self.height) > (aoi_y + aoi_h):
            self.height = aoi_y + aoi_h - self.y_top_left
        return True

    def resize(self, rescale_ratio):
        """ Rescale the bounding box according to size.

        Args:
            rescale_ratio (tuple 2): rescale ratio.
        """
        ratio_x, ratio_y = rescale_ratio
        self.x_top_left *= ratio_x
        self.y_top_left *= ratio_y
        self.width  *= ratio_x
        self.height *= ratio_y

    def points(self):
        """ Get top left corner and bottom right corner.

        Returns:
            point of top left corner, ponit of bottom right corner (tuple)
        """
        return (int(self.x_top_left), int(self.y_top_left)), \
               (int(self.x_top_left + self.width), int(self.y_top_left + self.height))

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

import torch.utils.data as Data

__all__ = ['BasicDataset']


class BasicDataset(Data.Dataset):
    """ This class is a subclass of the base :class:torch.utils.data.Dataset.
    """
    def __init__(self):
        super(BasicDataset, self).__init__()

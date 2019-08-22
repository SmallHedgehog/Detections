import logging as log

import torch
import torch.nn as nn

__all__ = ['BasicNet']


class BasicNet(nn.Module):
    """ This class provides an abstraction layer for every network implemented in this framework.
    There are 1 basic ways of useing this class:

    - Override the ``forward()`` function.
      This makes the networks behave just like PyTorch modules.
    """

    def __init__(self):
        super(BasicNet, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def load_weights(self, weight_file):
        """ This function will load the weights from a file.

        Args:
            weight_file (str): path to file
        """
        self_state = self.state_dict()
        load_state = torch.load(weight_file)['weights']
        if self_state.keys() != load_state.keys():
            log.warning('Modules not matching, performing partial update')
            update_state = {k: v for k, v in load_state.items() if k in self_state}
            self_state.update(update_state)
            load_state = self_state
        self.load_state_dict(load_state)
        log.info('Loaded weights from {}'.format(weight_file))

    def save_weights(self, weight_file):
        """ This function will save the weights to a file.

        Args:
            weight_file (str): path to file
        """
        state = {
            'weights': self.state_dict()
        }
        torch.save(state, weight_file)
        log.info('Saved weights as {}'.format(weight_file))

    def init_weights(self, mode='fan_in', slope=0):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=slope, mode=mode)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        log.info('Init weights by kaiming_normal_')

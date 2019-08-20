import logging as log
import time
import os
import os.path as osp

import torch.optim as opt

from .models import network
from . import loss
from ._parser_config import parser_config

__all__ = ['Execute']


class Execute(object):
    def __init__(self, config, dataloader):
        super(Execute, self).__init__()
        self.config = config
        self._create(self.config)
        self.dataloader = dataloader

    def _create(self, config):
        self.model = network.__dict__[config.MODEL_CONFIG.NAME](**config.MODEL_CONFIG.INPUT)
        if config.MODEL_CONFIG.CUDA:
            self.model = self.model.cuda()
        log.info('Created a `{}` model.'.format(self.model.__repr__()))

        self.loss = loss.__dict__[config.LOSS_CONFIG.NAME](**config.LOSS_CONFIG.INPUT)
        log.info('Loss function {} have created.'.format(self.loss.__repr__()))

        self._get_optimizer(config)
        log.info('Optimizer {} have created.'.format(self.optimizer.__repr__()))
        # log.info('Scheduler {} have created.'.format(self.scheduler.__repr__()))

    def _get_optimizer(self, config):
        _type = config.TRAIN.OPTIMIZER.TYPE
        if _type in ('SGD', 'sgd'):
            self.optimizer = opt.SGD(self.model.parameters(), lr=float(config.TRAIN.LR),
                    momentum=float(config.TRAIN.OPTIMIZER.MOMENTUM),
                    weight_decay=float(config.TRAIN.OPTIMIZER.WEIGHT_DECAY))
        else:
            raise ValueError

        _type = config.TRAIN.SCHEDULER.TYPE
        if _type in ('consine', 'Consine'):
            self.scheduler = opt.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.config.TRAIN.MAX_EPOCH)
        else:
            raise ValueError

    def set_optimizer(self, optimizer):
        pass

    def set_scheduler(self, scheduler):
        pass

    def adjust_learning_rate(self):
        pass

    def train(self):
        if not osp.isdir(self.config.CHECKPOINT_DIR):
            os.mkdir(self.config.CHECKPOINT_DIR)
        self.model.train()
        start_ = time.time()
        for epoch_idx in range(self.config.TRAIN.MAX_EPOCH):
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            else:
                self.adjust_learning_rate()
            epoch_loss = 0.
            for batch_idx, data in enumerate(self.dataloader):
                images, target = data
                if self.config.MODEL_CONFIG.CUDA:
                    images = images.cuda()

                out = self.model(images)
                loss = self.loss(out, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _count = epoch_idx * len(self.dataloader) + batch_idx
                if _count != 0 and _count % self.config.TRAIN.BACKUP_RATE == 0:
                    self.model.save_weights(osp.join(self.config.CHECKPOINT_DIR, self.config.BACKUP_NAME))
                if _count != 0 and _count % self.config.TRAIN.INNER_RATE == 0:
                    self.model.save_weights(osp.join(self.config.CHECKPOINT_DIR, 'model_{}.weights'.format(_count)))

                epoch_loss += loss.item()
            log.info('{}/{}, LOSS: {}'.format(
                epoch_idx, self.config.TRAIN.MAX_EPOCH, epoch_loss / len(self.dataloader)))
        log.info('Training phase finished, cost time: {}'.format(time.time() - start_))

    def test(self):
        self.model.eval()
        pass

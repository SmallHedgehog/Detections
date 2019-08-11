import torch.nn as nn

from .. import layer as basic_layer


class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()

        self.layer = nn.Sequential(
            basic_layer.Conv2dBNLeakyReLU(   3,   64, 7, 2),
            nn.MaxPool2d(2, 2),

            basic_layer.Conv2dBNLeakyReLU(  64,  192, 3, 1),
            nn.MaxPool2d(2, 2),

            basic_layer.Conv2dBNLeakyReLU( 192,  128, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 128,  256, 3, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  256, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  512, 3, 1),
            nn.MaxPool2d(2, 2),

            basic_layer.Conv2dBNLeakyReLU( 512,  256, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  512, 3, 1),
            basic_layer.Conv2dBNLeakyReLU( 512,  256, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  512, 3, 1),
            basic_layer.Conv2dBNLeakyReLU( 512,  256, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  512, 3, 1),
            basic_layer.Conv2dBNLeakyReLU( 512,  256, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 256,  512, 3, 1),
            basic_layer.Conv2dBNLeakyReLU( 512,  512, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 512, 1024, 3, 1),
            nn.MaxPool2d(2, 2),

            basic_layer.Conv2dBNLeakyReLU(1024,  512, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 512, 1024, 3, 1),
            basic_layer.Conv2dBNLeakyReLU(1024,  512, 1, 1),
            basic_layer.Conv2dBNLeakyReLU( 512, 1024, 3, 1),
            basic_layer.Conv2dBNLeakyReLU(1024, 1024, 3, 1),
            basic_layer.Conv2dBNLeakyReLU(1024, 1024, 3, 2),

            basic_layer.Conv2dBNLeakyReLU(1024, 1024, 3, 1),
            basic_layer.Conv2dBNLeakyReLU(1024, 1024, 3, 1),
        )

    def forward(self, x):
        return self.layer(x)

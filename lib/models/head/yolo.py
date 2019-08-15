import torch.nn as nn

from .. import layer as basic_layer


class Yolo(nn.Module):
    def __init__(self, num_boxes=3, num_classes=20, grid_size=(7, 7)):
        super(Yolo, self).__init__()

        self.layer = basic_layer.Conv2dBNLeakyReLU(1024, 256, 3, 1)

        self.classifier = nn.Sequential(
            nn.Linear(256 * grid_size[0] * grid_size[1], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, grid_size[0] * grid_size[1] * (num_boxes * 5 + num_classes))
        )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)

        return self.classifier(x)

import torch.nn as nn


__all__ = ['Conv2dBNLeakyReLU']


class Conv2dBNLeakyReLU(nn.Module):
    """ Grouping a 2D convolution, a 2D batchnorm and a leaky ReLu.
    They are executed in a sequential manner.
    
    Arg:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): Padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLu; Default 0.1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBNLeakyReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(s / 2) for s in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope

        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        return self.layers(x)

"""This module implements the networks for the Generator and Discriminator of
the proposed GAN for conditional image generation.

The Generator takes as input a random noise vector and the one-hot encoding
for the class condition. It generates a 32x32 image using both residual and 
upsample blocks. 

The Discriminator takes as input images, and their corresponding conditions
(labels), and validates their authenticity. In addition, it returns a list
of features for the input images, obtained from the initial four
convolutional blocks.
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils import spectral_norm


def conv3x3(
        in_channels: int, out_channels: int,
        stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding.

    Args:
        - in_channels (int) : number of input channels
        - out_channels (int) : number of output channels
        - stride (int, optional) : stride of the convolution (Default: 1)
        - padding (int, optional) : padding of the convolution (Default: 1)
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False
    )


class ResBlock(nn.Module):
    """Residual block.

    The residual block consists of the following:
        - 3x3 Convolution with padding
        - 2D Batch Normalization
        - ReLU
        - 3x3 Convolution with padding
        - 2D Batch Normalization
        - ReLU

    Args:
        - nfilters (int) : number of filters for the convolutions
    """

    def __init__(self, nfilters: int) -> None:
        """Initialize the residual block.

        Args:
        - nfilters (int) : number of filters for the convolutions
        """
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(nfilters, nfilters),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            conv3x3(nfilters, nfilters),
            nn.BatchNorm2d(nfilters))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class Generator(nn.Module):
    """Generator class.

    The Generator takes as input a random noise vector and the one-hot encoding
    for the class condition. It generates a 32x32 image using both residual 
    blocks and upsample blocks.

    Args:
    - ngf (int) : number of generator filters
    - z_dim (int) : noise dimension
    - c_dim (int) : condition dimension
    - project_dim (int, optional) : dimension to project the input noise vector 
                                    (Default: 128)
    - nc (int, optional) : number of channels of the output (Default: 3)
    """

    def __init__(
            self, ngf: int, z_dim: int, c_dim: int,
            project_dim: int = 128, nc: int = 3) -> None:
        """Initialize the Generator.

        Args:
            - ngf (int) : number of generator filters
            - z_dim (int) : noise dimension
            - c_dim (int) : condition dimension
            - project_dim (int, optional) : dimension to project the input 
                                            noise vector (Default: 128)
            - nc (int, optional) : number of channels of the output 
                                   (Default: 3)
        """
        super(Generator, self).__init__()
        self.ngf = ngf
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.nc = nc

        self.project_dim = project_dim

        self.define_module()

    def upBlock(self, in_planes: int, out_planes: int) -> nn.Sequential:
        """Upscale the spatial size by a factor of 2."""
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            nn.GLU(1)
        )
        return block

    def _make_layer(self, block: nn.Module,
                    nfilters: int, n_blocks: int) -> nn.Sequential:
        """Create a sequential model of <n_blocks> <blocks>."""
        layers = []
        for _ in range(n_blocks):
            layers.append(block(nfilters))
        return nn.Sequential(*layers)

    def define_module(self) -> None:
        """Define the Generator module."""
        ngf = self.ngf

        in_dim = self.c_dim + self.project_dim

        self.project = nn.Sequential(
            nn.Linear(self.z_dim, self.project_dim * 2, bias=False),
            nn.BatchNorm1d(self.project_dim * 2),
            nn.GLU(1))

        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            nn.GLU(1))

        self.upsample1 = self.upBlock(ngf, ngf // 2)
        self.upsample2 = self.upBlock(ngf // 2, ngf // 4)
        self.upsample3 = self.upBlock(ngf // 4, ngf // 8)

        self.residual = self._make_layer(ResBlock, ngf // 8, 3)

        self.img = nn.Sequential(
            conv3x3(ngf // 8, self.nc),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            - z (Tensor) : random noise vector
            - c (Tensor) : condition (one-hot encoding)
        """
        z_p = self.project(z)
        in_code = torch.cat((z_p, c), 1)  # (project_dim+c_dim) x 1

        h_code = self.fc(in_code)
        h_code = h_code.view(-1, self.ngf, 4, 4)  # ngf x 4 x 4

        out_code = self.upsample1(h_code)  # ngf//2 x 8 x 8
        out_code = self.upsample2(out_code)  # ngf//4 x 16 x 16
        out_code = self.upsample3(out_code)  # ngf//8 x 32 x 32

        out_code = self.residual(out_code)  # ngf//8 x 32 x 32

        img = self.img(out_code)  # 3 x 32 x 32

        return img


class Discriminator(nn.Module):
    """Discriminator class.

    The Discriminator takes as input images, and their corresponding conditions
    (labels), and validates their authenticity. In addition, it returns a list
    of features for the input images, obtained from the initial four
    convolutional blocks.

    Args:
    - ndf (int) : number of discriminator filters
    - c_dim (int) : condition dimension
    - project_dim (int, optional) : dimension to project the features before 
                                    concatenating them with the input condition
                                    (Default: 128)
    - nc (int, optional) : number of channels of the output (Default: 3)
    """

    def __init__(
            self, ndf: int, c_dim: int,
            project_dim: int = 128, nc: int = 3) -> None:
        """Initialize the Discriminator.

        Args:
            - ndf (int) : number of discriminator filters
            - c_dim (int) : condition dimension
            - project_dim (int, optional) : dimension to project the features 
                                            before concatenating them with the 
                                            input condition (Default: 128)
            - nc (int, optional) : number of channels of the output 
                                   (Default: 3)
        """
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.c_dim = c_dim
        self.nc = nc

        self.project_dim = project_dim

        self.define_module()

    def conv4x4(self, in_channels: int,
                out_channels: int, bias: bool = True) -> nn.Conv2d:
        """4x4 convolution with stride=2 and padding=1.

        Args:
        - in_channels (int) : number of input channels
        - out_channels (int) : number of output channels
        - bias (bool, optional) : defines the use of a weight bias 
                                  (Default: True)
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,
                         padding=1, bias=bias)

    def define_module(self) -> None:
        """Define the Discriminator module."""
        self.netD = nn.ModuleList([
            nn.Sequential(
                # input is nc x 32 x 32
                # ndf x 16 x 16
                spectral_norm(self.conv4x4(self.nc, self.ndf)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            nn.Sequential(
                spectral_norm(self.conv4x4(self.ndf, self.ndf)),  # ndf x 8 x 8
                nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            nn.Sequential(
                # 2ndf x 4 x 4
                spectral_norm(self.conv4x4(self.ndf, self.ndf * 2)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            nn.Sequential(
                # 4ndf x 2 x 2
                spectral_norm(self.conv4x4(self.ndf * 2, self.ndf * 4)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ])

        self.flatten = nn.Flatten()

        self.project = nn.Sequential(
            spectral_norm(
                nn.Linear(self.ndf*4*2*2, self.project_dim * 2, bias=False)),
            nn.GLU(1)
        )

        self.output = nn.Linear(self.project_dim + self.c_dim, 1, bias=False)

    def forward(self, x: torch.Tensor,
                c: torch.Tensor) -> "Tuple[torch.Tensor, list]":
        """Forward propagation.

        Args:
            - x (Tensor) : input image
            - c (Tensor) : condition (one-hot encoding)
        """
        features = []
        for block in self.netD:
            x = block(x)
            features.append(x)

        c = c.view(-1, self.c_dim)
        x = self.flatten(x)
        h = self.project(x)

        h = torch.cat((h, c), 1)

        return self.output(h), features

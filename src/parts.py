""" Parts of the U-Net model
    -- modified from https://github.com/milesial/Pytorch-UNet """

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, dilation=1, sf=None):
        super().__init__()
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, bias=False, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, sf=1):
        super().__init__()
        self._scaling_factor = sf
        
        padding = kernel_size // 2
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size,
                padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size,
                padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels:
            self.projection_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.double_conv(x)

        if hasattr(self, 'projection_conv'):
            x = self.projection_conv(x)
            
        out = out * self._scaling_factor + x

        return F.relu(out)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            block(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, block=ResBlock, bilinear=True,
                 **kwargs):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2,
                              **kwargs)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = block(in_channels, out_channels,
                              **kwargs)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             bias=False)

    def forward(self, x):
        return self.out(x)

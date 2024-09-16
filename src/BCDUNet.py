import torch
import torch.nn as nn
import numpy as np
from src.ConvLSTM import ConvBLSTM, ConvLSTM

class BCDUNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_filter=64, frame_size=(256, 256), bidirectional=False, norm='instance'):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.frame_size = np.array(frame_size)

        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            )

        self.conv1 = conv_block(input_dim, num_filter)
        self.conv2 = conv_block(num_filter, num_filter * 2)
        self.conv3 = conv_block(num_filter * 2, num_filter * 4)
        self.conv4 = conv_block(num_filter * 4, num_filter * 8)

        self.upconv3 = nn.ConvTranspose2d(num_filter * 8, num_filter * 4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(num_filter * 4, num_filter * 2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(num_filter * 2, num_filter, kernel_size=2, stride=2)

        self.conv3m = conv_block(num_filter * 8, num_filter * 4)
        self.conv2m = conv_block(num_filter * 4, num_filter * 2)
        self.conv1m = conv_block(num_filter * 2, num_filter)

        self.conv0 = nn.Conv2d(num_filter, output_dim, kernel_size=1)

        if bidirectional:
            self.clstm1 = ConvBLSTM(num_filter*8, num_filter*8, (3, 3), (1,1), 'tanh', list(self.frame_size//8))
            self.clstm2 = ConvBLSTM(num_filter*4, num_filter*4, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm3 = ConvBLSTM(num_filter*2, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
        else:
            self.clstm1 = ConvLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))

    def forward(self, x):
        N = self.frame_size
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.maxpool(conv3)
        conv4 = self.conv4(pool3)

        upconv3 = self.upconv3(conv4)
        concat3 = torch.cat((conv3, upconv3), 1)
        concat3 = self.clstm3(concat3)
        conv3m = self.conv3m(concat3)

        upconv2 = self.upconv2(conv3m)
        concat2 = torch.cat((conv2, upconv2), 1)
        concat2 = self.clstm2(concat2)
        conv2m = self.conv2m(concat2)

        upconv1 = self.upconv1(conv2m)
        concat1 = torch.cat((conv1, upconv1), 1)
        concat1 = self.clstm1(concat1)
        conv1m = self.conv1m(concat1)

        conv0 = self.conv0(conv1m)

        return conv0

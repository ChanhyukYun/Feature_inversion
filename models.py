import torch
import torch.nn as nn
    
class invNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(invNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # Convolution blocks
        self.conv_blk1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_blk2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_blk3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_blk4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upconvolution
        self.upconv_blk1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.upconv_blk2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upconv_blk3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv_blk4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_blk6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_blk7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_blk8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_blk9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_blk10 = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        dx1 = self.conv_blk1(x) # x: 128, dx1: 64
        dx2 = self.conv_blk2(self.maxpool(dx1)) # dx1: 64, dx2: 128
        dx3 = self.conv_blk3(self.maxpool(dx2)) # dx2:128, dx3: 256
        dx4 = self.conv_blk4(self.maxpool(dx3)) # dx3:256, dx4: 512
        dx5 = self.conv_blk5(self.maxpool(dx4)) # dx4:512, dx5:1024

        x = self.upconv_blk1(dx5) # 1024 -> 512
        x = self.upconv_blk2(self.conv_blk6(torch.cat([dx4, x], dim=1))) # 1024 -> 512 -> 256
        x = self.upconv_blk3(self.conv_blk7(torch.cat([dx3, x], dim=1))) # 512 -> 256 -> 128
        x = self.upconv_blk4(self.conv_blk8(torch.cat([dx2, x], dim=1))) # 256 -> 128 -> 64
        x = self.conv_blk9(torch.cat([dx1, x], dim=1))
        x = self.conv_blk10(x)
        return (x+1.0)*0.5
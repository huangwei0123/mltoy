import torch
import torch.nn as nn
import torch.nn.functional as F

from models.double_conv import DoubleConv


class ForecastUNet(nn.Module):

    def __init__(
        self,
        input_channels,
        forecast_steps
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            input_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            128,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.out_conv = nn.Conv2d(
            32,
            forecast_steps,
            kernel_size=1
        )

    def match_size(
        self,
        x,
        ref
    ):

        if x.shape[-2:] != ref.shape[-2:]:

            x = F.interpolate(
                x,
                size=ref.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return x

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)
        d3 = self.match_size(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.match_size(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.match_size(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

import torch
import torch.nn as nn
import torch.fft


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes_lat=8,
        modes_lon=8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_lat = modes_lat
        self.modes_lon = modes_lon

        scale = 1 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                modes_lat,
                modes_lon,
                dtype=torch.cfloat,
            )
        )

    def compl_mul2d(
        self,
        input,
        weights,
    ):
        return torch.einsum(
            "bixy,ioxy->boxy",
            input,
            weights,
        )

    def forward(self, x):

        batchsize = x.shape[0]

        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[
            :,
            :,
            : self.modes_lat,
            : self.modes_lon,
        ] = self.compl_mul2d(
            x_ft[
                :,
                :,
                : self.modes_lat,
                : self.modes_lon,
            ],
            self.weights,
        )

        x = torch.fft.irfft2(
            out_ft,
            s=(x.size(-2), x.size(-1)),
        )

        return x


class FNOBlock(nn.Module):
    def __init__(
        self,
        width,
        modes_lat,
        modes_lon,
    ):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes_lat,
            modes_lon,
        )

        self.conv = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.act = nn.GELU()

    def forward(self, x):

        y1 = self.spectral(x)
        y2 = self.conv(x)

        return self.act(y1 + y2)


class ForecastFNO(nn.Module):
    def __init__(
        self,
        input_channels,
        forecast_steps,
        width=32,
        modes_lat=8,
        modes_lon=8,
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(
            input_channels,
            width,
            kernel_size=1,
        )

        self.block1 = FNOBlock(
            width,
            modes_lat,
            modes_lon,
        )

        self.block2 = FNOBlock(
            width,
            modes_lat,
            modes_lon,
        )

        self.block3 = FNOBlock(
            width,
            modes_lat,
            modes_lon,
        )

        self.output_proj = nn.Conv2d(
            width,
            forecast_steps,
            kernel_size=1,
        )

    def forward(self, x):

        x = self.input_proj(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.output_proj(x)

        return x


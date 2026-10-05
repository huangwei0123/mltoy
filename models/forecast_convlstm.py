import torch
import torch.nn as nn

from utils.config import FORECAST


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x,
        h_prev,
        c_prev,
    ):
        """
        x:
            (B, C, H, W)

        h_prev:
            (B, hidden_dim, H, W)

        c_prev:
            (B, hidden_dim, H, W)
        """

        combined = torch.cat(
            [x, h_prev],
            dim=1,
        )

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1,
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = (
            f * c_prev
            + i * g
        )

        h = (
            o
            * torch.tanh(c)
        )

        return h, c


class ForecastConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels=6,
        hidden_dim=64,
        forecast_steps=FORECAST,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        self.cell = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                forecast_steps,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x,
    ):
        """
        x:

        (B, T, C, H, W)

        Example:

        B = batch size
        T = HISTORY
        C = 6
        H = 36
        W = 72
        """

        B, T, C, H, W = x.shape

        if C != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels "
                f"but received {C}"
            )

        h = x.new_zeros(
            B,
            self.hidden_dim,
            H,
            W,
        )

        c = x.new_zeros(
            B,
            self.hidden_dim,
            H,
            W,
        )

        for t in range(T):
            h, c = self.cell(
                x[:, t],
                h,
                c,
            )

        y = self.decoder(h)

        #
        # (B, forecast_steps, H, W)
        #
        y = y.unsqueeze(2)

        #
        # (B, forecast_steps, 1, H, W)
        #
        return y

# models/forecast_convlstm.py

import torch
import torch.nn as nn

from utils.config import (
    HISTORY,
    FORECAST,
)

INPUT_CHANNELS = HISTORY + 5

class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x,
        h_prev,
        c_prev,
    ):

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

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ForecastConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels=INPUT_CHANNELS,
        hidden_dim=64,
        forecast_steps=FORECAST,
    ):
        super().__init__()

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
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                forecast_steps,
                kernel_size=1,
            ),
        )

    def forward(self, x):

        #
        # x
        # (B,T,C,H,W)
        #

        B, T, C, H, W = x.shape

        device = x.device

        h = torch.zeros(
            B,
            self.hidden_dim,
            H,
            W,
            device=device,
        )

        c = torch.zeros(
            B,
            self.hidden_dim,
            H,
            W,
            device=device,
        )

        for t in range(T):
            h, c = self.cell(
                x[:, t],
                h,
                c,
            )

        y = self.decoder(h)

        #
        # (B,forecast,H,W)
        #
        y = y.unsqueeze(2)

        #
        # (B,forecast,1,H,W)
        #
        return y

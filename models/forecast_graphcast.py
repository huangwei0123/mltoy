import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBlock(nn.Module):
    """
    Simple GraphCast-style message passing block.

    Approximates neighborhood communication using
    local convolutions instead of expensive
    all-to-all attention.
    """

    def __init__(
        self,
        hidden_dim
    ):
        super().__init__()

        self.message = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            )
        )

        self.norm = nn.BatchNorm2d(
            hidden_dim
        )

    def forward(
        self,
        x
    ):
        residual = x

        msg = self.message(x)

        x = residual + msg

        x = self.norm(x)

        x = F.gelu(x)

        return x


class ForecastGraphCast(nn.Module):
    """
    GraphCast-Lite.

    Input
    -----
    [B, C, LAT, LON]

    Output
    ------
    [B, FORECAST, LAT, LON]
    """

    def __init__(
        self,
        input_channels,
        forecast_steps,
        hidden_dim=64,
        num_layers=6
    ):
        super().__init__()

        self.input_channels = (
            input_channels
        )

        self.forecast_steps = (
            forecast_steps
        )

        ####################################################
        # Input encoder
        ####################################################

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_dim,
                kernel_size=1
            ),
            nn.GELU()
        )

        ####################################################
        # Message passing blocks
        ####################################################

        self.graph_blocks = nn.ModuleList(
            [
                GraphBlock(
                    hidden_dim
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        ####################################################
        # Forecast head
        ####################################################

        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                forecast_steps,
                kernel_size=1
            )
        )

    def forward(
        self,
        x
    ):
        """
        Input:
            [B,C,LAT,LON]

        Output:
            [B,F,LAT,LON]
        """

        x = self.encoder(x)

        for block in self.graph_blocks:
            x = block(x)

        x = self.decoder(x)

        return x


if __name__ == "__main__":

    model = ForecastGraphCast(
        input_channels=13,
        forecast_steps=12,
        hidden_dim=64,
        num_layers=6
    )

    x = torch.randn(
        2,
        13,
        36,
        72
    )

    y = model(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "Output:",
        y.shape
    )

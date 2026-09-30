import torch.nn as nn


class ForecastCNN(nn.Module):

    def __init__(
        self,
        input_channels,
        forecast_steps
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                forecast_steps,
                kernel_size=1
            )
        )

    def forward(
        self,
        x
    ):
        return self.net(x)

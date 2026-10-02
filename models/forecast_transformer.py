import torch
import torch.nn as nn


class ForecastTransformer(nn.Module):
    """
    Transformer-based weather forecaster.

    Input:
        [B, C, LAT, LON]

    Output:
        [B, FORECAST, LAT, LON]

    where

        C = HISTORY + extra features

    Example:

        HISTORY = 8

        Inputs:
            temperature history
            sin_hour
            cos_hour
            sin_doy
            cos_doy
            latitude

        input_channels = 13
    """

    def __init__(
        self,
        input_channels,
        forecast_steps,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.forecast_steps = forecast_steps

        ####################################################
        # Input embedding
        ####################################################

        self.input_proj = nn.Linear(
            input_channels,
            d_model,
        )

        ####################################################
        # Transformer encoder
        ####################################################

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        ####################################################
        # Forecast head
        ####################################################

        self.output_proj = nn.Linear(
            d_model,
            forecast_steps,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Shape:
                [B, C, LAT, LON]

        Returns
        -------
        tensor
            Shape:
                [B, FORECAST, LAT, LON]
        """

        b, c, lat, lon = x.shape

        ####################################################
        # Convert grid to token sequence
        #
        # [B,C,LAT,LON]
        #   ->
        # [B,LAT,LON,C]
        #   ->
        # [B,LAT*LON,C]
        ####################################################

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = x.reshape(
            b,
            lat * lon,
            c,
        )

        ####################################################
        # Embed
        ####################################################

        x = self.input_proj(x)

        ####################################################
        # Transformer
        ####################################################

        x = self.encoder(x)

        ####################################################
        # Predict all forecast leads
        #
        # [B,TOKENS,d_model]
        #   ->
        # [B,TOKENS,FORECAST]
        ####################################################

        x = self.output_proj(x)

        ####################################################
        # Restore weather grid
        #
        # [B,TOKENS,F]
        #   ->
        # [B,LAT,LON,F]
        #   ->
        # [B,F,LAT,LON]
        ####################################################

        x = x.reshape(
            b,
            lat,
            lon,
            self.forecast_steps,
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        return x


if __name__ == "__main__":

    model = ForecastTransformer(
        input_channels=13,
        forecast_steps=12,
    )

    x = torch.randn(
        2,
        13,
        36,
        72,
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

from models.forecast_cnn import ForecastCNN
from models.forecast_fno import ForecastFNO
from models.forecast_unet import ForecastUNet
from models.forecast_convlstm import ForecastConvLSTM
from models.forecast_graphcast import ForecastGraphCast
from models.forecast_transformer import ForecastTransformer


def create_model(
    model_type,
    input_channels,
    forecast_steps,
    **kwargs
):

    model_type = model_type.lower()

    ########################################################
    # U-Net
    ########################################################

    if model_type == "unet":

        return ForecastUNet(
            input_channels=input_channels,
            forecast_steps=forecast_steps
        )

    ########################################################
    # CNN
    ########################################################

    elif model_type == "cnn":

        return ForecastCNN(
            input_channels=input_channels,
            forecast_steps=forecast_steps
        )

    ########################################################
    # ConvLSTM
    ########################################################

    elif model_type == "convlstm":

        return ForecastConvLSTM(
            input_channels=kwargs.get(
                "input_channels_per_timestep",
                6
            ),
            hidden_dim=kwargs.get(
                "hidden_dim",
                64
            ),
            forecast_steps=forecast_steps
        )

    ########################################################
    # FNO
    ########################################################

    elif model_type == "fno":

        return ForecastFNO(
            input_channels=input_channels,
            forecast_steps=forecast_steps,
            width=kwargs.get(
                "width",
                64
            ),
            modes_lat=kwargs.get(
                "modes_lat",
                12
            ),
            modes_lon=kwargs.get(
                "modes_lon",
                12
            )
        )

    ########################################################
    # GraphCast
    ########################################################

    elif model_type == "graphcast":

        return ForecastGraphCast(
            input_channels=input_channels,
            forecast_steps=forecast_steps,
            hidden_dim=kwargs.get(
                "hidden_dim",
                32
            ),
            num_layers=kwargs.get(
                "num_layers",
                4
            )
        )

    ########################################################
    # Transformer
    ########################################################

    elif model_type == "transformer":

        return ForecastTransformer(
            input_channels=input_channels,
            forecast_steps=forecast_steps,
            d_model=kwargs.get(
                "d_model",
                64
            ),
            nhead=kwargs.get(
                "nhead",
                4
            ),
            num_layers=kwargs.get(
                "num_layers",
                2
            ),
            dropout=kwargs.get(
                "dropout",
                0.1
            )
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )

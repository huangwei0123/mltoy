from models.forecast_transformer import (
    ForecastTransformer
)

model = ForecastTransformer(
    input_channels=13,
    forecast_steps=12
)

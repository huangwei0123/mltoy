from models.forecast_fno import ForecastFNO

model = ForecastFNO(
    input_channels=13,
    forecast_steps=12
)

trainer.fit(
    model,
    train_loader,
    val_loader
)

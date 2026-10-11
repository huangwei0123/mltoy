import torch

from forecasts.model_factory import create_model


class Predictor:

    def __init__(
        self,
        checkpoint_file,
        device="cpu"
    ):

        self.device = device

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        self.history = checkpoint["history"]
        self.forecast = checkpoint["forecast"]
        self.model_type = checkpoint.get("model_type", "unet")

        hidden_dim = checkpoint.get("hidden_dim")

        if hidden_dim is None and self.model_type == "convlstm":
            hidden_dim = (checkpoint["model_state_dict"] ["cell.conv.weight"] .shape[0] // 4)

        print("history =", self.history)
        print("forecast =", self.forecast)
        print("model_type =", self.model_type)

        for k, v in checkpoint["model_state_dict"].items():
            if "cell.conv.weight" in k:
                print("checkpoint conv:", v.shape)

        self.model = create_model(
            self.model_type,
            input_channels=self.history + 7,
            forecast_steps=self.forecast,

            width=checkpoint.get("width", 64),
            modes_lat=checkpoint.get("modes_lat", 12),
            modes_lon=checkpoint.get("modes_lon", 12),

            hidden_dim=hidden_dim,

            d_model=checkpoint.get("d_model", 64),
            nhead=checkpoint.get("nhead", 4),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=checkpoint.get("dropout", 0.1)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def predict(
        self,
        x
    ):

        with torch.no_grad():

            x = x.to(self.device)

            y = self.model(x)

        return y.cpu().numpy()

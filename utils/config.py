import yaml


class Config:

    def __init__(self, filename):

        with open(filename, "r") as f:
            cfg = yaml.safe_load(f)

        self.dataset = cfg["dataset"]

        self.model = cfg["model"]

        self.training = cfg["training"]

        self.checkpoint = cfg.get(
            "checkpoint",
            {}
        )

        self.output = cfg["output"]

import time
import torch


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        logger
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

    def train_epoch(
        self,
        train_loader
    ):

        self.model.train()

        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(x)

            loss = self.criterion(
                pred,
                y
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return (
            total_loss /
            len(train_loader)
        )

    def validate(
        self,
        val_loader
    ):

        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                loss = self.criterion(
                    pred,
                    y
                )

                total_loss += loss.item()

        return (
            total_loss /
            len(val_loader)
        )

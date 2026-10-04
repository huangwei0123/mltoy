import numpy as np
import torch

from torch.utils.data import Dataset


class ConvLSTMDataset(Dataset):

    def __init__(
        self,
        temperature,
        sin_hour,
        cos_hour,
        sin_doy,
        cos_doy,
        lat_grid,
        history,
        forecast
    ):
        """
        temperature:
            (time, lat, lon)

        time features:
            (time,)

        lat_grid:
            (lat, lon)
        """

        self.temperature = temperature

        self.sin_hour = sin_hour
        self.cos_hour = cos_hour

        self.sin_doy = sin_doy
        self.cos_doy = cos_doy

        self.lat_grid = lat_grid

        self.history = history
        self.forecast = forecast

        self.length = (
            len(temperature)
            - history
            - forecast
            + 1
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        ####################################################
        # History temperatures
        ####################################################

        history = self.temperature[
            idx : idx + self.history
        ]

        ####################################################
        # Future targets
        ####################################################

        target = self.temperature[
            idx + self.history :
            idx + self.history + self.forecast
        ]

        ####################################################
        # Grid size
        ####################################################

        lat_size = history.shape[1]
        lon_size = history.shape[2]

        ####################################################
        # Build ConvLSTM input
        #
        # Output:
        # (history, 6, lat, lon)
        ####################################################

        frames = []

        for t in range(self.history):

            current_index = idx + t

            ################################################
            # Temperature
            ################################################

            temp_channel = history[t]

            ################################################
            # Temporal features
            ################################################

            sin_h = np.full(
                (lat_size, lon_size),
                self.sin_hour[current_index],
                dtype=np.float32
            )

            cos_h = np.full(
                (lat_size, lon_size),
                self.cos_hour[current_index],
                dtype=np.float32
            )

            sin_d = np.full(
                (lat_size, lon_size),
                self.sin_doy[current_index],
                dtype=np.float32
            )

            cos_d = np.full(
                (lat_size, lon_size),
                self.cos_doy[current_index],
                dtype=np.float32
            )

            ################################################
            # Frame
            ################################################

            frame = np.stack(
                [
                    temp_channel,
                    sin_h,
                    cos_h,
                    sin_d,
                    cos_d,
                    self.lat_grid
                ],
                axis=0
            )

            frames.append(frame)

        ####################################################
        # x
        ####################################################

        x = np.stack(
            frames,
            axis=0
        )

        ####################################################
        # y
        #
        # (forecast, 1, lat, lon)
        ####################################################

        y = target[:, np.newaxis, :, :]

        ####################################################
        # Torch tensors
        ####################################################

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y

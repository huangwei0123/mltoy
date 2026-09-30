import numpy as np
import torch

from torch.utils.data import Dataset


class ForecastDataset(Dataset):

    def __init__(
        self,
        temp,
        sin_hour,
        cos_hour,
        sin_doy,
        cos_doy,
        lat_grid,
        history,
        forecast
    ):

        self.temp = temp

        self.sin_hour = sin_hour
        self.cos_hour = cos_hour

        self.sin_doy = sin_doy
        self.cos_doy = cos_doy

        self.lat_grid = lat_grid

        self.history = history
        self.forecast = forecast

        self.nlat = temp.shape[1]
        self.nlon = temp.shape[2]

    def __len__(self):

        return (
            len(self.temp)
            - self.history
            - self.forecast
        )

    def __getitem__(self, idx):

        temp_stack = self.temp[
            idx:idx + self.history
        ]

        current = idx + self.history - 1

        sh = np.full(
            (1, self.nlat, self.nlon),
            self.sin_hour[current],
            dtype=np.float32
        )

        ch = np.full(
            (1, self.nlat, self.nlon),
            self.cos_hour[current],
            dtype=np.float32
        )

        sd = np.full(
            (1, self.nlat, self.nlon),
            self.sin_doy[current],
            dtype=np.float32
        )

        cd = np.full(
            (1, self.nlat, self.nlon),
            self.cos_doy[current],
            dtype=np.float32
        )

        lat = self.lat_grid[np.newaxis, :, :]

        x = np.concatenate(
            [
                temp_stack,
                sh,
                ch,
                sd,
                cd,
                lat
            ],
            axis=0
        )

        y = self.temp[
            current + 1:
            current + 1 + self.forecast
        ]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

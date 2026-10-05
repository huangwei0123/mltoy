# datasets/convlstm_dataset.py

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
        forecast,
    ):
        """
        Parameters
        ----------
        temperature
            (time, lat, lon)

        sin_hour, cos_hour
            (time,)

        sin_doy, cos_doy
            (time,)

        lat_grid
            (lat, lon)

        history
            Number of input timesteps

        forecast
            Number of forecast timesteps
        """

        self.history = history
        self.forecast = forecast

        #
        # Ensure float32
        #
        self.temperature = np.asarray(
            temperature,
            dtype=np.float32
        )

        nt = self.temperature.shape[0]
        nlat = self.temperature.shape[1]
        nlon = self.temperature.shape[2]

        #
        # Precompute feature grids
        #
        # features:
        # (time, 5, lat, lon)
        #
        # channel 0 = sin_hour
        # channel 1 = cos_hour
        # channel 2 = sin_doy
        # channel 3 = cos_doy
        # channel 4 = latitude
        #
        self.features = np.empty(
            (nt, 5, nlat, nlon),
            dtype=np.float32
        )

        self.features[:, 0] = (
            np.asarray(
                sin_hour,
                dtype=np.float32
            )[:, None, None]
        )

        self.features[:, 1] = (
            np.asarray(
                cos_hour,
                dtype=np.float32
            )[:, None, None]
        )

        self.features[:, 2] = (
            np.asarray(
                sin_doy,
                dtype=np.float32
            )[:, None, None]
        )

        self.features[:, 3] = (
            np.asarray(
                cos_doy,
                dtype=np.float32
            )[:, None, None]
        )

        self.features[:, 4] = (
            np.asarray(
                lat_grid,
                dtype=np.float32
            )[None, :, :]
        )

        self.length = (
            nt
            - history
            - forecast
            + 1
        )

        if self.length <= 0:
            raise ValueError(
                f"Dataset length becomes "
                f"{self.length}. "
                f"history={history}, "
                f"forecast={forecast}, "
                f"nt={nt}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        ####################################################
        # Input history temperatures
        ####################################################

        temp_hist = self.temperature[
            idx : idx + self.history
        ]

        #
        # (history, 1, lat, lon)
        #
        temp_hist = temp_hist[:, None, :, :]

        ####################################################
        # Time/static features
        ####################################################

        feat_hist = self.features[
            idx : idx + self.history
        ]

        #
        # x:
        # (history, 6, lat, lon)
        #
        x = np.concatenate(
            (
                temp_hist,
                feat_hist,
            ),
            axis=1
        )

        ####################################################
        # Forecast targets
        ####################################################

        y = self.temperature[
            idx + self.history :
            idx + self.history + self.forecast
        ]

        #
        # (forecast, 1, lat, lon)
        #
        y = y[:, None, :, :]

        ####################################################
        # Torch tensors
        ####################################################

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y

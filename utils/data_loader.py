import numpy as np
import xarray as xr


class ERA5DataLoader:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

    def load(self):

        ds = xr.open_zarr(
            self.dataset_path
        )

        temp = ds["t2m"].values.astype(
            np.float32
        )

        times = ds["valid_time"].to_index()

        return (
            ds,
            temp,
            times
        )

    @staticmethod
    def split(temp):

        n = len(temp)

        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train = temp[:train_end]
        val = temp[train_end:val_end]
        test = temp[val_end:]

        return (
            train,
            val,
            test,
            train_end,
            val_end
        )

    @staticmethod
    def normalize(
        train,
        val,
        test
    ):

        mean = train.mean()
        std = train.std()

        train = (train - mean) / std
        val = (val - mean) / std
        test = (test - mean) / std

        return (
            train,
            val,
            test,
            mean,
            std
        )

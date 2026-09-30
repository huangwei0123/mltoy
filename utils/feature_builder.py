import numpy as np


class FeatureBuilder:

    @staticmethod
    def build_time_features(times):

        hours = times.hour.values

        sin_hour = np.sin(
            2 * np.pi * hours / 24.0
        ).astype(np.float32)

        cos_hour = np.cos(
            2 * np.pi * hours / 24.0
        ).astype(np.float32)

        doy = times.dayofyear.values

        sin_doy = np.sin(
            2 * np.pi * doy / 365.25
        ).astype(np.float32)

        cos_doy = np.cos(
            2 * np.pi * doy / 365.25
        ).astype(np.float32)

        return (
            sin_hour,
            cos_hour,
            sin_doy,
            cos_doy
        )

    @staticmethod
    def build_latitude_feature(
        latitudes,
        nlon
    ):

        lat_norm = (
            latitudes / 90.0
        ).astype(np.float32)

        lat_grid = np.repeat(
            lat_norm[:, None],
            nlon,
            axis=1
        )

        return lat_grid

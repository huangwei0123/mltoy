import numpy as np

from datetime import timedelta


class ClimatologyBuilder:

    @staticmethod
    def compute(
        temp,
        times
    ):
        doy = np.array(
            [t.dayofyear for t in times]
        )

        cycle = np.array(
            [t.hour // 6 for t in times]
        )

        climatology = {}

        for d in np.unique(doy):
            for c in range(4):

                idx = (
                    (doy == d)
                    &
                    (cycle == c)
                )

                climatology[(d, c)] = (
                    temp[idx].mean(axis=0)
                )

        return climatology

    @staticmethod
    def anomaly(
        temp,
        times,
        climatology
    ):
        result = np.empty_like(temp)

        for i, t in enumerate(times):

            d = t.dayofyear
            c = t.hour // 6

            result[i] = (
                temp[i]
                -
                climatology[(d, c)]
            )

        return result

class ClimatologyForecastHelper:

    @staticmethod
    def future_climatology(
        climatology,
        start_time,
        forecast_steps,
        lat_size,
        lon_size
    ):
        result = []

        for lead in range(forecast_steps):

            valid_time = (
                start_time
                + timedelta(hours=6 * (lead + 1))
            )

            doy = valid_time.dayofyear
            cycle = valid_time.hour // 6

            result.append(
                climatology[(doy, cycle)]
            )

        return result

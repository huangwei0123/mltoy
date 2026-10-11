import numpy as np


def rmse(pred, truth):

    return np.sqrt(np.mean((pred - truth) ** 2))


def mae(pred, truth):

    return np.mean(np.abs(pred - truth))


def bias(pred, truth):

    return np.mean(pred - truth)


def acc(pred, truth, climatology):

    pred_anom = (pred - climatology)

    truth_anom = (truth - climatology)

    numerator = np.sum(pred_anom * truth_anom)

    denominator = np.sqrt(np.sum(pred_anom ** 2) * np.sum(truth_anom ** 2))

    if denominator == 0:
        return np.nan

    return (numerator / denominator)

def get_climatology_field(climatology, timestamp):

    doy = timestamp.dayofyear

    cycle = (timestamp.hour // 6)

    return climatology[ (doy, cycle) ]

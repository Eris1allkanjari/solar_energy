import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.configs.evaluation import MAPE_PRODUCTION_THRESHOLD


def evaluate(
    y_true,
    y_pred,
    production_threshold=MAPE_PRODUCTION_THRESHOLD
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    # percentage metrics only for meaningful pv production
    mask = y_true > production_threshold

    if np.any(mask):
        mape = np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask]) / y_true[mask]
            )
        ) * 100

    else:
        mape = np.nan

    return mae, rmse, mape


def as_capacity_percentage(value, capacity_kwh, minimum_kwh=0.0):
    """Express a kWh error as a percentage of the training production range.

    This is the min-max scaled error. Scaling both series before subtracting,

        (pred - min) / (max - min) - (real - min) / (max - min)

    cancels the offset and leaves (pred - real) / (max - min), so the scaled MAE
    is the plain MAE over the same denominator. With minimum_kwh = 0 it reduces
    to max scaling, MAE / max, which is the NMAE-by-capacity convention.

    For this target the training minimum is exactly zero, since roughly half of
    all hours are dark, so the two forms coincide. The parameter is kept so the
    general case is what the code actually implements.
    """
    if capacity_kwh is None:
        return np.nan

    scale = capacity_kwh - minimum_kwh

    if scale <= 0:
        return np.nan

    return float(100 * np.asarray(value, dtype=float) / scale)


def clock_hour_mask(index, hours):
    """Boolean mask selecting timestamps whose hour is in `hours`.

    Reported periods are defined by clock hour rather than solar elevation. The
    boundaries are fixed, need no site geometry to interpret, and make any
    reported figure reproducible from the timestamp alone.
    """
    return np.isin(pd.DatetimeIndex(index).hour, np.asarray(hours))


def split_by_mask(y_true, y_pred, mask, metric):
    """Apply a metric inside a mask and inside its complement.

    Returns (inside, outside), with np.nan where a side has no hours, so that
    every period-split metric reports the complement rather than quietly
    dropping the hours it excluded.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.asarray(mask, dtype=bool)

    inside = metric(y_true[mask], y_pred[mask]) if mask.any() else np.nan
    outside = metric(y_true[~mask], y_pred[~mask]) if (~mask).any() else np.nan

    return inside, outside


def clock_period_mae(index, y_true, y_pred, hours):
    """MAE inside the given hours and over the complementary hours.

    Returns (inside, outside). Pooling the two understates the error that
    matters: the complement of the daylight hours is mostly dark, every model
    scores near zero there because predicting no production is trivial, and
    including it roughly halves the reported figure.
    """
    return split_by_mask(
        y_true,
        y_pred,
        clock_hour_mask(index, hours),
        lambda true, pred: float(np.abs(true - pred).mean())
    )


def clock_period_mape(
    index,
    y_true,
    y_pred,
    hours,
    production_threshold=MAPE_PRODUCTION_THRESHOLD
):
    """MAPE restricted to the given hours.

    Delegates to evaluate() so the production threshold behaves exactly as it
    does for every other percentage figure in the project. Percentage error is
    only reported over the ramp and midday periods; across a whole day or a
    whole year the denominator spends too long near zero for the mean to mean
    anything.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = clock_hour_mask(index, hours)

    if not mask.any():
        return np.nan

    _, _, mape = evaluate(
        y_true[mask],
        y_pred[mask],
        production_threshold=production_threshold
    )

    return mape

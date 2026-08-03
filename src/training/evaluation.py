import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.configs.evaluation import (
    DAYLIGHT_ELEVATION_DEGREES,
    MAPE_PRODUCTION_THRESHOLD,
    SITE_LATITUDE,
    SITE_LONGITUDE
)
from src.utils.solar import is_daylight


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


def daylight_mask_for(
    index,
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    elevation_threshold=DAYLIGHT_ELEVATION_DEGREES
):
    return is_daylight(
        index=index,
        latitude=latitude,
        longitude=longitude,
        elevation_threshold=elevation_threshold
    )


def daylight_mape(
    index,
    y_true,
    y_pred,
    production_threshold=MAPE_PRODUCTION_THRESHOLD,
    **daylight_kwargs
):
    """MAPE restricted to hours when the sun is above the horizon.

    The headline MAPE selects hours by a production threshold, which is only a
    proxy for daylight. Recomputing it on an explicit solar-elevation mask tests
    whether that proxy is sound: if the two agree, the threshold is doing its
    job. MAPE has no night counterpart because production is zero after sunset
    and the percentage denominator vanishes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = daylight_mask_for(index=index, **daylight_kwargs)

    if not mask.any():
        return np.nan

    _, _, mape = evaluate(
        y_true[mask],
        y_pred[mask],
        production_threshold=production_threshold
    )

    return mape

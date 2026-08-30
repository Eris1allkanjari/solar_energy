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


def as_capacity_percentage(value, capacity_kwh):
    """Express a kWh error as a percentage of peak capacity.

    Dividing by a fixed reference peak turns an absolute error into the scaled
    (normalised) form used to compare forecasts across sites and datasets, where
    a raw kWh figure means nothing without knowing the size of the installation.
    """
    if capacity_kwh is None or capacity_kwh <= 0:
        return np.nan

    return float(100 * np.asarray(value, dtype=float) / capacity_kwh)


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


def daylight_mae(
    index,
    y_true,
    y_pred,
    **daylight_kwargs
):
    """MAE over daylight hours only, with the night MAE alongside it.

    Roughly half of all hours are dark, and every model scores near zero on them
    because predicting no production is trivial. Pooling those hours therefore
    halves the reported MAE and understates the error on the hours that actually
    carry forecasting difficulty. Returns (daylight_mae, night_mae).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = daylight_mask_for(index=index, **daylight_kwargs)

    day = (
        float(np.abs(y_true[mask] - y_pred[mask]).mean())
        if mask.any()
        else np.nan
    )
    night = (
        float(np.abs(y_true[~mask] - y_pred[~mask]).mean())
        if (~mask).any()
        else np.nan
    )

    return day, night


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


def daylight_wmape(
    index,
    y_true,
    y_pred,
    **daylight_kwargs
):
    """Total daylight error as a percentage of total daylight production.

    MAPE divides each hour by its own production, so a dawn hour producing a
    couple of kWh can contribute a percentage error in the hundreds and drag the
    mean up regardless of how small the absolute miss was. Dividing summed error
    by summed production instead makes the denominator a period total that no
    single small hour can distort, which answers the question actually worth
    asking: what fraction of the energy generated did the forecast get wrong.

    No production threshold is applied. MAPE needs one to keep its per-hour
    denominators away from zero; this ratio has no such failure mode, so every
    daylight hour counts.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = daylight_mask_for(index=index, **daylight_kwargs)

    if not mask.any():
        return np.nan

    total_production = np.sum(y_true[mask])

    if total_production <= 0:
        return np.nan

    total_error = np.sum(
        np.abs(y_true[mask] - y_pred[mask])
    )

    return float(100 * total_error / total_production)


def daylight_mdape(
    index,
    y_true,
    y_pred,
    production_threshold=MAPE_PRODUCTION_THRESHOLD,
    **daylight_kwargs
):
    """Median absolute percentage error over scored daylight hours.

    Same mask and same production threshold as daylight_mape, so the pair is
    directly comparable and the only difference is mean against median. The mean
    is pulled up by a thin tail of hours whose percentage error is enormous
    because the denominator is small; the median reports what a typical daylight
    hour actually looks like.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = daylight_mask_for(index=index, **daylight_kwargs)
    scored = mask & (y_true > production_threshold)

    if not scored.any():
        return np.nan

    absolute_percentage_error = np.abs(
        (y_true[scored] - y_pred[scored]) / y_true[scored]
    )

    return float(100 * np.median(absolute_percentage_error))

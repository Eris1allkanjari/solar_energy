import numpy as np
import pandas as pd

from src.configs.dataset_selection import STEP
from src.configs.evaluation import (
    EXOG_LAG_STEPS,
    SITE_LATITUDE,
    SITE_LONGITUDE
)
from src.utils.solar import solar_elevation


# Sine of the sun's elevation at the middle of each interval, floored at zero.
# It is proportional to the irradiance a cloudless sky would deliver on a
# horizontal surface, so it tells a linear model how much production the sun's
# position allows in the interval being forecast.
SOLAR_ELEVATION_FEATURE = "solar_elevation_sin"

# Inputs computed from the timestamp alone. They are known for the interval
# being forecast before it starts, so unlike measured weather they are not
# lagged and using their current value leaks nothing.
KNOWN_IN_ADVANCE_FEATURES = (SOLAR_ELEVATION_FEATURE,)

# Cloud cover of the previous interval times the solar elevation of the interval
# being forecast. Plain cloud cover gets one linear coefficient for all hours, so
# its night-time values shift night forecasts away from zero; weighted by the
# sun it is zero at night and only acts while the sun is up.
DAYLIGHT_CLOUD_FEATURE = "cloud_cover_daylight"

# Inputs that are already built from the previous interval's measurement, so
# they must not be lagged a second time.
LAGGED_AT_SOURCE_FEATURES = (DAYLIGHT_CLOUD_FEATURE,)

DEFAULT_NEURAL_FEATURES = [
    "solar_radiation_Wm2",
    "cloud_cover_okta",
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent",
    "hour_sin",
    "hour_cos"
]

# KNMI's code for variable wind, which has no single bearing.
VARIABLE_WIND_CODE = 990


def fractional_hour(index):
    """Hour of day including minutes, so 10:40 becomes 10.667.

    At hourly resolution the minutes are zero and this is the plain hour. At
    10-minute resolution it keeps the six intervals of an hour apart; encoding
    the whole hour alone would give all six identical time features.
    """
    return index.hour + index.minute / 60


def add_solar_geometry(df):
    """Add the clear-sky proxy for each interval's own position of the sun.

    Rows are labelled by interval start, so the elevation is taken at the
    middle of the interval. Night gives a negative sine, which is floored at
    zero because the sun below the horizon contributes nothing.
    """
    midpoints = df.index + STEP / 2
    elevation = solar_elevation(midpoints, SITE_LATITUDE, SITE_LONGITUDE)
    df[SOLAR_ELEVATION_FEATURE] = np.clip(
        np.sin(np.deg2rad(elevation)),
        0.0,
        None
    )
    return df


def add_daylight_cloud(df):
    """Add previous-interval cloud cover weighted by the sun's elevation.

    Needs the cloud cover and solar elevation columns; without cloud cover the
    frame is returned unchanged.
    """
    if "cloud_cover_okta" not in df.columns:
        return df

    if SOLAR_ELEVATION_FEATURE not in df.columns:
        add_solar_geometry(df)

    # Fill gaps in cloud cover before weighting, the way the plain input is
    # filled; filling the product instead would carry an earlier hour's sun.
    df[DAYLIGHT_CLOUD_FEATURE] = (
        df["cloud_cover_okta"].ffill().shift(EXOG_LAG_STEPS)
        * df[SOLAR_ELEVATION_FEATURE]
    )
    return df


def add_time_features(df):
    add_solar_geometry(df)
    add_daylight_cloud(df)
    hour = fractional_hour(df.index)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    day_angle = 2 * np.pi * (df.index.dayofyear - 1) / 365.25
    df["day_of_year_sin"] = np.sin(day_angle)
    df["day_of_year_cos"] = np.cos(day_angle)

    if "wind_direction_deg" in df.columns:
        direction = df["wind_direction_deg"]
        direction_radians = np.deg2rad(direction % 360)

        # Variable and calm wind have no bearing, so both become the zero
        # vector. Taken literally, 990 % 360 would read as a west wind. Calm is
        # found through the speed rather than the direction: the validated
        # hourly product marks it as direction 0, but in the 10-minute product
        # 0 is a genuine northerly and calm comes with an arbitrary direction.
        no_direction = direction.eq(VARIABLE_WIND_CODE)

        if "wind_speed_ms" in df.columns:
            no_direction |= df["wind_speed_ms"].eq(0)

        df["wind_direction_sin"] = np.where(
            no_direction, 0.0, np.sin(direction_radians)
        )
        df["wind_direction_cos"] = np.where(
            no_direction, 0.0, np.cos(direction_radians)
        )

    return df


def add_target_time_features(df):
    # The target is the next row, one step after the row's own timestamp.
    target_hour = fractional_hour(df.index + STEP) % 24
    target_angle = 2 * np.pi * target_hour / 24
    # The last interval of the day starts at 23:00 hourly and 23:50 at 10
    # minutes; dividing by it keeps the linear encoding within [0, 1].
    latest_hour = 24 - STEP / pd.Timedelta(hours=1)

    df["target_hour_linear"] = target_hour / latest_hour
    df["target_hour_cosine"] = (1 - np.cos(target_angle)) / 2
    df["target_hour_sin"] = np.sin(target_angle)
    df["target_hour_cos"] = np.cos(target_angle)

    return df

def select_features(df):
    features = [
        "pv_total_kWh",
        *DEFAULT_NEURAL_FEATURES
    ]
    return df[features]

def clean_data(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if "pv_total_kWh" in df.columns:
        df["pv_total_kWh"] = df["pv_total_kWh"].clip(lower=0)

    # Impute causally. Forward-fill only carries past values across a gap, so
    # imputation never reaches backward over a train/test boundary. No backward
    # fill: the dataset has no missing values in its first row, so there is no
    # leading gap for forward-fill to fail on.
    return df.ffill()

import numpy as np
import pandas as pd

from src.configs.dataset_selection import STEP


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


def add_time_features(df):
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

import numpy as np


DEFAULT_NEURAL_FEATURES = [
    "solar_radiation_Wm2",
    "cloud_cover_okta",
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent",
    "hour_sin",
    "hour_cos"
]

def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    day_angle = 2 * np.pi * (df.index.dayofyear - 1) / 365.25
    df["day_of_year_sin"] = np.sin(day_angle)
    df["day_of_year_cos"] = np.cos(day_angle)

    if "wind_direction_deg" in df.columns:
        direction_radians = np.deg2rad(df["wind_direction_deg"] % 360)
        df["wind_direction_sin"] = np.sin(direction_radians)
        df["wind_direction_cos"] = np.cos(direction_radians)

    return df


def add_target_time_features(df):
    target_hour = (df.index.hour + 1) % 24
    target_angle = 2 * np.pi * target_hour / 24

    df["target_hour_linear"] = target_hour / 23
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

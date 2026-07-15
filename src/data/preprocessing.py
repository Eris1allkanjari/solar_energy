import numpy as np

def add_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    if "wind_direction_deg" in df.columns:
        direction_radians = np.deg2rad(df["wind_direction_deg"] % 360)
        df["wind_direction_sin"] = np.sin(direction_radians)
        df["wind_direction_cos"] = np.cos(direction_radians)

    return df

def select_features(df):
    features = [
        "pv_total_kWh",
        "solar_radiation_Wm2",
        "cloud_cover_okta",
        "temperature_C",
        "wind_speed_ms",
        "humidity_percent",
        "hour_sin",
        "hour_cos"
    ]
    return df[features]

def clean_data(df):
    return df.interpolate().bfill().ffill()

import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("dataset must contain a time column")

    df["time"] = pd.to_datetime(df["time"])

    if df["time"].duplicated().any():
        raise ValueError("dataset contains duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("dataset timestamps must be sorted")

    df = df.set_index("time").asfreq("h")

    if "pv_total_kWh" not in df.columns:
        raise ValueError("dataset must contain pv_total_kWh")

    return df

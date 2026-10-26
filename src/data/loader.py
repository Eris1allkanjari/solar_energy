import pandas as pd
from pandas.tseries.frequencies import to_offset

from src.configs.dataset_selection import ACTIVE_DATASET


def load_dataset(path, freq=None):
    """Load a modelling dataset on a regular grid of the given frequency.

    The frequency defaults to the active dataset's, so callers need not pass it.
    asfreq keeps only the timestamps on the requested grid, so loading 10-minute
    data at an hourly frequency would silently discard five rows in six. Data
    finer than the requested frequency is therefore refused rather than thinned.
    """
    freq = ACTIVE_DATASET.freq if freq is None else freq
    df = pd.read_csv(path)

    if "time" not in df.columns:
        raise ValueError("dataset must contain a time column")

    df["time"] = pd.to_datetime(df["time"])

    if df["time"].duplicated().any():
        raise ValueError("dataset contains duplicate timestamps")

    if not df["time"].is_monotonic_increasing:
        raise ValueError("dataset timestamps must be sorted")

    df = df.set_index("time")
    finest_step = df.index.to_series().diff().min()

    if pd.notna(finest_step) and finest_step < pd.Timedelta(to_offset(freq)):
        raise ValueError(
            f"dataset has {finest_step} steps, finer than the requested "
            f"frequency {freq!r}; pass freq to load it at its own resolution"
        )

    df = df.asfreq(freq)

    if "pv_total_kWh" not in df.columns:
        raise ValueError("dataset must contain pv_total_kWh")

    return df

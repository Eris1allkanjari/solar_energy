"""Merge PV production with KNMI weather into modelling datasets.

    python script.py                                   # the two datasets in use
    python script.py --resolution hourly_validated     # the legacy dataset

The datasets in use share one weather source, KNMI's 10-minute observations:

    10min               the 10-minute record, merged with 10-minute PV energy
    hourly_from_10min   the 10-minute dataset aggregated to hours

Building the hourly dataset from the 10-minute one guarantees the two rest on
identical readings and identical gap fills. The validated hourly product is
kept as hourly_validated but is no longer used.

Every dataset puts PV and weather on one timestamp convention, so that row t
describes the interval from t to t+step for the target and the weather alike,
and each refuses to be written if production and radiation are out of step. See
to_interval_start and check_alignment.
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
UTRECHT_DIR = PROJECT_ROOT / "utrecht"

HOUR = pd.Timedelta(hours=1)
ALIGNMENT_CHECK_LAGS = range(-3, 4)

RESOLUTIONS = {
    "hourly_validated": {
        "step": HOUR,
        "pv": UTRECHT_DIR / "pv_data" / "pv_hourly_2014_2017.csv",
        "weather": UTRECHT_DIR / "knmi_data" / "knmi_hourly_2014_2017.csv",
        "output": UTRECHT_DIR / "processed_data" / "utrecht_pv_data.csv",
    },
    "10min": {
        "step": pd.Timedelta(minutes=10),
        "pv": UTRECHT_DIR / "pv_data" / "pv_10min_2014_2017.csv",
        "weather": UTRECHT_DIR / "knmi_data" / "knmi_10min_2014_2017.csv",
        "output": UTRECHT_DIR / "processed_data" / "utrecht_pv_data_10min.csv",
    },
    "hourly_from_10min": {
        "step": HOUR,
        "source": UTRECHT_DIR / "processed_data" / "utrecht_pv_data_10min.csv",
        "output": (
            UTRECHT_DIR
            / "processed_data"
            / "utrecht_pv_data_hourly_from_10min.csv"
        ),
    },
}

# The datasets in use, in build order: the hourly one is aggregated from the
# finished 10-minute one.
DEFAULT_RESOLUTIONS = ["10min", "hourly_from_10min"]

# KNMI's hourly product codes variable wind as 990. The 10-minute product never
# uses it, so a gap fill must not carry it across.
KNMI_VARIABLE_WIND = 990

# Weather readings that describe a moment rather than a period. KNMI's hourly
# value for these is the reading at the end of the hour.
SNAPSHOT_COLUMNS = [
    "wind_direction_deg",
    "wind_speed_ms",
    "cloud_cover_okta",
    "pressure_hpa",
    "temperature_C",
    "humidity_percent",
]

# Precision of the validated hourly file, used to compare our aggregation with
# it: a value counts as agreeing when it rounds to the published figure.
# Radiation is published as whole J/cm2 per hour, steps of 2.7778 W/m2.
VALIDATED_PRECISION = {
    "wind_direction_deg": 10,
    "wind_speed_ms": 1,
    "cloud_cover_okta": 1,
    "pressure_hpa": 0.1,
    "temperature_C": 0.1,
    "humidity_percent": 1,
    "solar_radiation_Wm2": 10000 / 3600,
    "sunshine_hours": 0.1,
}

# Weather columns both resolutions share, in the hourly file's order.
SHARED_WEATHER_COLUMNS = [
    "wind_direction_deg",
    "wind_speed_ms",
    "cloud_cover_okta",
    "pressure_hpa",
    "sunshine_hours",
    "temperature_C",
    "humidity_percent",
    "solar_radiation_Wm2",
]

# Totals over the interval rather than rates or states, so an hourly value has
# to be scaled to the interval length before it can stand in for a 10-minute one.
ACCUMULATED_COLUMNS = ["sunshine_hours"]

RADIATION_RANGE_COLUMNS = [
    "solar_radiation_min_Wm2",
    "solar_radiation_max_Wm2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge PV production with KNMI weather."
    )
    parser.add_argument(
        "--resolution",
        nargs="+",
        choices=list(RESOLUTIONS),
        default=DEFAULT_RESOLUTIONS,
        help=(
            "Datasets to build (default: 10min then hourly_from_10min, the two "
            "in use)."
        )
    )
    return parser.parse_args()


def normalize_timestamp(series):
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def read_timestamped(path, usecols=None):
    frame = pd.read_csv(path, usecols=usecols)
    frame["time"] = normalize_timestamp(frame["time"])

    if frame["time"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate timestamps")

    return frame


def to_interval_start(weather_df, step):
    """Relabel KNMI observations from interval end to interval start.

    KNMI labels a value by the end of the interval it describes, so at hourly
    resolution 10:00 holds the radiation received from 09:00 to 10:00, and at
    10-minute resolution 10:10 holds 10:00 to 10:10. The PV target is built
    with resample, which labels by the start. Merging on the raw labels paired
    every PV interval with the weather of the interval before it. Shifting the
    weather back one step puts both on one convention: row t describes the
    interval from t to t+step.

    No information crosses the forecast origin. A model predicting row t+1
    reads rows up to t, and everything now stored in row t, including snapshot
    variables taken at the end of that interval, is observed before t+1, when
    the forecast is issued.
    """
    weather_df = weather_df.copy()
    weather_df["time"] = weather_df["time"] - step
    return weather_df


def alignment_correlations(
    merged,
    target="pv_total_kWh",
    reference="solar_radiation_Wm2"
):
    series = merged.set_index("time")

    return {
        lag: series[target].corr(series[reference].shift(-lag))
        for lag in ALIGNMENT_CHECK_LAGS
    }


def check_alignment(merged, step=HOUR):
    """Fail unless production and radiation line up at lag zero.

    Radiation drives production within the same interval, so on correctly
    aligned data their correlation peaks with no shift. A peak at any other lag
    means the two sources use different labelling conventions, which is the
    error the hourly merge once contained without anything noticing.
    """
    correlations = alignment_correlations(merged)
    best_lag = max(correlations, key=correlations.get)

    if best_lag != 0:
        table = ", ".join(
            f"{lag:+d}: {value:.4f}" for lag, value in correlations.items()
        )
        raise ValueError(
            "PV and radiation are misaligned: correlation peaks at lag "
            f"{best_lag:+d} steps of {step} instead of 0 ({table})"
        )

    return correlations


def print_alignment(correlations, step):
    minutes = int(step / pd.Timedelta(minutes=1))
    print("PV vs radiation correlation by lag:")

    for lag, value in correlations.items():
        marker = "  <- peak" if lag == 0 else ""
        print(f"  {lag * minutes:+5d} min  {value:.4f}{marker}")


def build_hourly_validated(config):
    pv_df = read_timestamped(config["pv"], usecols=["time", "pv_total_kWh"])
    weather_df = to_interval_start(
        read_timestamped(config["weather"]),
        config["step"]
    )

    merged = weather_df.merge(
        pv_df,
        on="time",
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    pv_start = pv_df["time"].min()
    pv_end = pv_df["time"].max()
    weather_only = merged["_merge"] == "left_only"
    pv_only = merged["_merge"] == "right_only"

    # The relabel has two expected edge effects: the first KNMI hour now starts
    # before the PV record, and the final PV hour has no weather row because its
    # KNMI end label falls past the fetched range. Anything else is a genuine
    # gap and fails, as the strict merge always did.
    unexpected = (
        weather_only & merged["time"].between(pv_start, pv_end)
    ) | (
        pv_only & (merged["time"] != pv_end)
    )

    if unexpected.any():
        counts = merged.loc[unexpected, "_merge"].value_counts()
        raise ValueError(
            f"PV and weather timestamps do not match: {counts.to_dict()}"
        )

    merged = merged[~weather_only].drop(columns="_merge").sort_values("time")
    print(
        f"PV hours without weather: {int(pv_only.sum())} "
        "(final hour; filled causally by clean_data)"
    )

    return merged


def hourly_weather_aligned():
    config = RESOLUTIONS["hourly_validated"]
    weather = to_interval_start(
        read_timestamped(config["weather"]),
        config["step"]
    )
    return weather.set_index("time")[SHARED_WEATHER_COLUMNS]


def fill_from_hourly(merged, step):
    """Fill 10-minute station gaps from the latest completed hourly value.

    The 10-minute record has gaps, one of them nearly eight hours on a summer
    morning, and forward fill would carry pre-dawn weather through the whole of
    it. The hourly record is validated and complete there, so gaps take the
    hourly value instead: specifically the most recent hour that has fully
    ended by the time a forecast from this row is issued, at t+step. That keeps
    the fill causal, never more than about seventy minutes old, and never
    drawn from the hour still in progress.

    Accumulated quantities are scaled to the interval length, which assumes
    the hour was uniform. Filled rows are flagged in weather_from_hourly.
    """
    hourly = hourly_weather_aligned()
    source_hour = (merged["time"] + step).dt.floor("h") - HOUR
    fallback = hourly.reindex(source_hour.to_numpy())
    fallback.index = merged.index

    filled = pd.DataFrame(False, index=merged.index, columns=[])

    for column in SHARED_WEATHER_COLUMNS:
        values = fallback[column]

        if column in ACCUMULATED_COLUMNS:
            values = values * (step / HOUR)

        if column == "wind_direction_deg":
            values = values.where(values != KNMI_VARIABLE_WIND)

        filled[column] = merged[column].isna() & values.notna()
        merged[column] = merged[column].fillna(values)

    # Without 10-minute readings the within-interval range is unknown, so the
    # range collapses onto the mean rather than being invented. Where the mean
    # itself came from the hourly record, any partial raw minimum or maximum is
    # overwritten too: KNMI sometimes reports the minimum alone, and keeping it
    # beside an hourly mean can leave the minimum above the mean.
    radiation_filled = filled["solar_radiation_Wm2"]

    for column in RADIATION_RANGE_COLUMNS:
        merged.loc[radiation_filled, column] = merged.loc[
            radiation_filled,
            "solar_radiation_Wm2"
        ]
        merged[column] = merged[column].fillna(merged["solar_radiation_Wm2"])

    merged["weather_from_hourly"] = filled.any(axis=1)

    return merged


def build_10min(config):
    step = config["step"]
    pv_df = read_timestamped(
        config["pv"],
        usecols=[
            "time",
            "pv_total_kWh",
            "observation_fraction",
            "minute_rows"
        ]
    )
    weather_df = to_interval_start(read_timestamped(config["weather"]), step)

    expected_grid = pd.date_range(
        pv_df["time"].min(),
        pv_df["time"].max(),
        freq=step
    )

    if not pd.DatetimeIndex(pv_df["time"]).equals(expected_grid):
        raise ValueError("10-minute PV record is not a complete regular grid")

    off_grid = ~weather_df["time"].isin(expected_grid) & weather_df[
        "time"
    ].between(expected_grid[0], expected_grid[-1])

    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum())} weather timestamps fall between PV intervals"
        )

    # PV defines the rows: every 10-minute interval of the record is kept, and
    # the station gaps in the weather are then filled from the hourly record.
    merged = weather_df.merge(
        pv_df,
        on="time",
        how="right",
        validate="one_to_one"
    ).sort_values("time").reset_index(drop=True)

    station_gaps = int(merged[SHARED_WEATHER_COLUMNS].isna().any(axis=1).sum())
    merged = fill_from_hourly(merged, step)
    still_missing = merged[SHARED_WEATHER_COLUMNS].isna().any(axis=1)

    print(f"intervals with a weather gap : {station_gaps}")
    print(
        "filled from the hourly record: "
        f"{int(merged['weather_from_hourly'].sum())}"
    )
    print(
        f"still missing               : {int(still_missing.sum())} "
        f"{merged.loc[still_missing, 'time'].dt.strftime('%Y-%m-%d %H:%M').tolist()}"
    )

    return merged


def build_hourly_from_10min(config):
    """Aggregate the 10-minute dataset to hours.

    Built from the finished 10-minute dataset rather than from the raw files,
    so both resolutions share identical readings and identical gap fills, and
    any difference between hourly and 10-minute results comes from the
    resolution alone. Each rule mirrors how KNMI derives its own hourly
    product from the same 10-minute measurements:

    - PV energy and sunshine are totals over the hour, so they are summed.
    - Radiation is a mean irradiance, so it is averaged.
    - Snapshot readings take the last 10-minute value of the hour, which is the
      reading at the hour's end, the time KNMI's hourly value refers to.
    - Coverage is weighted by the minute rows each interval holds, which is how
      the PV quality audit defines it.

    Sunshine is left missing when any of the hour's intervals is missing,
    rather than summed over a partial hour and understated.
    """
    ten = read_timestamped(config["source"]).set_index("time")
    hour = ten.index.floor("h")
    intervals_per_hour = int(HOUR / RESOLUTIONS["10min"]["step"])
    counts = ten.groupby(hour).size()

    if not counts.eq(intervals_per_hour).all():
        raise ValueError(
            "10-minute dataset has hours without exactly "
            f"{intervals_per_hour} intervals"
        )

    grouped = ten.groupby(hour)
    hourly = pd.DataFrame(index=counts.index)
    hourly.index.name = "time"

    for column in SNAPSHOT_COLUMNS:
        hourly[column] = grouped[column].last()

    hourly["sunshine_hours"] = grouped["sunshine_hours"].sum(
        min_count=intervals_per_hour
    )
    hourly["solar_radiation_Wm2"] = grouped["solar_radiation_Wm2"].mean()
    hourly["pv_total_kWh"] = grouped["pv_total_kWh"].sum()

    observed = (ten["observation_fraction"] * ten["minute_rows"]).groupby(
        hour
    ).sum()
    minute_rows = grouped["minute_rows"].sum()
    hourly["observation_fraction"] = (observed / minute_rows).fillna(0.0)
    hourly["minute_rows"] = minute_rows
    hourly["weather_from_hourly"] = grouped["weather_from_hourly"].any()

    ordered = SHARED_WEATHER_COLUMNS + [
        "pv_total_kWh",
        "observation_fraction",
        "minute_rows",
        "weather_from_hourly",
    ]

    merged = hourly[ordered].reset_index()
    compare_with_validated(merged)

    return merged


def compare_with_validated(merged):
    """Print how our hourly aggregation agrees with KNMI's validated product.

    KNMI states its validated hourly data derives from the same 10-minute
    measurements. A value agrees when it rounds to the validated file's own
    precision. Wind direction is compared only where the validated product
    gives a bearing, not its codes for calm (0) or variable wind (990). The
    table is printed on every build so the evidence behind using our own
    aggregation is regenerated with the data.
    """
    validated_path = RESOLUTIONS["hourly_validated"]["output"]

    if not validated_path.exists():
        print("validated hourly dataset not found; comparison skipped")
        return

    validated = read_timestamped(validated_path).set_index("time")
    ours = merged.set_index("time").reindex(validated.index)

    pv_difference = (ours["pv_total_kWh"] - validated["pv_total_kWh"]).abs()
    print(
        "PV vs validated dataset       : max difference "
        f"{pv_difference.max():.2e} kWh"
    )
    print("agreement with the validated hourly product, after rounding:")

    for column, precision in VALIDATED_PRECISION.items():
        published = validated[column]
        aggregated = ours[column]
        comparable = published.notna() & aggregated.notna()

        if column == "wind_direction_deg":
            comparable &= ~published.isin([0, KNMI_VARIABLE_WIND])

        rounded = ((aggregated / precision + 0.5) // 1) * precision
        difference = (published - rounded).abs()

        if column == "wind_direction_deg":
            difference = difference.where(difference <= 180, 360 - difference)

        agree = difference[comparable] < precision / 2
        print(f"  {column:22s} {agree.mean():7.1%} of {int(comparable.sum())} hours")


BUILDERS = {
    "hourly_validated": build_hourly_validated,
    "10min": build_10min,
    "hourly_from_10min": build_hourly_from_10min,
}


def main():
    args = parse_args()

    for resolution in args.resolution:
        config = RESOLUTIONS[resolution]
        print(f"\n=== {resolution} ===")

        merged = BUILDERS[resolution](config)
        correlations = check_alignment(merged, config["step"])

        config["output"].parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(config["output"], index=False)

        print(f"rows written: {len(merged)}")
        print_alignment(correlations, config["step"])
        print(f"saved {config['output']}")


if __name__ == "__main__":
    main()

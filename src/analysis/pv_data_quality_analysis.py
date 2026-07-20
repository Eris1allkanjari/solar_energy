import matplotlib.pyplot as plt
import pandas as pd

from src.configs.evaluation import TRAIN_RATIO, VALIDATION_END_RATIO
from src.experiments.constants import (
    ANALYSIS_RESULTS_DIR,
    RAW_PV_FILE_PATH
)


RESULTS_DIR = ANALYSIS_RESULTS_DIR
CHUNK_SIZE = 100_000
EXPECTED_MINUTES_PER_HOUR = 60


def aggregate_chunk(chunk, system_columns):
    timestamps = pd.to_datetime(chunk.pop("DateTime"), utc=True)
    observed_systems = chunk[system_columns].notna().sum(axis=1)
    minute_quality = pd.DataFrame(
        {
            "time": timestamps.dt.floor("h"),
            "minute_rows": 1,
            "observed_readings": observed_systems,
            "reporting_systems_sum": observed_systems,
            "fully_missing_minutes": observed_systems.eq(0).astype(int)
        }
    )
    return minute_quality.groupby("time", as_index=False).sum()


def load_hourly_quality(path=RAW_PV_FILE_PATH, chunk_size=CHUNK_SIZE):
    header = pd.read_csv(path, nrows=0)
    system_columns = [
        column for column in header.columns
        if column != "DateTime"
    ]
    partial_results = []

    for chunk_index, chunk in enumerate(
        pd.read_csv(path, chunksize=chunk_size),
        start=1
    ):
        print(f"processing PV quality chunk {chunk_index}")
        partial_results.append(
            aggregate_chunk(chunk, system_columns)
        )

    hourly = pd.concat(partial_results, ignore_index=True)
    hourly = hourly.groupby("time", as_index=False).sum()
    expected_readings = (
        hourly["minute_rows"] * len(system_columns)
    )
    hourly["observation_fraction"] = (
        hourly["observed_readings"] / expected_readings
    )
    hourly["mean_reporting_systems"] = (
        hourly["reporting_systems_sum"] / hourly["minute_rows"]
    )
    hourly["fully_missing_hour"] = hourly["observed_readings"].eq(0)
    hourly["complete_hour"] = hourly["minute_rows"].eq(
        EXPECTED_MINUTES_PER_HOUR
    )
    return hourly


def assign_split(hourly):
    split = pd.Series("test", index=hourly.index, dtype="object")
    train_end = int(len(hourly) * TRAIN_RATIO)
    validation_end = int(len(hourly) * VALIDATION_END_RATIO)
    split.iloc[:train_end] = "train"
    split.iloc[train_end:validation_end] = "validation"
    return split


def summarize_quality(hourly):
    quality = hourly.copy()
    quality["split"] = assign_split(quality)
    rows = []

    for split_name, split_df in quality.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "start": split_df["time"].iloc[0],
                "end": split_df["time"].iloc[-1],
                "hours": len(split_df),
                "fully_missing_hours": int(
                    split_df["fully_missing_hour"].sum()
                ),
                "fully_missing_hour_fraction": float(
                    split_df["fully_missing_hour"].mean()
                ),
                "hours_below_50_percent_observed": int(
                    (split_df["observation_fraction"] < 0.50).sum()
                ),
                "mean_observation_fraction": float(
                    split_df["observation_fraction"].mean()
                ),
                "median_observation_fraction": float(
                    split_df["observation_fraction"].median()
                ),
                "minimum_observation_fraction": float(
                    split_df["observation_fraction"].min()
                ),
                "incomplete_timestamp_hours": int(
                    (~split_df["complete_hour"]).sum()
                )
            }
        )

    return pd.DataFrame(rows)


def plot_quality(hourly, output_path):
    daily = hourly.set_index("time")[
        "observation_fraction"
    ].resample("1D").mean()
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(daily.index, daily, linewidth=0.8)
    axis.axhline(0.5, color="red", linestyle="--", linewidth=1)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Observed PV readings / expected readings")
    axis.set_title("Daily PV measurement availability")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main():
    hourly = load_hourly_quality()
    summary = summarize_quality(hourly)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hourly_path = RESULTS_DIR / "pv_data_quality_hourly.csv"
    summary_path = RESULTS_DIR / "pv_data_quality_summary.csv"
    plot_path = RESULTS_DIR / "pv_data_quality.png"
    hourly.to_csv(hourly_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_quality(hourly, plot_path)
    print(summary.to_string(index=False))
    print(f"saved hourly quality to {hourly_path}")
    print(f"saved summary to {summary_path}")
    print(f"saved plot to {plot_path}")


if __name__ == "__main__":
    main()

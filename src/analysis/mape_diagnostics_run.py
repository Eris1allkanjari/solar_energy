"""Per-hour MAPE diagnostic log for the test period.

Writes every test timestamp with its actual value, predicted value, percentage
error and position within the day, followed by the MAPE the pipeline computes
over the same data. The point is to make the per-hour arithmetic inspectable, so
the shape of the percentage error can be read off the data rather than argued
from summary statistics.
"""

import argparse

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    DAYLIGHT_ELEVATION_DEGREES,
    MAPE_PRODUCTION_THRESHOLD,
    SITE_LATITUDE,
    SITE_LONGITUDE
)
from src.experiments.constants import (
    ANALYSIS_RESULTS_DIR,
    FINAL_COMPARISON_RESULTS_DIR
)
from src.training.evaluation import evaluate
from src.utils.solar import solar_elevation


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a per-hour MAPE diagnostic log for one model's test "
            "predictions."
        )
    )
    parser.add_argument(
        "--model",
        default="gru",
        help="Model whose saved test predictions are logged (default: gru)."
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=40,
        help="How many worst-APE hours to list in the trailing summary."
    )
    return parser.parse_args()


def load_predictions(model_name):
    path = (
        FINAL_COMPARISON_RESULTS_DIR
        / f"{model_name}_final_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run the final model comparison first."
        )

    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.set_index("time")


def add_day_position(frame):
    """Where each hour sits inside its own daylight span.

    Clock hour alone is not comparable across the year at this latitude, where
    day length varies by more than eight hours, so each daylight hour is also
    placed on a 0-to-1 axis running from that date's first lit hour to its last.
    """
    index = pd.DatetimeIndex(frame.index)
    frame["solar_elevation_deg"] = solar_elevation(
        index=index,
        latitude=SITE_LATITUDE,
        longitude=SITE_LONGITUDE
    )
    frame["is_daylight"] = (
        frame["solar_elevation_deg"] > DAYLIGHT_ELEVATION_DEGREES
    )
    frame["hour_of_day"] = index.hour
    frame["day_fraction"] = np.nan

    for _, day_rows in frame.groupby(index.date):
        lit = day_rows[day_rows["is_daylight"]]

        if len(lit) < 2:
            continue

        first_hour = lit["hour_of_day"].iloc[0]
        last_hour = lit["hour_of_day"].iloc[-1]
        span = last_hour - first_hour

        if span <= 0:
            continue

        frame.loc[lit.index, "day_fraction"] = (
            lit["hour_of_day"] - first_hour
        ) / span

    return frame


def add_errors(frame, production_threshold):
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)

    frame["error"] = y_pred - y_true

    # The percentage error is undefined where production is zero, which is
    # exactly the behaviour under investigation, so it is computed without
    # guarding and the non-finite results are kept visible in the log.
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["pct_error"] = 100 * (y_pred - y_true) / y_true

    frame["abs_pct_error"] = np.abs(frame["pct_error"])
    frame["counted_in_mape"] = y_true > production_threshold
    return frame


def format_row(timestamp, row):
    def number(value, width, decimals):
        if value is None or not np.isfinite(value):
            return f"{'n/a':>{width}}"
        return f"{value:{width}.{decimals}f}"

    day_fraction = (
        f"{row['day_fraction']:6.3f}"
        if np.isfinite(row["day_fraction"])
        else f"{'-':>6}"
    )

    return (
        f"{timestamp:%Y-%m-%d %H:%M}"
        f" {row['hour_of_day']:>4d}"
        f" {day_fraction}"
        f" {number(row['solar_elevation_deg'], 8, 2)}"
        f" {'day' if row['is_daylight'] else 'night':>5}"
        f" {number(row['y_true'], 10, 3)}"
        f" {number(row['y_pred'], 10, 3)}"
        f" {number(row['error'], 10, 3)}"
        f" {number(row['pct_error'], 12, 2)}"
        f" {'yes' if row['counted_in_mape'] else 'no':>4}"
    )


HEADER = (
    f"{'timestamp':16}"
    f" {'hour':>4}"
    f" {'dayfrac':>6}"
    f" {'elev_deg':>8}"
    f" {'period':>5}"
    f" {'actual_kWh':>10}"
    f" {'pred_kWh':>10}"
    f" {'error_kWh':>10}"
    f" {'pct_error':>12}"
    f" {'mape':>4}"
)


def production_band_summary(frame):
    """Mean APE and MAPE contribution by actual-production band."""
    counted = frame[frame["counted_in_mape"]].copy()
    edges = [0, 10, 20, 40, 80, 160, np.inf]
    labels = [
        "5-10", "10-20", "20-40", "40-80", "80-160", "160+"
    ]
    counted["band"] = pd.cut(
        counted["y_true"],
        bins=edges,
        labels=labels,
        right=False
    )
    total = len(counted)
    lines = []

    for band, rows in counted.groupby("band", observed=True):
        if rows.empty:
            continue

        share = len(rows) / total
        contribution = rows["abs_pct_error"].sum() / total
        lines.append(
            f"  {str(band):>8} kWh"
            f"  hours={len(rows):5d} ({100 * share:5.1f}%)"
            f"  mean_APE={rows['abs_pct_error'].mean():8.2f}%"
            f"  mean_abs_error={rows['error'].abs().mean():7.2f} kWh"
            f"  contributes={contribution:6.2f} pts of MAPE"
        )

    return lines


def main():
    args = parse_args()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_predictions(args.model)
    frame = add_day_position(frame)
    frame = add_errors(frame, MAPE_PRODUCTION_THRESHOLD)

    mae, rmse, mape = evaluate(
        frame["y_true"].to_numpy(),
        frame["y_pred"].to_numpy()
    )

    counted = frame[frame["counted_in_mape"]]
    excluded = frame[~frame["counted_in_mape"]]
    output_path = ANALYSIS_RESULTS_DIR / f"mape_diagnostics_{args.model}.log"

    with open(output_path, "w", encoding="utf-8") as log:
        log.write(f"MAPE diagnostics for {args.model}\n")
        log.write("=" * 110 + "\n\n")
        log.write(
            f"test period          : {frame.index[0]} to {frame.index[-1]}\n"
        )
        log.write(f"test hours           : {len(frame)}\n")
        log.write(
            f"production threshold : {MAPE_PRODUCTION_THRESHOLD} kWh "
            "(hours at or below are excluded from MAPE)\n"
        )
        log.write(
            f"hours counted in MAPE: {len(counted)} "
            f"({100 * len(counted) / len(frame):.1f}%)\n"
        )
        log.write(f"hours excluded       : {len(excluded)}\n\n")
        log.write("METRICS OVER THIS TEST PERIOD, COMPUTED FROM THESE ROWS\n")
        log.write(f"  MAE  = {mae:.6f} kWh\n")
        log.write(f"  RMSE = {rmse:.6f} kWh\n")
        log.write(f"  MAPE = {mape:.6f} %\n\n")
        log.write(
            "pct_error is (predicted - actual) / actual * 100, signed.\n"
            "MAPE averages its absolute value over the rows marked "
            "mape=yes.\n"
            "dayfrac is 0 at the first lit hour of that date and 1 at the "
            "last.\n\n"
        )
        log.write(HEADER + "\n")
        log.write("-" * len(HEADER) + "\n")

        for timestamp, row in frame.iterrows():
            log.write(format_row(timestamp, row) + "\n")

        log.write("\n\n")
        log.write("=" * 110 + "\n")
        log.write("BREAKDOWN BY ACTUAL PRODUCTION (MAPE-counted hours only)\n")
        log.write("=" * 110 + "\n")

        for line in production_band_summary(frame):
            log.write(line + "\n")

        log.write("\n")
        log.write("=" * 110 + "\n")
        log.write(f"WORST {args.worst} HOURS BY ABSOLUTE PERCENTAGE ERROR\n")
        log.write("=" * 110 + "\n")
        log.write(HEADER + "\n")
        log.write("-" * len(HEADER) + "\n")

        worst = counted.nlargest(args.worst, "abs_pct_error")

        for timestamp, row in worst.iterrows():
            log.write(format_row(timestamp, row) + "\n")

        log.write("\n")
        log.write("=" * 110 + "\n")
        log.write("MEAN APE BY HOUR OF DAY (MAPE-counted hours only)\n")
        log.write("=" * 110 + "\n")

        for hour, rows in counted.groupby("hour_of_day"):
            log.write(
                f"  hour {hour:02d}"
                f"  hours={len(rows):5d}"
                f"  mean_actual={rows['y_true'].mean():8.2f} kWh"
                f"  mean_abs_error={rows['error'].abs().mean():7.2f} kWh"
                f"  mean_APE={rows['abs_pct_error'].mean():8.2f}%\n"
            )

    print(f"saved {output_path}")
    print(f"MAPE over the test period: {mape:.4f}%")
    print(f"MAE  over the test period: {mae:.4f} kWh")


if __name__ == "__main__":
    main()

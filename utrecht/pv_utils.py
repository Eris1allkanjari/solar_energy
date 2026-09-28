"""Aggregate the 1-minute PV power measurements to interval energy.

Each system reports AC power in watts once a minute. Dividing by 60 gives the
energy of that minute in Wh, summing gives the interval total, and dividing by
1000 gives kWh. The target, pv_total_kWh, is the sum over all systems.

Intervals are labelled by their start, so 10:00 holds 10:00 to 10:59 at hourly
resolution and 10:00 to 10:09 at 10-minute resolution.

Missing minutes count as zero in the sum, so an outage looks like low
production. The coverage columns make that visible: minute_rows is how many
minute rows the interval contains, and observation_fraction is the share of
expected system readings actually present, defined exactly as in the PV quality
audit so the two agree.

    python utrecht/pv_utils.py                 # hourly
    python utrecht/pv_utils.py --freq 10min    # 10-minute
"""

import argparse
from pathlib import Path

import pandas as pd

UTRECHT_DIR = Path(__file__).resolve().parent
RAW_PATH = UTRECHT_DIR / "pv_data" / "filtered_pv_power_measurements_ac.csv"

OUTPUT_PATHS = {
    "1h": UTRECHT_DIR / "pv_data" / "pv_hourly_2014_2017.csv",
    "10min": UTRECHT_DIR / "pv_data" / "pv_10min_2014_2017.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate 1-minute PV power to interval energy."
    )
    parser.add_argument(
        "--freq",
        nargs="+",
        choices=list(OUTPUT_PATHS),
        default=["1h"],
        help="Resolutions to build (default: 1h). The raw file is read once."
    )
    return parser.parse_args()


def load_power():
    pv_df = pd.read_csv(RAW_PATH)
    pv_df = pv_df.rename(columns={"DateTime": "time"})
    pv_df["time"] = pd.to_datetime(pv_df["time"], utc=True)
    return pv_df.set_index("time")


def aggregate(pv_power, freq):
    pv_energy = pv_power / 60
    pv_interval = pv_energy.resample(freq).sum() / 1000
    pv_interval["pv_total_kWh"] = pv_interval.sum(axis=1)

    minute_rows = pv_power.index.to_series().resample(freq).count()
    observed_readings = pv_power.notna().resample(freq).sum().sum(axis=1)
    expected_readings = minute_rows * pv_power.shape[1]

    pv_interval["minute_rows"] = minute_rows
    # An interval with no minute rows at all has nothing observed, which is a
    # coverage of zero rather than an undefined ratio.
    pv_interval["observation_fraction"] = (
        observed_readings / expected_readings
    ).fillna(0.0)

    return pv_interval


def main():
    args = parse_args()
    pv_power = load_power()

    for freq in args.freq:
        output_path = OUTPUT_PATHS[freq]
        aggregate(pv_power, freq).to_csv(output_path)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()

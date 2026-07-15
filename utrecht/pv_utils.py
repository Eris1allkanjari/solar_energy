from pathlib import Path

import pandas as pd

UTRECHT_DIR = Path(__file__).resolve().parent


def main():
    pv_df = pd.read_csv(
        UTRECHT_DIR / "pv_data" / "filtered_pv_power_measurements_ac.csv"
    )
    pv_df = pv_df.rename(columns={"DateTime": "time"})

    pv_df["time"] = pd.to_datetime(pv_df["time"], utc=True)
    pv_df = pv_df.set_index("time")

    pv_energy = pv_df / 60
    pv_hourly = pv_energy.resample("1h").sum() / 1000
    pv_hourly["pv_total_kWh"] = pv_hourly.sum(axis=1)

    output_path = UTRECHT_DIR / "pv_data" / "pv_hourly_2014_2017.csv"
    pv_hourly.to_csv(output_path)


if __name__ == "__main__":
    main()

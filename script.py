from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
UTRECHT_DIR = PROJECT_ROOT / "utrecht"


def normalize_timestamp(series):
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def main():
    pv_df = pd.read_csv(
        UTRECHT_DIR / "pv_data" / "pv_hourly_2014_2017.csv"
    )
    weather_df = pd.read_csv(
        UTRECHT_DIR / "knmi_data" / "knmi_hourly_2014_2017.csv"
    )

    pv_df = pv_df[["time", "pv_total_kWh"]].copy()
    weather_df = weather_df.copy()
    pv_df["time"] = normalize_timestamp(pv_df["time"])
    weather_df["time"] = normalize_timestamp(weather_df["time"])

    if pv_df["time"].duplicated().any():
        raise ValueError("PV data contains duplicate hourly timestamps")
    if weather_df["time"].duplicated().any():
        raise ValueError("Weather data contains duplicate hourly timestamps")

    merged_pv_data = weather_df.merge(
        pv_df,
        on="time",
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    unmatched = merged_pv_data["_merge"] != "both"
    if unmatched.any():
        counts = merged_pv_data.loc[unmatched, "_merge"].value_counts()
        raise ValueError(
            "PV and weather timestamps do not match: "
            f"{counts.to_dict()}"
        )

    merged_pv_data = merged_pv_data.drop(
        columns="_merge"
    ).sort_values("time")

    output_path = UTRECHT_DIR / "processed_data" / "utrecht_pv_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_pv_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()

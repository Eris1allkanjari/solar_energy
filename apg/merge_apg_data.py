import re
from pathlib import Path

import pandas as pd


INPUT_FILES = [
    "./data/AGPT_2022-12-31T23_00_00Z_2023-12-31T23_00_00Z_15M_en_2026-06-10T17_21_35Z.csv",
    "./data/AGPT_2023-12-31T23_00_00Z_2024-12-31T23_00_00Z_15M_en_2026-06-10T17_20_30Z.csv",
    "./data/AGPT_2024-12-31T23_00_00Z_2025-12-31T23_00_00Z_15M_en_2026-06-10T17_19_23Z.csv",
    "./data/AGPT_2025-12-31T23_00_00Z_2026-06-10T22_00_00Z_15M_en_2026-06-10T17_18_31Z.csv",
]

OUTPUT_FILE = "solar_merged_15min.csv"


def normalize_dst_hour(value):
    value = str(value)

    value = re.sub(r"(\d{4}-\d{2}-\d{2}) 2A:", r"\1 02:", value)
    value = re.sub(r"(\d{4}-\d{2}-\d{2}) 2B:", r"\1 02:", value)

    return value


def dst_order(value):
    value = str(value)

    if " 2A:" in value:
        return 0

    if " 2B:" in value:
        return 1

    return 0


def load_solar_file(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    df = df.rename(
        columns={
            "Time from [CET/CEST]": "time_from",
            "Time to [CET/CEST]": "time_to",
            "Solar [MW]": "solar_MW",
        }
    )

    df = df[
        [
            "time_from",
            "time_to",
            "solar_MW",
        ]
    ]

    df["solar_MW"] = pd.to_numeric(
        df["solar_MW"],
        errors="coerce"
    )

    return df


def main():
    dataframes = []

    for file in INPUT_FILES:
        path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        dataframes.append(
            load_solar_file(path)
        )

    merged = pd.concat(
        dataframes,
        ignore_index=True
    )

    merged = merged.drop_duplicates(
        subset=["time_from", "time_to"],
        keep="first"
    )

    merged["sort_time"] = pd.to_datetime(
        merged["time_from"].apply(normalize_dst_hour),
        errors="coerce"
    )

    merged["dst_order"] = merged["time_from"].apply(dst_order)

    merged = merged.sort_values(
        ["sort_time", "dst_order", "time_from"]
    )

    merged = merged.drop(
        columns=[
            "sort_time",
            "dst_order",
        ]
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"saved merged solar data to {OUTPUT_FILE}")
    print(f"rows: {len(merged)}")
    print(merged.head())


if __name__ == "__main__":
    main()
"""Fetch KNMI 10-minute observations for De Bilt, 2014-2017.

The 10-minute collection uses different parameter codes from the hourly one,
and radiation arrives as a 10-minute mean in W/m^2 rather than an hourly total
in J/cm^2, so this is a separate script with its own mapping and its own output
file. The hourly file the modelling dataset is built from is never touched.

Requests go one month at a time: four years of 10-minute values in a single
request is far larger than one call should carry.

Timestamps are stored exactly as KNMI labels them. Relabelling to the PV
convention belongs in the merge step, as it does for the hourly data.

Run with KNMI_API_KEY set in the environment:

    python utrecht/knmi_10min.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UTRECHT_DIR = Path(__file__).resolve().parent
HOURLY_PATH = UTRECHT_DIR / "knmi_data" / "knmi_hourly_2014_2017.csv"
OUTPUT_PATH = UTRECHT_DIR / "knmi_data" / "knmi_10min_2014_2017.csv"

COLLECTION = "10-minute-in-situ-meteorological-observations"
BASE_URL = f"https://api.dataplatform.knmi.nl/edr/v1/collections/{COLLECTION}"
STATION_ID = "0-20000-0-06260"

START = pd.Timestamp("2014-01-01")
END = pd.Timestamp("2018-01-01")
STEP = pd.Timedelta(minutes=10)

# 10-minute code -> column name, reusing the hourly dataset's names where the
# quantity is the same. The minimum and maximum irradiance within each interval
# are kept because intra-interval variability is exactly what hourly data
# cannot show.
#
# Two codes carry total cloud cover, n and nc, and the metadata describes them
# identically. nc is the one the hourly N is built from: it agrees with the
# hourly value in about 94% of hours, against about 68% for n, so only nc is
# fetched. The remaining disagreement is mostly one okta, shrinks year on year,
# and is plausibly the validation applied to the hourly collection but not to
# this one.
#
# Columns follow the hourly file's order, with the two radiation columns that
# have no hourly counterpart appended at the end.
PARAMETERS = {
    "dd": "wind_direction_deg",
    "ff": "wind_speed_ms",
    "nc": "cloud_cover_okta",
    "pp": "pressure_hpa",
    "ss": "sunshine_hours",
    "ta": "temperature_C",
    "rh": "humidity_percent",
    "qg": "solar_radiation_Wm2",
    "qgn": "solar_radiation_min_Wm2",
    "qgx": "solar_radiation_max_Wm2",
}


def utc_label(timestamp):
    return f"{timestamp:%Y-%m-%dT%H:%M:%SZ}"


def fetch_window(start, end, headers):
    params = {
        "datetime": f"{utc_label(start)}/{utc_label(end)}",
        "parameter-name": ",".join(PARAMETERS)
    }
    response = requests.get(
        f"{BASE_URL}/locations/{STATION_ID}",
        headers=headers,
        params=params,
        timeout=300
    )

    if not response.ok:
        raise RuntimeError(
            f"{start:%Y-%m}: HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    coverage = response.json()["coverages"][0]
    frame = pd.DataFrame(
        {"time": coverage["domain"]["axes"]["t"]["values"]}
    )

    for code in PARAMETERS:
        values = coverage["ranges"].get(code, {}).get("values")
        frame[code] = values if values is not None else np.nan

    return frame


def fetch_all(headers):
    frames = []

    for month_start in pd.date_range(START, END, freq="MS", inclusive="left"):
        # The API range is inclusive at both ends, so stopping one step short
        # of the next month keeps boundary timestamps from being fetched twice.
        month_end = month_start + pd.offsets.MonthBegin(1) - STEP
        print(f"fetching {month_start:%Y-%m}")
        frames.append(fetch_window(month_start, month_end, headers))

    data = pd.concat(frames, ignore_index=True)
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_localize(None)
    data = data.drop_duplicates("time").sort_values("time")
    data = data.rename(columns=PARAMETERS)

    # KNMI reports sunshine in minutes per interval. Hours keep the unit the
    # hourly file uses, so six 10-minute values sum to the hourly figure.
    data["sunshine_hours"] = data["sunshine_hours"] / 60

    return data


def report_gaps(data):
    """Summarise timestamps the station did not report.

    The fetch itself leaves no holes, so any missing stamp is a genuine gap in
    the station record. They are listed because a long daytime gap, forward
    filled later, would feed the model stale weather for hours.
    """
    expected = pd.date_range(START, END - STEP, freq=STEP)
    missing = pd.Series(expected.difference(data["time"]))

    if missing.empty:
        print("missing timestamps: none")
        return

    runs = (missing.diff() != STEP).cumsum()
    gaps = missing.groupby(runs.to_numpy()).agg(["first", "size"])
    longest = gaps.loc[gaps["size"].idxmax()]
    print(
        f"missing timestamps: {len(missing)} in {len(gaps)} station gaps; "
        f"longest {int(longest['size']) * 10} min from {longest['first']}"
    )


def mean_abs_diff(a, b):
    both = pd.concat([a, b], axis=1).dropna()
    return float((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean())


def match_rate(a, b):
    both = pd.concat([a, b], axis=1).dropna()
    return float((both.iloc[:, 0] == both.iloc[:, 1]).mean())


def compare_with_hourly(ten_min):
    """Check the 10-minute data against the hourly file it should agree with.

    Radiation settles the labelling convention. The hourly value is the mean
    irradiance over the hour ending at its label, so if 10-minute labels also
    mark interval ends, averaging the six values labelled T-50min to T must
    reproduce it. Under start labels the matching window would be T-60min to
    T-10min instead. Whichever window agrees shows how the 10-minute data is
    labelled, which the merge needs to know before relabelling it.

    Cloud cover and temperature confirm the code mapping: the hourly file holds
    the reading taken at its label time, so the matching 10-minute code should
    agree at the same timestamp. Cloud cover agrees in about 94% of hours rather
    than all of them, since only the hourly collection is validated.
    """
    hourly = pd.read_csv(HOURLY_PATH, parse_dates=["time"]).set_index("time")
    ten = ten_min.set_index("time")
    daylight = hourly["solar_radiation_Wm2"] > 0

    irradiance = ten["solar_radiation_Wm2"]
    end_window = irradiance.rolling("60min").mean()
    start_window = end_window.shift(freq=STEP)
    reference = hourly.loc[daylight, "solar_radiation_Wm2"]

    end_error = mean_abs_diff(reference, end_window.reindex(reference.index))
    start_error = mean_abs_diff(
        reference,
        start_window.reindex(reference.index)
    )

    print("\nagreement with the hourly file (daylight hours):")
    print(f"  radiation, end-label window   : mean |diff| {end_error:6.2f} W/m2")
    print(f"  radiation, start-label window : mean |diff| {start_error:6.2f} W/m2")
    print(
        "  -> 10-minute labels mark interval "
        f"{'ENDS' if end_error < start_error else 'STARTS'}"
    )

    sunshine = ten["sunshine_hours"].rolling("60min").sum()
    sunshine_error = mean_abs_diff(
        hourly.loc[daylight, "sunshine_hours"],
        sunshine.reindex(reference.index)
    )
    print(f"  sunshine, summed over the hour: mean |diff| {sunshine_error:.3f} h")

    rate = match_rate(
        hourly["cloud_cover_okta"],
        ten["cloud_cover_okta"].reindex(hourly.index)
    )
    print(f"  cloud cover, same label       : equal in {rate:.1%} of hours")

    temperature_error = mean_abs_diff(
        hourly["temperature_C"],
        ten["temperature_C"].reindex(hourly.index)
    )
    print(f"  temperature, same label       : mean |diff| {temperature_error:.2f} C")


def main():
    api_key = os.environ.get("KNMI_API_KEY")

    if not api_key:
        raise RuntimeError("Set the KNMI_API_KEY environment variable")

    data = fetch_all({"Authorization": api_key})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    expected = int((END - START) / STEP)
    print(f"\nrows: {len(data)} (expected {expected})")
    report_gaps(data)
    print(f"saved {OUTPUT_PATH}")

    compare_with_hourly(data)


if __name__ == "__main__":
    main()

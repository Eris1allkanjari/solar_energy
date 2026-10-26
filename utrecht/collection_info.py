"""Inspect a KNMI EDR collection before fetching from it.

Prints the collection's parameter codes, its temporal extent, whether the De
Bilt station is one of its locations, and the result of a one-day test request.
Parameter codes differ between collections, so a fetch that works on the hourly
collection can fail on the 10-minute one; this shows the codes to use.

Run with KNMI_API_KEY set in the environment:

    python utrecht/collection_info.py 10-minute-in-situ-meteorological-observations
"""

import os
import sys

import requests

BASE_URL = "https://api.dataplatform.knmi.nl/edr/v1/collections"
DEFAULT_COLLECTION = "10-minute-in-situ-meteorological-observations"
STATION_ID = "0-20000-0-06260"
TEST_WINDOW = "2014-06-01T00:00:00Z/2014-06-02T00:00:00Z"


def get_json(url, headers, params=None):
    response = requests.get(url, headers=headers, params=params, timeout=120)

    if not response.ok:
        # raise_for_status() discards the body, and the body is where the
        # server explains what was wrong with the request.
        print(f"  HTTP {response.status_code}: {response.text[:1500]}")
        return None

    return response.json()


def unit_of(info):
    unit = info.get("unit", {})
    symbol = unit.get("symbol", "")

    if isinstance(symbol, dict):
        symbol = symbol.get("value", "")

    return symbol or unit.get("label", "")


def main():
    api_key = os.environ.get("KNMI_API_KEY")

    if not api_key:
        raise RuntimeError("Set the KNMI_API_KEY environment variable")

    collection = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_COLLECTION
    headers = {"Authorization": api_key}
    collection_url = f"{BASE_URL}/{collection}"

    print(f"collection: {collection}\n")
    metadata = get_json(collection_url, headers)

    if metadata is None:
        return

    temporal = metadata.get("extent", {}).get("temporal", {})
    print(f"temporal extent: {temporal.get('interval')}\n")

    print("parameters:")

    for code, info in sorted(metadata.get("parameter_names", {}).items()):
        label = (
            info.get("observedProperty", {}).get("label")
            or info.get("description", "")
        )
        print(f"  {code:12s} {label} [{unit_of(info)}]")

    print()
    locations = get_json(f"{collection_url}/locations", headers)

    if locations is not None:
        station_names = {
            feature["id"]: feature.get("properties", {}).get("name", "")
            for feature in locations.get("features", [])
        }
        name = station_names.get(STATION_ID)
        print(
            f"De Bilt ({STATION_ID}) in collection: "
            f"{'yes, ' + name if name else 'NO'}"
        )

    print(f"\ntest request, one day at De Bilt: {TEST_WINDOW}")
    sample = get_json(
        f"{collection_url}/locations/{STATION_ID}",
        headers,
        params={"datetime": TEST_WINDOW}
    )

    if sample is not None:
        coverage = sample["coverages"][0]
        times = coverage["domain"]["axes"]["t"]["values"]
        print(
            f"  timestamps returned: {len(times)} "
            "(145 for 10-minute data: the range includes both ends)"
        )
        print(f"  first/last: {times[0]} / {times[-1]}")
        print(f"  parameters returned: {', '.join(sorted(coverage['ranges']))}")


if __name__ == "__main__":
    main()

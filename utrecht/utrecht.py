import os
from pathlib import Path

import requests
import pandas as pd

API_KEY = os.environ.get("KNMI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the KNMI_API_KEY environment variable")

UTRECHT_DIR = Path(__file__).resolve().parent

collection = "hourly-in-situ-meteorological-observations-validated"
base_url = f"https://api.dataplatform.knmi.nl/edr/v1/collections/{collection}"

headers = {"Authorization": API_KEY}

station_id = "0-20000-0-06260"

# T,TD,FF,DD,P,Q,N,SQ,U

#T - Past 1 minute air temperature at 1.50 meters, in degrees Celsius
# SQ - Hourly sunshine duration, calculated from global solar radiation, in hours
# FF - Past 10 minute mean wind speed, representative for 10 meters, in meters per second
# DD - Past 10 minute mean wind direction, representative for 10 meters, in degrees; 360=north; 90=east; 180=south; 270=west; 0=calm; 990=variable
# P - Past 1 minute mean air pressure at sea level
# Q - Hourly global solar radiation, in joules per square centimeter
# N - Past 10 minute cloud cover, in okta; 9=sky invisible
# U - Past 1 minute relative atmospheric humidity at the time of observation, at 1.50 meters, as percentage

params = {
    "datetime": "2014-01-01T00:00:00Z/2017-12-31T23:00:00Z",
    "parameter-name": "T,SQ,FF,DD,P,Q,N,U",
}

response = requests.get(
    f"{base_url}/locations/{station_id}",
    headers=headers,
    params=params
)

response.raise_for_status()

data = response.json()

coverage = data["coverages"][0]

times = coverage["domain"]["axes"]["t"]["values"]
ranges = coverage["ranges"]

records = []

for i, t in enumerate(times):

    row = {"time": t}

    for param in ranges:
        row[param] = ranges[param]["values"][i]

    records.append(row)

df = pd.DataFrame(records)
df["time"] = pd.to_datetime(df["time"])

weather_df = df.rename(columns={
    "T": "temperature_C",
    "SQ": "sunshine_hours",
    "FF": "wind_speed_ms",
    "DD": "wind_direction_deg",
    "P": "pressure_hpa",
    "Q": "solar_radiation_Jcm2",
    "N": "cloud_cover_okta",
    "U": "humidity_percent"
})

# Convert J/cm² per hour to W/m²
weather_df["solar_radiation_Wm2"] = weather_df["solar_radiation_Jcm2"] * 2.77778
weather_df.drop(columns="solar_radiation_Jcm2", inplace=True)

weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)

output_path = UTRECHT_DIR / "knmi_data" / "knmi_hourly_2014_2017.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
weather_df.to_csv(output_path, index=False)

print(weather_df.head())

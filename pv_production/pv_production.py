import requests
import pandas as pd

lat = 47.949
lon = 16.843

params = {
    "lat": lat,
    "lon": lon,
    "startyear": 2015,
    "endyear": 2023,
    "raddatabase": "PVGIS-SARAH3",
    "pvcalculation": 1,
    "peakpower": 1000,        # 1000 kW system (1 MW plant)
    "loss": 14,               # system losses %
    "trackingtype": 0,        # fixed panels
    "angle": 30,              # tilt
    "aspect": 0,              # south-facing
    "outputformat": "json"
}

url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

response = requests.get(url, params=params)
data = response.json()

df_pv = pd.DataFrame(data["outputs"]["hourly"])

df_pv["time"] = pd.to_datetime(
    df_pv["time"],
    format="%Y%m%d:%H%M"
)
df_pv = df_pv.set_index("time")

df_pv.to_csv("pv_hourly_2015_2024.csv")

print(df_pv.head())
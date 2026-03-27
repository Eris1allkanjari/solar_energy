import pandas as pd

pv_df = pd.read_csv("pv_data/filtered_pv_power_measurements_ac.csv")
pv_df = pv_df.rename(columns={"DateTime": "time"})

pv_df["time"] = pd.to_datetime(pv_df["time"], utc=True)
pv_df = pv_df.set_index("time")

# Convert power to energy/hour
pv_energy = pv_df / 60

# Aggregate to hourly production and convert to kWh
pv_hourly = pv_energy.resample("1h").sum()
pv_hourly = pv_hourly / 1000

pv_hourly["pv_total_kWh"] = pv_hourly.sum(axis=1)

pv_hourly.to_csv("pv_data/pv_hourly_2014_2017.csv")
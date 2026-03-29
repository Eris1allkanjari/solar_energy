import pandas as pd
import numpy as np


# Loading data
solar = pd.read_csv("unisolar/Solar_Energy_Generation.csv")
weather = pd.read_csv("unisolar/Weather_Data_reordered_all.csv")
site = pd.read_csv("unisolar/Solar_Site_Details.csv")

# timestamp processing
solar["Timestamp"] = pd.to_datetime(solar["Timestamp"])
weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])

# merge solar + weather
df = solar.merge(
    weather,
    on=["CampusKey", "Timestamp"],
    how="left"
)

# merge site metadata
df = df.merge(
    site,
    on=["CampusKey", "SiteKey"],
    how="left"
)

# handle missing values

# solar: nan = no sun → set to 0
df["SolarGeneration"] = df["SolarGeneration"].fillna(0)

# sort before filling
df = df.sort_values(["CampusKey", "Timestamp"])


#  aggregate all sites
# df = df.groupby("Timestamp").agg({
#     "SolarGeneration": "sum",
#     "AirTemperature": "mean",
#     "RelativeHumidity": "mean",
#     "WindSpeed": "mean",
#     "DewPointTemperature": "mean"
# }).reset_index()

# resample to hourly
df = df.set_index("Timestamp")

df_hourly = df.resample("h").agg({
    "SolarGeneration": "sum",
    "AirTemperature": "mean",
    "RelativeHumidity": "mean",
    "WindSpeed": "mean",
    "DewPointTemperature": "mean"
})

df_hourly = df_hourly.reset_index()

# feature engineering
df_hourly["hour"] = df_hourly["Timestamp"].dt.hour
df_hourly["day"] = df_hourly["Timestamp"].dt.day
df_hourly["month"] = df_hourly["Timestamp"].dt.month
df_hourly["dayofyear"] = df_hourly["Timestamp"].dt.dayofyear

# handle missing after resampling
df_hourly = df_hourly.interpolate()

df_hourly = df_hourly.rename(columns={
    "Timestamp": "time",
    "SolarGeneration": "solar_generation_kWh",
    "AirTemperature": "air_temperature_C",
    "RelativeHumidity": "relative_humidity_pct",
    "WindSpeed": "wind_speed_mps",
    "DewPointTemperature": "dew_point_temperature_C"
})

# save final dataset
df_hourly.to_csv("processed_data/unisolar_pv_data.csv", index=False)


import pandas as pd
import numpy as np


# Loading data
solar = pd.read_csv("unisolar/Solar_Energy_Generation.csv")
weather = pd.read_csv("unisolar/Weather_Data_reordered_all.csv")
site = pd.read_csv("unisolar/Solar_Site_Details.csv")

solar["Timestamp"] = pd.to_datetime(solar["Timestamp"])
weather["Timestamp"] = pd.to_datetime(weather["Timestamp"])

# Merging Solar with Weather data
df = solar.merge(
    weather,
    on=["CampusKey", "Timestamp"],
    how="left"
)

#  Merging Site Metadata
df = df.merge(
    site,
    on=["CampusKey", "SiteKey"],
    how="left"
)

# Solar generation is NaN at night → replace with 0
df["SolarGeneration"] = df["SolarGeneration"].fillna(0)

# # Optional: forward fill weather (if needed)
# weather_cols = [
#     "ApparentTemperature",
#     "AirTemperature",
#     "DewPointTemperature",
#     "RelativeHumidity",
#     "WindSpeed",
#     "WindDirection"
# ]
#
# df[weather_cols] = df[weather_cols].fillna(method="ffill")

# Feature Engineering
df["hour"] = df["Timestamp"].dt.hour
df["day"] = df["Timestamp"].dt.day
df["month"] = df["Timestamp"].dt.month
df["dayofyear"] = df["Timestamp"].dt.dayofyear


# Sort and clean
df = df.sort_values(by=["SiteKey", "Timestamp"]).reset_index(drop=True)


df.to_csv("processed_data/unisolar_solar_weather.csv", index=False)


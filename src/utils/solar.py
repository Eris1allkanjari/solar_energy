"""Solar geometry used to split forecasts into day and night periods.

Utrecht daylight ranges from roughly 7.75 hours in December to 16.5 hours in
June, so a fixed clock window cannot separate day from night. The solar
elevation angle is computed instead, following the NOAA solar position
equations. The split depends only on timestamp and site coordinates, never on
the observed or predicted production, so it cannot leak target information into
the evaluation.
"""

import numpy as np
import pandas as pd


def solar_elevation(index, latitude, longitude):
    """Solar elevation angle in degrees for a UTC DatetimeIndex."""
    index = pd.DatetimeIndex(index)

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    day_of_year = index.dayofyear.to_numpy()
    utc_hour = (
        index.hour.to_numpy()
        + index.minute.to_numpy() / 60.0
    )

    # fractional year, radians
    gamma = (
        2 * np.pi / 365.0
        * (day_of_year - 1 + (utc_hour - 12) / 24.0)
    )

    # equation of time, minutes
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    # solar declination, radians
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.001480 * np.sin(3 * gamma)
    )

    true_solar_time = utc_hour * 60.0 + equation_of_time + 4.0 * longitude
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    latitude_radians = np.deg2rad(latitude)
    sine_elevation = (
        np.sin(latitude_radians) * np.sin(declination)
        + np.cos(latitude_radians)
        * np.cos(declination)
        * np.cos(hour_angle)
    )

    return np.rad2deg(np.arcsin(np.clip(sine_elevation, -1.0, 1.0)))


def is_daylight(index, latitude, longitude, elevation_threshold=0.0):
    """Boolean mask marking hours where the sun is above the threshold."""
    return solar_elevation(
        index=index,
        latitude=latitude,
        longitude=longitude
    ) > elevation_threshold

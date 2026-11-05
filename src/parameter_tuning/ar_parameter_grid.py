from src.configs.dataset_selection import hours_to_steps
from src.configs.evaluation import MIN_SEASONAL_WINDOW, SEASONAL_PERIOD
from src.data.preprocessing import SOLAR_ELEVATION_FEATURE


# Rolling fit windows of 7, 14, 30, 60 and 90 days, in hours, converted to rows
# of the active dataset. The 90-day window was added because the 10-minute
# models preferred the longest window of the original grid.
AR_WINDOW_HOURS = [
    24 * 7,
    24 * 14,
    24 * 30,
    24 * 60,
    24 * 90
]
AR_WINDOW_LENGTHS = [hours_to_steps(hours) for hours in AR_WINDOW_HOURS]

AR_WINDOW_LENGTHS_BY_MODEL = {
    "arima": AR_WINDOW_LENGTHS,
    "arimax": AR_WINDOW_LENGTHS,
    "sarima": [
        window_length
        for window_length in AR_WINDOW_LENGTHS
        if window_length >= MIN_SEASONAL_WINDOW
    ],
    "sarimax": [
        window_length
        for window_length in AR_WINDOW_LENGTHS
        if window_length >= MIN_SEASONAL_WINDOW
    ]
}

# Non-seasonal orders. (3, 0, 1) and (2, 0, 2) extend the original grid beyond
# p = 2 and q = 1, where the hourly models found their best fit.
NON_SEASONAL_ORDERS = [
    (1, 0, 1),
    (2, 0, 1),
    (2, 1, 1),
    (3, 0, 1),
    (2, 0, 2)
]
SEASONAL_BASE_ORDERS = [
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (2, 0, 2)
]
SEASONAL_ORDERS = [
    (1, 0, 1, SEASONAL_PERIOD),
    (1, 1, 1, SEASONAL_PERIOD)
]

RADIATION = "solar_radiation_Wm2"
CLOUD = "cloud_cover_okta"
TEMPERATURE = "temperature_C"
HUMIDITY = "humidity_percent"

# Exogenous input sets. Measured weather enters lagged by one interval, so on
# its own it adds little beyond the previous production value. Humidity is the
# weather variable that still carries some information once that value is
# known. The solar elevation of the interval being forecast is computed from
# the timestamp, so it is known in advance and enters unlagged. Radiation on its
# own was added after removing cloud cover improved the selected SARIMAX in the
# feature significance study.
EXOG_FEATURE_SETS = [
    [RADIATION],
    [RADIATION, CLOUD],
    [RADIATION, CLOUD, TEMPERATURE],
    [RADIATION, HUMIDITY],
    [RADIATION, CLOUD, HUMIDITY],
    [RADIATION, SOLAR_ELEVATION_FEATURE],
    [RADIATION, CLOUD, SOLAR_ELEVATION_FEATURE]
]

AR_HYPERPARAMETER_GRIDS = {
    "arima": {
        "order": NON_SEASONAL_ORDERS,
        "max_iter": [200],
        "enforce_stationarity": [True],
        "enforce_invertibility": [True]
    },

    "sarima": {
        "order": SEASONAL_BASE_ORDERS,
        "seasonal_order": SEASONAL_ORDERS,
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    },

    "arimax": {
        "order": NON_SEASONAL_ORDERS,
        "exog_features": EXOG_FEATURE_SETS,
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    },

    "sarimax": {
        "order": SEASONAL_BASE_ORDERS,
        "seasonal_order": SEASONAL_ORDERS,
        "exog_features": EXOG_FEATURE_SETS,
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    }
}

from src.configs.dataset_selection import hours_to_steps
from src.configs.evaluation import MIN_SEASONAL_WINDOW, SEASONAL_PERIOD


# Rolling fit windows of 7, 14, 30 and 60 days, in hours, converted to rows of
# the active dataset.
AR_WINDOW_HOURS = [
    24 * 7,
    24 * 14,
    24 * 30,
    24 * 60
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

AR_HYPERPARAMETER_GRIDS = {
    "arima": {
        "order": [
            (1, 0, 1),
            (2, 0, 1),
            (2, 1, 1)
        ],
        "max_iter": [200],
        "enforce_stationarity": [True],
        "enforce_invertibility": [True]
    },

    "sarima": {
        "order": [
            (1, 0, 1),
            (2, 0, 1)
        ],
        "seasonal_order": [
            (1, 0, 1, SEASONAL_PERIOD),
            (1, 1, 1, SEASONAL_PERIOD)
        ],
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    },

    "arimax": {
        "order": [
            (1, 0, 1),
            (2, 0, 1),
            (2, 1, 1)
        ],
        "exog_features": [
            [
                "solar_radiation_Wm2",
                "cloud_cover_okta"
            ],
            [
                "solar_radiation_Wm2",
                "cloud_cover_okta",
                "temperature_C"
            ]
        ],
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    },

    "sarimax": {
        "order": [
            (1, 0, 1),
            (2, 0, 1)
        ],
        "seasonal_order": [
            (1, 0, 1, SEASONAL_PERIOD),
            (1, 1, 1, SEASONAL_PERIOD)
        ],
        "exog_features": [
            [
                "solar_radiation_Wm2",
                "cloud_cover_okta"
            ],
            [
                "solar_radiation_Wm2",
                "cloud_cover_okta",
                "temperature_C"
            ]
        ],
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    }
}

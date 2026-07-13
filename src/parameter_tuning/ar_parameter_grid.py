AR_WINDOW_LENGTHS = [
    24 * 7,
    24 * 14,
    24 * 30,
    24 * 60
]

AR_VALIDATION_STEPS = 168
AR_TEST_STEPS = 1000

AR_HYPERPARAMETER_GRIDS = {
    "arima": {
        "order": [
            (1, 0, 1),
            (2, 0, 1),
            (2, 1, 1)
        ],
        "max_iter": [50],
        "enforce_stationarity": [True],
        "enforce_invertibility": [True]
    },

    "sarima": {
        "order": [
            (1, 0, 1),
            (2, 0, 1)
        ],
        "seasonal_order": [
            (1, 0, 1, 24),
            (1, 1, 1, 24)
        ],
        "max_iter": [50],
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
        "max_iter": [50],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    },

    "sarimax": {
        "order": [
            (1, 0, 1),
            (2, 0, 1)
        ],
        "seasonal_order": [
            (1, 0, 1, 24),
            (1, 1, 1, 24)
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
        "max_iter": [50],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    }
}

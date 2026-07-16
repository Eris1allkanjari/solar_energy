from src.configs.evaluation import MIN_SEASONAL_WINDOW


AR_WINDOW_LENGTHS = [
    24 * 7,
    24 * 14,
    24 * 30,
    24 * 60
]

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
            (1, 0, 1, 24),
            (1, 1, 1, 24)
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
        "max_iter": [200],
        "enforce_stationarity": [False],
        "enforce_invertibility": [False]
    }
}

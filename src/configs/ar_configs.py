class StatisticalConfig:
    # model fitting
    METHOD = "lbfgs"

    MAX_ITER = 200
    ENFORCE_STATIONARITY = True
    ENFORCE_INVERTIBILITY = True

class ARIMAConfig(StatisticalConfig):
    ORDER = (2, 0, 1)

class ARIMAXConfig(StatisticalConfig):
    ORDER = (2, 0, 1)
    EXOG_FEATURES = [
        "solar_radiation_Wm2",
        "cloud_cover_okta"
    ]

class SARIMAConfig(StatisticalConfig):
    ORDER = (1, 0, 1)
    SEASONAL_ORDER = (1, 0, 1, 24)

class SARIMAXConfig(StatisticalConfig):

    ORDER = (1, 0, 1)
    SEASONAL_ORDER = (1, 0, 1, 24)
    EXOG_FEATURES = [
        "solar_radiation_Wm2",
        "cloud_cover_okta"
    ]
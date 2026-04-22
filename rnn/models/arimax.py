from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_arimax(train_data, exog, config):

    model = SARIMAX(
        train_data,
        exog=exog,
        order=config.ORDER,
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=config.ENFORCE_STATIONARITY,
        enforce_invertibility=config.ENFORCE_INVERTIBILITY
    )

    return model
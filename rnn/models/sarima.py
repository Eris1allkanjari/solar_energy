from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_sarima(train_data, config):

    model = SARIMAX(
        train_data,
        order=config.ORDER,
        seasonal_order=config.SEASONAL_ORDER,
        enforce_stationarity=config.ENFORCE_STATIONARITY,
        enforce_invertibility=config.ENFORCE_INVERTIBILITY
    )

    return model
from src.configs.ar_configs import (
    ARIMAConfig,
    SARIMAConfig,
    ARIMAXConfig,
    SARIMAXConfig
)

from src.models.ar.arima import build_arima
from src.models.ar.arimax import build_arimax
from src.models.ar.sarima import build_sarima
from src.models.ar.sarimax import build_sarimax

from src.forecasting.direct_forecast import direct_forecast
from src.forecasting.rolling_forecast import rolling_forecast

from src.training.evaluation import evaluate
from src.training.statistical_trainer import fit_statistical_model

experiments = [

    # {
    #     "name": "arima",
    #
    #     "builder": build_arima,
    #
    #     "config": ARIMAConfig(),
    #
    #     "forecasting": "direct",
    #
    #     "use_exog": False
    # },
    #
    # {
    #     "name": "sarima",
    #
    #     "builder": build_sarima,
    #
    #     "config": SARIMAConfig(),
    #
    #     "forecasting": "direct",
    #
    #     "use_exog": False
    # },
    #
    # {
    #     "name": "arimax",
    #
    #     "builder": build_arimax,
    #
    #     "config": ARIMAXConfig(),
    #
    #     "forecasting": "direct",
    #
    #     "use_exog": True
    # },

    {
        "name": "sarimax",

        "builder": build_sarimax,

        "config": SARIMAXConfig(),

        "forecasting": "rolling",

        "use_exog": True
    }
]


def run_experiment(
    build_model,
    model_name,

    y_train,
    y_test,

    config,

    exog_train=None,
    exog_test=None,

    forecasting_strategy="direct"
):

    # direct forecasting

    if forecasting_strategy == "direct":

        if exog_train is not None:

            model = build_model(
                y_train,
                exog_train,
                config
            )

        else:

            model = build_model(
                y_train,
                config
            )

        results = fit_statistical_model(
            model,
            config
        )

        predictions = direct_forecast(
            results,
            len(y_test),
            exog_test
        )

    # rolling forecasting

    elif forecasting_strategy == "rolling":

        predictions = rolling_forecast(

            build_model_fn=build_model,

            train=y_train,
            test=y_test,

            config=config,

            exog_train=exog_train,
            exog_test=exog_test
        )

    else:

        raise ValueError(
            f"invalid forecasting strategy: {forecasting_strategy}"
        )

    # evaluation

    mae, rmse, mape, smape = evaluate(
        y_test,
        predictions
    )

    return {

        "model": model_name,

        "forecasting": forecasting_strategy,

        "order": str(config.ORDER),

        "seasonal_order": (
            str(config.SEASONAL_ORDER)
            if hasattr(config, "SEASONAL_ORDER")
            else None
        ),

        "mae": mae,

        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }

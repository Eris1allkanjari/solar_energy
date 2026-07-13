import numpy as np
import pandas as pd
from copy import copy


def get_first_forecast_value(forecast):
    if hasattr(forecast, "iloc"):
        return forecast.iloc[0]

    return np.asarray(forecast).reshape(-1)[0]


def clone_relaxed_config(config):
    relaxed_config = copy(config)

    relaxed_config.ENFORCE_STATIONARITY = False
    relaxed_config.ENFORCE_INVERTIBILITY = False

    return relaxed_config


def initialize_diffuse_if_available(model):
    if hasattr(model, "initialize_approximate_diffuse"):
        model.initialize_approximate_diffuse()

    return model


def fit_statistical_model(model, config):
    return model.fit(
        method=config.METHOD,
        maxiter=config.MAX_ITER,
        disp=False
    )


def fallback_prediction(history):
    return history[-1]


def build_statistical_model(
    build_model_fn,
    history_window,
    config,
    exog_hist=None
):
    if exog_hist is not None:
        return build_model_fn(
            history_window,
            exog_hist,
            config
        )

    return build_model_fn(
        history_window,
        config
    )


def should_refit(step, fitted, refit_interval):
    if fitted is None:
        return True

    if refit_interval is None or refit_interval == 0:
        return False

    return step % refit_interval == 0


def append_observation(
    fitted,
    observation,
    exog_row=None
):
    append_kwargs = {
        "refit": False
    }

    if exog_row is not None:
        append_kwargs["exog"] = exog_row.to_numpy()

    return fitted.append(
        np.asarray([observation]),
        **append_kwargs
    )


def rolling_forecast(
    build_model_fn,
    train,
    test,
    config,
    exog_train=None,
    exog_test=None,
    refit_interval=1
):

    if refit_interval is not None and refit_interval < 0:
        raise ValueError(
            "refit_interval must be None, 0, or a positive integer"
        )

    history = list(train)
    predictions = []
    fitted = None

    max_history = getattr(config, "MAX_HISTORY", None)

    if exog_train is not None and exog_test is not None:
        exog_full = pd.concat(
            [exog_train, exog_test],
            axis=0
        )
    else:
        exog_full = None

    for t in range(len(test)):

        if t % 100 == 0:
            print(f"step {t}/{len(test)}")

        if max_history is not None:
            history_window = history[-max_history:]
        else:
            history_window = history

        exog_hist = None

        if exog_full is not None:

            end_idx = len(history)
            start_idx = end_idx - len(history_window)

            exog_hist = exog_full.iloc[start_idx:end_idx]

        if should_refit(
            step=t,
            fitted=fitted,
            refit_interval=refit_interval
        ):
            try:
                model = build_statistical_model(
                    build_model_fn=build_model_fn,
                    history_window=history_window,
                    config=config,
                    exog_hist=exog_hist
                )

                fitted = fit_statistical_model(
                    model,
                    config
                )

            except Exception as first_error:
                print(
                    "rolling fit failed at "
                    f"step {t}; retrying with relaxed constraints: "
                    f"{first_error}"
                )

                try:
                    relaxed_config = clone_relaxed_config(
                        config
                    )

                    model = build_statistical_model(
                        build_model_fn=build_model_fn,
                        history_window=history_window,
                        config=relaxed_config,
                        exog_hist=exog_hist
                    )

                    model = initialize_diffuse_if_available(
                        model
                    )

                    fitted = fit_statistical_model(
                        model,
                        relaxed_config
                    )

                except Exception as second_error:
                    print(
                        "relaxed rolling fit also failed at "
                        f"step {t}; using persistence fallback: "
                        f"{second_error}"
                    )

                    fitted = None
                    pred = fallback_prediction(
                        history
                    )

                    predictions.append(pred)
                    history.append(test.iloc[t])

                    continue

        try:
            if exog_test is not None:

                forecast = fitted.forecast(
                    steps=1,
                    exog=exog_test.iloc[[t]]
                )

                pred = get_first_forecast_value(
                    forecast
                )

            else:

                forecast = fitted.forecast(
                    steps=1
                )

                pred = get_first_forecast_value(
                    forecast
                )

        except Exception as forecast_error:
            print(
                "rolling forecast failed at "
                f"step {t}; using persistence fallback: "
                f"{forecast_error}"
            )

            pred = fallback_prediction(
                history
            )

        predictions.append(pred)

        observation = test.iloc[t]
        history.append(observation)

        keep_fitted_state = (
            refit_interval is None
            or refit_interval == 0
            or refit_interval > 1
        )

        if keep_fitted_state and fitted is not None:
            exog_row = None

            if exog_test is not None:
                exog_row = exog_test.iloc[[t]]

            try:
                fitted = append_observation(
                    fitted=fitted,
                    observation=observation,
                    exog_row=exog_row
                )

            except Exception as append_error:
                print(
                    "state update failed at "
                    f"step {t}; refitting on the next step: "
                    f"{append_error}"
                )

                fitted = None

    return predictions

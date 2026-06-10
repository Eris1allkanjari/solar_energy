import pandas as pd


def rolling_forecast(
    build_model_fn,
    train,
    test,
    config,
    exog_train=None,
    exog_test=None
):

    history = list(train)
    predictions = []

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

        if exog_full is not None:

            end_idx = len(history)
            start_idx = end_idx - len(history_window)

            exog_hist = exog_full.iloc[start_idx:end_idx]

            model = build_model_fn(
                history_window,
                exog_hist,
                config
            )

        else:

            model = build_model_fn(
                history_window,
                config
            )

        fitted = model.fit(
            method=config.METHOD,
            maxiter=config.MAX_ITER,
            disp=False
        )

        if exog_test is not None:

            forecast = fitted.forecast(
                steps=1,
                exog=exog_test.iloc[[t]]
            )

            pred = forecast.iloc[0]

        else:

            forecast = fitted.forecast(
                steps=1
            )

            pred = forecast.iloc[0]

        predictions.append(pred)

        history.append(test.iloc[t])

    return predictions
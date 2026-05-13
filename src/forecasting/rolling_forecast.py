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

    for t in range(len(test)):

        if exog_train is not None:
            exog_hist = exog_train.iloc[:len(history)]

            model = build_model_fn(
                history,
                exog_hist,
                config
            )

        else:
            model = build_model_fn(
                history,
                config
            )

        fitted = model.fit(
            method=config.METHOD,
            maxiter=config.MAX_ITER,
            disp=False
        )

        if exog_test is not None:
            pred = fitted.forecast(
                steps=1,
                exog=exog_test.iloc[[t]]
            )[0]
        else:
            pred = fitted.forecast(steps=1)[0]

        predictions.append(pred)

        history.append(test.iloc[t])

    return predictions
def direct_forecast(results, steps, exog=None):

    forecast = results.get_forecast(
        steps=steps,
        exog=exog
    )

    return forecast.predicted_mean
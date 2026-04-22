def fit_statistical_model(model, config):

    results = model.fit(
        method=config.METHOD,
        maxiter=config.MAX_ITER,
        disp=False
    )

    return results
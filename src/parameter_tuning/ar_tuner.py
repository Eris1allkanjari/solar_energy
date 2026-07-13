import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.forecasting.rolling_forecast import rolling_forecast
from src.training.evaluation import evaluate


class ARExperimentConfig:
    def __init__(self, params, window_length):
        self.ORDER = params["order"]

        if "seasonal_order" in params:
            self.SEASONAL_ORDER = params["seasonal_order"]

        if "exog_features" in params:
            self.EXOG_FEATURES = params["exog_features"]

        self.METHOD = params.get("method", "lbfgs")
        self.MAX_ITER = params["max_iter"]
        self.ENFORCE_STATIONARITY = params["enforce_stationarity"]
        self.ENFORCE_INVERTIBILITY = params["enforce_invertibility"]
        self.MAX_HISTORY = window_length


def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def prepare_ar_data(df):
    y = df["pv_total_kWh"].clip(lower=0)
    y = y.interpolate().bfill().ffill()

    exog = df.drop(
        columns=["pv_total_kWh"],
        errors="ignore"
    )

    exog = exog.replace(
        [np.inf, -np.inf],
        np.nan
    )

    exog = exog.interpolate().bfill().ffill()

    n_raw = len(y)

    train_end = int(n_raw * 0.65)
    val_end = int(n_raw * 0.80)

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    exog_train = exog.iloc[:train_end]
    exog_val = exog.iloc[train_end:val_end]
    exog_test = exog.iloc[val_end:]

    return (
        y_train,
        y_val,
        y_test,
        exog_train,
        exog_val,
        exog_test
    )


def scale_exog_splits(
    exog_train,
    exog_val,
    exog_test,
    exog_features
):
    scaler = StandardScaler()

    train_subset = exog_train[exog_features]
    val_subset = exog_val[exog_features]
    test_subset = exog_test[exog_features]

    exog_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_subset),
        index=train_subset.index,
        columns=train_subset.columns
    )

    exog_val_scaled = pd.DataFrame(
        scaler.transform(val_subset),
        index=val_subset.index,
        columns=val_subset.columns
    )

    exog_test_scaled = pd.DataFrame(
        scaler.transform(test_subset),
        index=test_subset.index,
        columns=test_subset.columns
    )

    return exog_train_scaled, exog_val_scaled, exog_test_scaled


def scale_exog_train_test(
    exog_train,
    exog_test,
    exog_features
):
    scaler = StandardScaler()

    train_subset = exog_train[exog_features]
    test_subset = exog_test[exog_features]

    exog_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_subset),
        index=train_subset.index,
        columns=train_subset.columns
    )

    exog_test_scaled = pd.DataFrame(
        scaler.transform(test_subset),
        index=test_subset.index,
        columns=test_subset.columns
    )

    return exog_train_scaled, exog_test_scaled


def uses_exog(params):
    return "exog_features" in params


def evaluate_on_validation(
    model_name,
    build_model,
    y_train,
    y_val,
    exog_train,
    exog_val,
    exog_test,
    window_length,
    params,
    validation_steps=None
):
    config = ARExperimentConfig(
        params=params,
        window_length=window_length
    )

    y_val_eval = y_val

    if validation_steps is not None:
        y_val_eval = y_val_eval.iloc[:validation_steps]

    model_exog_train = None
    model_exog_val = None

    if uses_exog(params):
        model_exog_train, model_exog_val, _ = scale_exog_splits(
            exog_train=exog_train,
            exog_val=exog_val,
            exog_test=exog_test,
            exog_features=config.EXOG_FEATURES
        )

        model_exog_val = model_exog_val.loc[
            y_val_eval.index
        ]

    predictions = rolling_forecast(
        build_model_fn=build_model,
        train=y_train,
        test=y_val_eval,
        config=config,
        exog_train=model_exog_train,
        exog_test=model_exog_val
    )

    mae, rmse, mape, smape = evaluate(
        y_val_eval,
        predictions
    )

    return {
        "model": model_name,
        "seq_len": window_length,
        "order": str(config.ORDER),
        "seasonal_order": (
            str(config.SEASONAL_ORDER)
            if hasattr(config, "SEASONAL_ORDER")
            else None
        ),
        "exog_features": (
            ",".join(config.EXOG_FEATURES)
            if hasattr(config, "EXOG_FEATURES")
            else None
        ),
        "max_iter": config.MAX_ITER,
        "enforce_stationarity": config.ENFORCE_STATIONARITY,
        "enforce_invertibility": config.ENFORCE_INVERTIBILITY,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_mape": mape,
        "val_smape": smape
    }


def tune_ar_model(
    model_name,
    build_model,
    df,
    param_grid,
    window_lengths,
    validation_steps=None
):
    y_train, y_val, y_test, exog_train, exog_val, exog_test = prepare_ar_data(
        df
    )

    all_results = []
    best_per_l = []

    for window_length in window_lengths:
        print(
            f"\nsearching {model_name} with L={window_length}"
        )

        best_result_for_l = None

        for params in generate_param_combinations(param_grid):
            try:
                result = evaluate_on_validation(
                    model_name=model_name,
                    build_model=build_model,
                    y_train=y_train,
                    y_val=y_val,
                    exog_train=exog_train,
                    exog_val=exog_val,
                    exog_test=exog_test,
                    window_length=window_length,
                    params=params,
                    validation_steps=validation_steps
                )

            except Exception as e:
                result = {
                    "model": model_name,
                    "seq_len": window_length,
                    "order": str(params.get("order")),
                    "seasonal_order": str(params.get("seasonal_order")),
                    "exog_features": (
                        ",".join(params["exog_features"])
                        if "exog_features" in params
                        else None
                    ),
                    "max_iter": params.get("max_iter"),
                    "enforce_stationarity": params.get(
                        "enforce_stationarity"
                    ),
                    "enforce_invertibility": params.get(
                        "enforce_invertibility"
                    ),
                    "val_mae": np.nan,
                    "val_rmse": np.nan,
                    "val_mape": np.nan,
                    "val_smape": np.nan,
                    "error": str(e)
                }

            all_results.append(result)

            if np.isnan(result["val_mae"]):
                continue

            if (
                best_result_for_l is None
                or result["val_mae"] < best_result_for_l["val_mae"]
            ):
                best_result_for_l = result

        if best_result_for_l is not None:
            best_per_l.append(best_result_for_l)

    return all_results, best_per_l


def select_best_l(best_per_l):
    return min(
        best_per_l,
        key=lambda result: result["val_mae"]
    )


def parse_tuple(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, tuple):
        return value

    return tuple(
        int(part.strip())
        for part in str(value).strip("()").split(",")
        if part.strip()
    )


def parse_exog_features(value):
    if value is None or pd.isna(value) or value == "":
        return None

    return [
        feature.strip()
        for feature in str(value).split(",")
        if feature.strip()
    ]


def params_from_best_setting(best_setting):
    params = {
        "order": parse_tuple(best_setting["order"]),
        "max_iter": int(best_setting["max_iter"]),
        "enforce_stationarity": bool(
            best_setting["enforce_stationarity"]
        ),
        "enforce_invertibility": bool(
            best_setting["enforce_invertibility"]
        )
    }

    seasonal_order = parse_tuple(
        best_setting.get("seasonal_order")
    )

    if seasonal_order is not None:
        params["seasonal_order"] = seasonal_order

    exog_features = parse_exog_features(
        best_setting.get("exog_features")
    )

    if exog_features is not None:
        params["exog_features"] = exog_features

    return params


def final_test(
    model_name,
    build_model,
    df,
    best_setting,
    test_steps=None,
    include_validation_in_training=True,
    refit_interval=1
):
    y_train, y_val, y_test, exog_train, exog_val, exog_test = prepare_ar_data(
        df
    )

    window_length = int(
        best_setting["seq_len"]
    )

    params = params_from_best_setting(
        best_setting
    )

    config = ARExperimentConfig(
        params=params,
        window_length=window_length
    )

    if include_validation_in_training:
        y_train_final = pd.concat(
            [y_train, y_val],
            axis=0
        )
        exog_train_final = pd.concat(
            [exog_train, exog_val],
            axis=0
        )
    else:
        y_train_final = y_train
        exog_train_final = exog_train

    y_test_eval = y_test

    if test_steps is not None:
        y_test_eval = y_test_eval.iloc[:test_steps]

    model_exog_train = None
    model_exog_test = None

    if uses_exog(params):
        model_exog_train, model_exog_test = scale_exog_train_test(
            exog_train=exog_train_final,
            exog_test=exog_test,
            exog_features=config.EXOG_FEATURES
        )

        model_exog_test = model_exog_test.loc[
            y_test_eval.index
        ]

    predictions = rolling_forecast(
        build_model_fn=build_model,
        train=y_train_final,
        test=y_test_eval,
        config=config,
        exog_train=model_exog_train,
        exog_test=model_exog_test,
        refit_interval=refit_interval
    )

    predictions = np.clip(
        predictions,
        0,
        None
    )

    mae, rmse, mape, smape = evaluate(
        y_test_eval,
        predictions
    )

    return {
        "model": model_name,
        "seq_len": window_length,
        "order": str(config.ORDER),
        "seasonal_order": (
            str(config.SEASONAL_ORDER)
            if hasattr(config, "SEASONAL_ORDER")
            else None
        ),
        "exog_features": (
            ",".join(config.EXOG_FEATURES)
            if hasattr(config, "EXOG_FEATURES")
            else None
        ),
        "max_iter": config.MAX_ITER,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "y_test": y_test_eval.to_numpy(),
        "y_pred": np.asarray(predictions)
    }

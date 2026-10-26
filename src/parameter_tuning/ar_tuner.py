import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    EXOG_LAG_STEPS,
    SELECTION_PROTOCOL,
    VALIDATION_STEPS,
    split_positions
)
from src.forecasting.rolling_forecast import rolling_forecast
from src.parameter_tuning.selection import (
    calculate_monthly_validation_metrics,
    select_robust_candidate
)
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

    y = y.ffill().bfill()

    exog = df.drop(
        columns=["pv_total_kWh"],
        errors="ignore"
    )

    exog = exog.replace(
        [np.inf, -np.inf],
        np.nan
    )

    exog = exog.ffill().bfill()
    exog_unlagged = exog.copy()
    exog = exog.shift(EXOG_LAG_STEPS)
    exog.iloc[:EXOG_LAG_STEPS] = exog_unlagged.iloc[:EXOG_LAG_STEPS]

    n_raw = len(y)

    train_end, val_end = split_positions(n_raw)

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

    if val_subset.empty:
        exog_val_scaled = pd.DataFrame(
            index=val_subset.index,
            columns=val_subset.columns,
            dtype=float
        )
    else:
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
    validation_steps=None,
    refit_interval=AR_REFIT_INTERVAL,
    return_predictions=False
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

    predictions, fit_diagnostics = rolling_forecast(
        build_model_fn=build_model,
        train=y_train,
        test=y_val_eval,
        config=config,
        exog_train=model_exog_train,
        exog_test=model_exog_val,
        refit_interval=refit_interval,
        return_diagnostics=True
    )

    # MAPE over the whole period is not recorded; see the neural tuner.
    mae, rmse, _ = evaluate(
        y_val_eval,
        predictions
    )
    block_metrics = calculate_monthly_validation_metrics(
        index=y_val_eval.index,
        y_true=y_val_eval,
        y_pred=predictions
    )

    result = {
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
        "input_features": ",".join(
            [
                "pv_total_kWh",
                *getattr(config, "EXOG_FEATURES", [])
            ]
        ),
        "feature_count": 1 + len(getattr(config, "EXOG_FEATURES", [])),
        "max_iter": config.MAX_ITER,
        "enforce_stationarity": config.ENFORCE_STATIONARITY,
        "enforce_invertibility": config.ENFORCE_INVERTIBILITY,
        "selection_protocol": SELECTION_PROTOCOL,
        "exog_lag_steps": EXOG_LAG_STEPS if uses_exog(params) else None,
        "validation_start": y_val_eval.index[0],
        "validation_end": y_val_eval.index[-1],
        "validation_steps": len(y_val_eval),
        "validation_refit_interval": refit_interval,
        **fit_diagnostics,
        **block_metrics,
        "val_mae": mae,
        "val_rmse": rmse
    }

    if return_predictions:
        return result, {
            "validation_index": y_val_eval.index,
            "y_true": y_val_eval.to_numpy(),
            "ensemble_prediction": np.asarray(predictions)
        }

    return result


def _key_text(value):
    if value is None:
        return ""

    if isinstance(value, float):
        if np.isnan(value):
            return ""

        if value.is_integer():
            value = int(value)

    text = str(value)

    # pandas reads the text None back as NaN, so both spell "no value".
    return "" if text in {"None", "nan"} else text


def configuration_key(
    seq_len,
    order,
    seasonal_order,
    exog_features,
    max_iter,
    enforce_stationarity,
    enforce_invertibility
):
    return tuple(
        _key_text(value)
        for value in (
            seq_len,
            order,
            seasonal_order,
            exog_features,
            max_iter,
            enforce_stationarity,
            enforce_invertibility
        )
    )


def params_key(window_length, params):
    exog_features = params.get("exog_features")

    return configuration_key(
        window_length,
        params.get("order"),
        params.get("seasonal_order"),
        ",".join(exog_features) if exog_features else None,
        params.get("max_iter"),
        params.get("enforce_stationarity"),
        params.get("enforce_invertibility")
    )


def result_key(result):
    return configuration_key(
        result.get("seq_len"),
        result.get("order"),
        result.get("seasonal_order"),
        result.get("exog_features"),
        result.get("max_iter"),
        result.get("enforce_stationarity"),
        result.get("enforce_invertibility")
    )


def load_checkpoint(checkpoint_path):
    """Completed results from an interrupted run under the current protocol.

    Rows written under another protocol or dataset describe a different
    experiment and are ignored, so they are recomputed rather than reused.
    """
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return {}

    rows = pd.read_csv(checkpoint_path).to_dict(orient="records")

    return {
        result_key(row): row
        for row in rows
        if row.get("selection_protocol") == SELECTION_PROTOCOL
    }


def save_checkpoint(checkpoint_path, results):
    """Rewrite the checkpoint with every result completed so far.

    Success and failure rows carry different columns, so appending would
    misalign them; the whole file is rewritten instead, which is cheap at grid
    sizes of a few dozen. Writing beside the file and then renaming keeps an
    interruption mid-write from leaving a truncated checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    pd.DataFrame(results).to_csv(temporary_path, index=False)
    os.replace(temporary_path, checkpoint_path)


def tune_ar_model(
    model_name,
    build_model,
    df,
    param_grid,
    window_lengths,
    validation_steps=VALIDATION_STEPS,
    refit_interval=AR_REFIT_INTERVAL,
    checkpoint_path=None
):
    """Evaluate every window length and parameter combination on validation.

    With a checkpoint path, each configuration's result is saved as soon as it
    finishes, and a rerun after an interruption skips the configurations
    already done. Seasonal models at 10-minute resolution can run for days, so
    losing all progress to one crash is not acceptable.
    """
    y_train, y_val, y_test, exog_train, exog_val, exog_test = prepare_ar_data(
        df
    )

    completed = load_checkpoint(checkpoint_path)

    if completed:
        print(
            f"resuming {model_name}: {len(completed)} configurations "
            "already completed"
        )

    all_results = []
    best_per_l = []

    for window_length in window_lengths:
        print(
            f"\nsearching {model_name} with L={window_length}"
        )

        results_for_l = []

        for params in generate_param_combinations(param_grid):
            key = params_key(window_length, params)

            if key in completed:
                result = completed[key]
                all_results.append(result)
                results_for_l.append(result)
                continue

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
                    validation_steps=validation_steps,
                    refit_interval=refit_interval
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
                    "selection_protocol": SELECTION_PROTOCOL,
                    "exog_lag_steps": (
                        EXOG_LAG_STEPS if uses_exog(params) else None
                    ),
                    "validation_start": y_val.index[0],
                    "validation_end": (
                        y_val.iloc[:validation_steps].index[-1]
                        if validation_steps is not None
                        else y_val.index[-1]
                    ),
                    "validation_steps": (
                        min(validation_steps, len(y_val))
                        if validation_steps is not None
                        else len(y_val)
                    ),
                    "validation_refit_interval": refit_interval,
                    "aic": np.nan,
                    "bic": np.nan,
                    "fit_converged": False,
                    "validation_blocks": np.nan,
                    "val_block_mae_mean": np.nan,
                    "val_block_mae_std": np.nan,
                    "val_block_rmse_mean": np.nan,
                    "val_block_rmse_std": np.nan,
                    "val_mae": np.nan,
                    "val_rmse": np.nan,
                    "error": str(e)
                }

            all_results.append(result)
            results_for_l.append(result)

            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, all_results)

        try:
            best_per_l.append(
                select_robust_candidate(
                    results_for_l,
                    information_criterion="bic",
                    require_converged=True
                )
            )
        except ValueError:
            pass

    return all_results, best_per_l


def select_best_l(best_per_l):
    return select_robust_candidate(
        best_per_l,
        require_converged=True
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


def parse_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no"}:
            return False

        raise ValueError(f"cannot parse boolean value: {value}")

    return bool(value)


def params_from_best_setting(best_setting):
    params = {
        "order": parse_tuple(best_setting["order"]),
        "max_iter": int(best_setting["max_iter"]),
        "enforce_stationarity": parse_bool(
            best_setting["enforce_stationarity"]
        ),
        "enforce_invertibility": parse_bool(
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
    test_offset=0,
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

    pretest_context = y_test.iloc[:test_offset]
    y_test_eval = y_test.iloc[test_offset:]

    if test_steps is not None:
        y_test_eval = y_test_eval.iloc[:test_steps]

    if include_validation_in_training:
        y_train_final = pd.concat(
            [y_train, y_val],
            axis=0
        )
        exog_train_final = pd.concat(
            [exog_train, exog_val],
            axis=0
        )
        state_context = (
            pretest_context
            if len(pretest_context)
            else None
        )
    else:
        y_train_final = y_train
        exog_train_final = exog_train
        state_context = pd.concat(
            [y_val, pretest_context],
            axis=0
        )

    model_exog_train = None
    model_exog_context = None
    model_exog_test = None

    if uses_exog(params):
        if include_validation_in_training:
            (
                model_exog_train,
                model_exog_context,
                model_exog_test
            ) = scale_exog_splits(
                exog_train=exog_train_final,
                exog_val=exog_test.iloc[:test_offset],
                exog_test=exog_test,
                exog_features=config.EXOG_FEATURES
            )
        else:
            (
                model_exog_train,
                model_exog_context,
                model_exog_test
            ) = scale_exog_splits(
                exog_train=exog_train,
                exog_val=pd.concat(
                    [exog_val, exog_test.iloc[:test_offset]],
                    axis=0
                ),
                exog_test=exog_test,
                exog_features=config.EXOG_FEATURES
            )

        model_exog_test = model_exog_test.loc[
            y_test_eval.index
        ]

    predictions, fit_diagnostics = rolling_forecast(
        build_model_fn=build_model,
        train=y_train_final,
        test=y_test_eval,
        config=config,
        exog_train=model_exog_train,
        exog_test=model_exog_test,
        refit_interval=refit_interval,
        context=state_context,
        exog_context=model_exog_context,
        return_diagnostics=True
    )

    if not fit_diagnostics["forecast_valid"]:
        raise RuntimeError(
            f"final {model_name} forecast was not fully valid: "
            f"{fit_diagnostics}"
        )

    predictions = np.clip(
        predictions,
        0,
        None
    )

    mae, rmse, _ = evaluate(
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
        "input_features": ",".join(
            [
                "pv_total_kWh",
                *getattr(config, "EXOG_FEATURES", [])
            ]
        ),
        "feature_count": 1 + len(getattr(config, "EXOG_FEATURES", [])),
        "max_iter": config.MAX_ITER,
        "final_aic": fit_diagnostics["aic"],
        "final_bic": fit_diagnostics["bic"],
        "final_fit_converged": fit_diagnostics["fit_converged"],
        "forecast_valid": fit_diagnostics["forecast_valid"],
        "fit_attempt_count": fit_diagnostics["fit_attempt_count"],
        "fit_retry_count": fit_diagnostics["fit_retry_count"],
        "fit_failure_count": fit_diagnostics["fit_failure_count"],
        "forecast_fallback_count": fit_diagnostics[
            "forecast_fallback_count"
        ],
        "state_update_failure_count": fit_diagnostics[
            "state_update_failure_count"
        ],
        "exog_lag_steps": EXOG_LAG_STEPS if uses_exog(params) else None,
        "training_data": (
            "train_validation"
            if include_validation_in_training
            else "train_only"
        ),
        "metric_aggregation": "deterministic",
        "state_context_steps": (
            len(state_context)
            if state_context is not None
            else 0
        ),
        "mae": mae,
        "rmse": rmse,
        "test_index": y_test_eval.index,
        "y_test": y_test_eval.to_numpy(),
        "y_pred": np.asarray(predictions)
    }

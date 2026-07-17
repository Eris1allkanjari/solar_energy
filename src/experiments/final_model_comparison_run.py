import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    EXOG_LAG_STEPS,
    FINAL_COMPARISON_PROTOCOL,
    MIN_SEASONAL_WINDOW,
    MAPE_PRODUCTION_THRESHOLD,
    NEURAL_SELECTION_PROTOCOL,
    SELECTION_PROTOCOL,
    TEST_OFFSET,
    TEST_STEPS,
    VALIDATION_END_RATIO,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import DATA_FILE_PATH
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.ar_tuner import final_test as final_ar_test
from src.parameter_tuning.plots import plot_real_vs_predicted
from src.parameter_tuning.selection import (
    SELECTION_MAE_KEY,
    select_robust_candidate
)
from src.parameter_tuning.tuner import final_test as final_neural_test
from src.training.evaluation import evaluate


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINAL_TEST_STEPS = TEST_STEPS

NEURAL_MODELS = [
    "lstm",
    "gru"
]

AR_MODELS = [
    "arima",
    "sarima",
    "arimax",
    "sarimax"
]

BASELINE_MODELS = [
    "persistence",
    "seasonal_naive_24h"
]

ALL_MODELS = NEURAL_MODELS + AR_MODELS + BASELINE_MODELS

PROTOCOL_COLUMNS = {
    "selection_protocol",
    "validation_start",
    "validation_end",
    "validation_steps",
    "validation_refit_interval",
    "validation_blocks",
    "val_block_mae_mean",
    "val_block_mae_std",
    "val_block_rmse_mean",
    "val_block_rmse_std"
}

NEURAL_PROTOCOL_COLUMNS = {
    "input_features",
    "feature_count",
    "loss",
    "huber_delta",
    "weight_decay",
    "gradient_clip",
    "shuffle_training",
    "seeds",
    "seed_count",
    "best_epochs",
    "best_epoch",
    "val_seed_mae_std",
    "val_seed_rmse_std",
    "val_seed_mae_values",
    "val_seed_rmse_values"
}

AR_PROTOCOL_COLUMNS = {
    "aic",
    "bic",
    "exog_features",
    "fit_converged",
    "exog_lag_steps"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final comparisons without repeating tuning."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help="Models to rerun; existing rows for other models are preserved."
    )

    return parser.parse_args()


def load_existing_summaries(models_to_replace):
    comparison_path = RESULTS_DIR / "final_model_comparison.csv"

    if not comparison_path.exists():
        return []

    comparison_df = pd.read_csv(
        comparison_path
    )

    required_columns = {"model", "selection_protocol"}
    missing_columns = required_columns.difference(
        comparison_df.columns
    )

    if missing_columns:
        return []

    comparison_df = comparison_df[
        comparison_df["selection_protocol"] == FINAL_COMPARISON_PROTOCOL
    ]

    if comparison_df.empty:
        return []

    resume_columns = {
        "state_context_steps",
        "test_offset",
        "training_data",
        "tuning_protocol"
    }

    if resume_columns.difference(comparison_df.columns):
        return []

    comparison_df = comparison_df[
        pd.to_numeric(
            comparison_df["test_offset"],
            errors="coerce"
        ) == TEST_OFFSET
    ]
    comparison_df = comparison_df[
        comparison_df["training_data"].isin(
            ["train_validation", "not_applicable"]
        )
    ]
    comparison_df = comparison_df[
        ~comparison_df["model"].isin(models_to_replace)
    ]

    return comparison_df.to_dict(
        orient="records"
    )


def validate_tuning_protocol(
    results_df,
    path,
    autoregressive
):
    missing_columns = PROTOCOL_COLUMNS.difference(
        results_df.columns
    )

    if missing_columns:
        raise RuntimeError(
            f"stale tuning results in {path}. Missing protocol columns: "
            f"{sorted(missing_columns)}. Rerun parameter tuning first."
        )

    if not autoregressive:
        missing_neural_columns = NEURAL_PROTOCOL_COLUMNS.difference(
            results_df.columns
        )

        if missing_neural_columns:
            raise RuntimeError(
                f"stale neural tuning results in {path}. Missing columns: "
                f"{sorted(missing_neural_columns)}. Rerun neural "
                "parameter tuning first."
            )

    else:
        missing_ar_columns = AR_PROTOCOL_COLUMNS.difference(
            results_df.columns
        )

        if missing_ar_columns:
            raise RuntimeError(
                f"stale AR tuning results in {path}. Missing columns: "
                f"{sorted(missing_ar_columns)}. Rerun autoregressive "
                "parameter tuning first."
            )

    protocols = set(
        results_df["selection_protocol"].dropna().astype(str)
    )

    expected_protocol = (
        SELECTION_PROTOCOL
        if autoregressive
        else NEURAL_SELECTION_PROTOCOL
    )

    if protocols != {expected_protocol}:
        raise RuntimeError(
            f"incompatible selection protocol in {path}: {protocols}. "
            "Rerun parameter tuning first."
        )

    validation_lengths = set(
        results_df["validation_steps"].dropna().astype(int)
    )

    if validation_lengths != {VALIDATION_STEPS}:
        raise RuntimeError(
            f"incompatible validation length in {path}: "
            f"{validation_lengths}. Expected {VALIDATION_STEPS}."
        )

    if autoregressive:
        refit_intervals = set(
            results_df["validation_refit_interval"].dropna().astype(int)
        )

        if refit_intervals != {AR_REFIT_INTERVAL}:
            raise RuntimeError(
                f"incompatible AR refit interval in {path}: "
                f"{refit_intervals}. Expected {AR_REFIT_INTERVAL}."
            )

        exogenous_rows = results_df[results_df["exog_features"].notna()]
        if not exogenous_rows.empty:
            lag_steps = set(
                exogenous_rows["exog_lag_steps"].dropna().astype(int)
            )
            if lag_steps != {EXOG_LAG_STEPS}:
                raise RuntimeError(
                    f"incompatible exogenous lag in {path}: {lag_steps}. "
                    f"Expected {EXOG_LAG_STEPS}."
                )

        converged = results_df["fit_converged"].astype(str).str.lower()
        if not converged.eq("true").all():
            raise RuntimeError(
                f"non-converged AR settings found in {path}. Rerun "
                "autoregressive parameter tuning first."
            )


def load_best_setting(model_name, autoregressive=False):
    suffix = "_ar_best_per_l.csv" if autoregressive else "_best_per_l.csv"
    path = RESULTS_DIR / f"{model_name}{suffix}"

    if not path.exists():
        raise FileNotFoundError(
            f"missing tuning results for {model_name}: {path}"
        )

    results_df = pd.read_csv(path)
    validate_tuning_protocol(
        results_df=results_df,
        path=path,
        autoregressive=autoregressive
    )

    valid_results = results_df.dropna(
        subset=[SELECTION_MAE_KEY]
    )

    if valid_results.empty:
        raise RuntimeError(
            f"no successful tuning result found for {model_name}"
        )

    if (
        model_name in {"sarima", "sarimax"}
        and valid_results["seq_len"].astype(int).min()
        < MIN_SEASONAL_WINDOW
    ):
        raise RuntimeError(
            f"incompatible seasonal windows in {path}. Rerun "
            "autoregressive parameter tuning first."
        )

    return select_robust_candidate(
        valid_results.to_dict(
            orient="records"
        )
    )


def result_summary(
    final_result,
    best_setting,
    model_family,
    test_start,
    test_end
):
    y_test = np.asarray(final_result["y_test"])
    percentage_metric_samples = int(
        (y_test > MAPE_PRODUCTION_THRESHOLD).sum()
    )

    return {
        "model": final_result["model"],
        "model_family": model_family,
        "seq_len": final_result["seq_len"],
        "test_start": test_start,
        "test_end": test_end,
        "test_offset": TEST_OFFSET,
        "test_steps": len(final_result["y_test"]),
        "training_data": final_result["training_data"],
        "selection_protocol": FINAL_COMPARISON_PROTOCOL,
        "tuning_protocol": best_setting["selection_protocol"],
        "validation_start": best_setting["validation_start"],
        "validation_end": best_setting["validation_end"],
        "validation_steps": (
            None if model_family == "baseline" else VALIDATION_STEPS
        ),
        "validation_blocks": best_setting["validation_blocks"],
        "val_block_mae_mean": best_setting["val_block_mae_mean"],
        "val_block_mae_std": best_setting["val_block_mae_std"],
        "val_block_rmse_mean": best_setting["val_block_rmse_mean"],
        "val_block_rmse_std": best_setting["val_block_rmse_std"],
        "val_seed_mae_std": best_setting.get("val_seed_mae_std"),
        "val_seed_rmse_std": best_setting.get("val_seed_rmse_std"),
        "val_seed_mae_values": best_setting.get("val_seed_mae_values"),
        "val_seed_rmse_values": best_setting.get("val_seed_rmse_values"),
        "best_epochs": best_setting.get("best_epochs"),
        "validation_refit_interval": best_setting[
            "validation_refit_interval"
        ],
        "ar_refit_interval": (
            AR_REFIT_INTERVAL
            if model_family == "autoregressive"
            else None
        ),
        "state_context_steps": final_result.get(
            "state_context_steps",
            final_result["seq_len"]
        ),
        "metric_aggregation": final_result.get("metric_aggregation"),
        "hidden_units_1": final_result.get("hidden_units_1"),
        "hidden_units_2": final_result.get("hidden_units_2"),
        "dropout": final_result.get("dropout"),
        "learning_rate": final_result.get("learning_rate"),
        "batch_size": final_result.get("batch_size"),
        "loss": final_result.get("loss"),
        "huber_delta": final_result.get("huber_delta"),
        "weight_decay": final_result.get("weight_decay"),
        "gradient_clip": final_result.get("gradient_clip"),
        "shuffle_training": final_result.get("shuffle_training"),
        "seeds": final_result.get("seeds"),
        "seed_count": final_result.get("seed_count"),
        "refit_epochs": final_result.get("refit_epochs"),
        "mae_by_seed": final_result.get("mae_by_seed"),
        "rmse_by_seed": final_result.get("rmse_by_seed"),
        "order": final_result.get("order"),
        "seasonal_order": final_result.get("seasonal_order"),
        "exog_features": final_result.get("exog_features"),
        "input_features": final_result.get("input_features"),
        "feature_count": final_result.get("feature_count"),
        "max_iter": final_result.get("max_iter"),
        "tuning_aic": best_setting.get("aic"),
        "tuning_bic": best_setting.get("bic"),
        "tuning_fit_converged": best_setting.get("fit_converged"),
        "exog_lag_steps": final_result.get("exog_lag_steps"),
        "mape_production_threshold": MAPE_PRODUCTION_THRESHOLD,
        "percentage_metric_samples": percentage_metric_samples,
        "percentage_metric_coverage": (
            percentage_metric_samples / len(y_test)
        ),
        "mae": final_result["mae"],
        "mae_std": final_result.get("mae_std"),
        "rmse": final_result["rmse"],
        "rmse_std": final_result.get("rmse_std"),
        "mape": final_result["mape"],
        "mape_std": final_result.get("mape_std"),
        "smape": final_result["smape"],
        "smape_std": final_result.get("smape_std"),
        "ensemble_mae": final_result.get("ensemble_mae"),
        "ensemble_rmse": final_result.get("ensemble_rmse"),
        "ensemble_mape": final_result.get("ensemble_mape"),
        "ensemble_smape": final_result.get("ensemble_smape")
    }


def save_model_result(final_result, summary):
    model_name = final_result["model"]
    output_path = RESULTS_DIR / (
        f"{model_name}_comparison_final_test_result.csv"
    )

    pd.DataFrame([summary]).to_csv(
        output_path,
        index=False
    )

    prediction_df = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(final_result["test_index"]),
            "y_true": np.asarray(final_result["y_test"]),
            "y_pred": np.asarray(final_result["y_pred"])
        }
    )
    prediction_df["absolute_error"] = np.abs(
        prediction_df["y_true"] - prediction_df["y_pred"]
    )
    prediction_df["squared_error"] = (
        prediction_df["y_true"] - prediction_df["y_pred"]
    ) ** 2
    prediction_df.to_csv(
        RESULTS_DIR / f"{model_name}_final_test_predictions.csv",
        index=False
    )

    plot_real_vs_predicted(
        y_true=final_result["y_test"],
        y_pred=final_result["y_pred"],
        output_path=str(
            RESULTS_DIR
            / f"{model_name}_comparison_actual_vs_predicted.png"
        ),
        title=(
            f"final comparison for {model_name}, "
            f"L={final_result['seq_len']}"
        ),
        max_points=500,
        show=False
    )


def save_comparison(summaries):
    comparison_df = pd.DataFrame(summaries)

    persistence_rows = comparison_df[
        comparison_df["model"] == "persistence"
    ]
    if not persistence_rows.empty:
        persistence_mae = float(persistence_rows.iloc[0]["mae"])
        persistence_rmse = float(persistence_rows.iloc[0]["rmse"])
        comparison_df["mae_skill_vs_persistence"] = (
            1 - comparison_df["mae"] / persistence_mae
        )
        comparison_df["rmse_skill_vs_persistence"] = (
            1 - comparison_df["rmse"] / persistence_rmse
        )

    comparison_df = comparison_df.sort_values(
        "mae"
    )

    comparison_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False
    )


def run_baseline(model_name, df_proc, test_start_index, test_steps):
    lags = {
        "persistence": 1,
        "seasonal_naive_24h": 24
    }
    lag = lags[model_name]
    target = df_proc["pv_total_kWh"]
    y_test = target.iloc[
        test_start_index:test_start_index + test_steps
    ]
    y_pred = target.shift(lag).loc[y_test.index].to_numpy()
    y_pred = np.clip(y_pred, 0, None)
    mae, rmse, mape, smape = evaluate(y_test, y_pred)

    return {
        "model": model_name,
        "seq_len": lag,
        "input_features": "pv_total_kWh",
        "feature_count": 1,
        "training_data": "not_applicable",
        "metric_aggregation": "deterministic",
        "state_context_steps": lag,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "test_index": y_test.index,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred
    }


def baseline_setting():
    return {
        "selection_protocol": "fixed_baseline_v1",
        "validation_start": None,
        "validation_end": None,
        "validation_blocks": None,
        "val_block_mae_mean": None,
        "val_block_mae_std": None,
        "val_block_rmse_mean": None,
        "val_block_rmse_std": None,
        "validation_refit_interval": None
    }


def main():
    args = parse_args()
    selected_models = set(
        args.models
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset(
        DATA_FILE_PATH
    )
    df_proc = prepare_dataframe(
        df
    )

    n_raw = len(df_proc)
    test_split_start = int(n_raw * VALIDATION_END_RATIO)
    test_start_index = test_split_start + TEST_OFFSET
    test_df = df_proc.iloc[
        test_start_index:test_start_index + FINAL_TEST_STEPS
    ]

    if len(test_df) < FINAL_TEST_STEPS:
        raise RuntimeError(
            f"requested {FINAL_TEST_STEPS} test steps, "
            f"but only {len(test_df)} are available"
        )

    test_start = test_df.index[0]
    test_end = test_df.index[-1]
    summaries = load_existing_summaries(
        models_to_replace=selected_models
    )
    percentage_metric_samples = int(
        (test_df["pv_total_kWh"] > MAPE_PRODUCTION_THRESHOLD).sum()
    )
    percentage_metric_coverage = (
        percentage_metric_samples / len(test_df)
    )

    for summary in summaries:
        summary["percentage_metric_samples"] = percentage_metric_samples
        summary["percentage_metric_coverage"] = percentage_metric_coverage

    for model_name in NEURAL_MODELS:
        if model_name not in selected_models:
            continue

        print(
            f"\nrunning final comparison for {model_name}"
        )

        best_setting = load_best_setting(
            model_name=model_name
        )

        final_result = final_neural_test(
            model_name=model_name,
            get_model=get_model,
            df_proc=df_proc,
            best_setting=best_setting,
            test_steps=FINAL_TEST_STEPS,
            test_offset=TEST_OFFSET
        )

        summary = result_summary(
            final_result=final_result,
            best_setting=best_setting,
            model_family="neural",
            test_start=test_start,
            test_end=test_end
        )
        summaries.append(summary)

        save_model_result(
            final_result=final_result,
            summary=summary
        )
        save_comparison(
            summaries
        )

    for model_name in AR_MODELS:
        if model_name not in selected_models:
            continue

        print(
            f"\nrunning final comparison for {model_name}"
        )

        best_setting = load_best_setting(
            model_name=model_name,
            autoregressive=True
        )

        final_result = final_ar_test(
            model_name=model_name,
            build_model=get_ar_model_builder(model_name),
            df=df,
            best_setting=best_setting,
            test_steps=FINAL_TEST_STEPS,
            test_offset=TEST_OFFSET,
            include_validation_in_training=True,
            refit_interval=AR_REFIT_INTERVAL
        )

        summary = result_summary(
            final_result=final_result,
            best_setting=best_setting,
            model_family="autoregressive",
            test_start=test_start,
            test_end=test_end
        )
        summaries.append(summary)

        save_model_result(
            final_result=final_result,
            summary=summary
        )
        save_comparison(
            summaries
        )

    for model_name in BASELINE_MODELS:
        if model_name not in selected_models:
            continue

        print(f"\nrunning final comparison for {model_name}")
        final_result = run_baseline(
            model_name=model_name,
            df_proc=df_proc,
            test_start_index=test_start_index,
            test_steps=FINAL_TEST_STEPS
        )
        summary = result_summary(
            final_result=final_result,
            best_setting=baseline_setting(),
            model_family="baseline",
            test_start=test_start,
            test_end=test_end
        )
        summaries.append(summary)
        save_model_result(
            final_result=final_result,
            summary=summary
        )
        save_comparison(summaries)

    print(
        "\nsaved final model comparison to "
        f"{RESULTS_DIR / 'final_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()

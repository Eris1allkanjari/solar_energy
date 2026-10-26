import argparse

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    DAY_PERIOD_HOURS,
    EXOG_LAG_STEPS,
    BY_DAY_PERIOD_HOURS,
    PERIOD_DEFINITION,
    UPWARDS_DOWNWARDS_PERIOD_HOURS,
    FINAL_COMPARISON_PROTOCOL,
    MIN_SEASONAL_WINDOW,
    MAPE_PRODUCTION_THRESHOLD,
    NEURAL_SELECTION_PROTOCOL,
    PV_QUALITY_THRESHOLD,
    SELECTION_PROTOCOL,
    TEST_OFFSET,
    TEST_STEPS,
    VALIDATION_STEPS,
    split_positions
)
from src.configs.dataset_selection import (
    ACTIVE_DATASET,
    add_dataset_argument,
    hours_to_steps
)
from src.data.loader import load_dataset
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import (
    AR_TUNING_RESULTS_DIR,
    DATA_FILE_PATH,
    FINAL_COMPARISON_RESULTS_DIR,
    NEURAL_TUNING_RESULTS_DIR,
    PV_QUALITY_HOURLY_PATH
)
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.ar_tuner import final_test as final_ar_test
from src.parameter_tuning.plots import plot_real_vs_predicted
from src.parameter_tuning.selection import (
    SELECTION_MAE_KEY,
    calculate_monthly_validation_metrics,
    select_robust_candidate
)
from src.parameter_tuning.tuner import final_test as final_neural_test
from src.training.evaluation import (
    as_capacity_percentage,
    clock_hour_mask,
    clock_period_mae,
    clock_period_mape,
    evaluate,
    period_mapes
)
from src.utils.capacity import training_production_range


RESULTS_DIR = FINAL_COMPARISON_RESULTS_DIR
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
    "weight_decay",
    "gradient_clip",
    "lr_decay",
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
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help=(
            "Rewrite the derived summary tables from the saved comparison CSV "
            "without refitting. Use after a change that only affects which "
            "columns are written."
        )
    )

    add_dataset_argument(parser)
    return parser.parse_args()


# The MAPE periods were renamed without any change to how they are computed, so
# results saved under the old names are read back under the new ones instead of
# forcing every model to be refitted.
RENAMED_COLUMNS = {
    "ramp_mape": "upwards_downwards_mape",
    "ramp_mape_samples": "upwards_downwards_mape_samples",
    "midday_mape": "by_day_mape",
    "midday_mape_samples": "by_day_mape_samples"
}


HIGH_QUALITY_PERIOD_MAPE_COLUMNS = [
    "high_quality_upwards_downwards_mape",
    "high_quality_by_day_mape"
]


def high_quality_period_mapes_from_predictions(model_name):
    """Recompute the high-quality period MAPEs from a saved prediction file.

    Results saved before these columns existed still have their predictions on
    disk, so the columns can be filled in without refitting. Neural models are
    scored per seed and averaged, exactly as a fresh run would, which needs the
    per-seed columns; a file saved before those existed cannot be used.
    """
    path = RESULTS_DIR / f"{model_name}_final_test_predictions.csv"
    predictions_df = pd.read_csv(path, parse_dates=["time"])
    seed_columns = [
        column
        for column in predictions_df.columns
        if column.startswith("y_pred_seed_")
    ]

    if model_name in NEURAL_MODELS and not seed_columns:
        raise RuntimeError(
            f"{path} has no per-seed predictions, so {model_name} cannot be "
            "rescored the way a fresh run would score it. Rerun the final "
            f"comparison for {model_name}."
        )

    prediction_columns = seed_columns or ["y_pred"]

    return period_mapes(
        predictions_df["time"],
        predictions_df["y_true"],
        [predictions_df[column] for column in prediction_columns],
        mask=(
            predictions_df["pv_observation_fraction"] >= PV_QUALITY_THRESHOLD
        ),
        prefix="high_quality_"
    )


def read_comparison(comparison_path):
    comparison_df = pd.read_csv(comparison_path).rename(
        columns=RENAMED_COLUMNS
    )

    if "period_definition" in comparison_df.columns:
        comparison_df["period_definition"] = PERIOD_DEFINITION

    for column in HIGH_QUALITY_PERIOD_MAPE_COLUMNS:
        if column not in comparison_df.columns:
            comparison_df[column] = np.nan

    missing = comparison_df[HIGH_QUALITY_PERIOD_MAPE_COLUMNS].isna().any(axis=1)

    for row in comparison_df.index[missing]:
        values = high_quality_period_mapes_from_predictions(
            comparison_df.at[row, "model"]
        )

        for column, value in values.items():
            comparison_df.at[row, column] = value

    # Whole-period percentage error is no longer recorded anywhere, so it is
    # dropped from saved results as well when the tables are rewritten.
    return comparison_df.drop(
        columns=[
            "mape",
            "mape_std",
            "ensemble_mape",
            "high_quality_mape",
            "high_quality_mape_std",
            "percentage_metric_samples",
            "percentage_metric_coverage"
        ],
        errors="ignore"
    )


def load_existing_summaries(
    models_to_replace,
    test_start,
    test_end,
    test_steps
):
    comparison_path = RESULTS_DIR / "final_model_comparison.csv"

    if not comparison_path.exists():
        return []

    comparison_df = read_comparison(comparison_path)

    required_columns = {"model", "selection_protocol"}
    missing_columns = required_columns.difference(
        comparison_df.columns
    )

    if missing_columns:
        return []

    resume_columns = {
        "state_context_steps",
        "test_offset",
        "test_steps",
        "test_start",
        "test_end",
        "training_data",
        "tuning_protocol",
        "metric_aggregation",
        "high_quality_mae",
        "nmae",
        "day_nmae",
        "night_nmae",
        "upwards_downwards_mape",
        "by_day_mape"
    }

    if resume_columns.difference(comparison_df.columns):
        if set(models_to_replace) != set(ALL_MODELS):
            raise RuntimeError(
                "existing final results are stale; rerun all models under "
                "the current comparison protocol"
            )
        return []

    compatible = (
        comparison_df["selection_protocol"].eq(
            FINAL_COMPARISON_PROTOCOL
        )
        & pd.to_numeric(
            comparison_df["test_offset"],
            errors="coerce"
        ).eq(TEST_OFFSET)
        & pd.to_numeric(
            comparison_df["test_steps"],
            errors="coerce"
        ).eq(test_steps)
        & pd.to_datetime(comparison_df["test_start"]).eq(test_start)
        & pd.to_datetime(comparison_df["test_end"]).eq(test_end)
        & comparison_df["training_data"].isin(
            ["train_validation", "not_applicable"]
        )
        # Neural training can change without the final protocol changing, as
        # with the learning-rate decay, so a kept neural row must also have been
        # tuned under the current neural protocol.
        & (
            ~comparison_df["model"].isin(NEURAL_MODELS)
            | comparison_df["tuning_protocol"].eq(NEURAL_SELECTION_PROTOCOL)
        )
    )
    incompatible_preserved_models = set(
        comparison_df.loc[
            ~compatible
            & ~comparison_df["model"].isin(models_to_replace),
            "model"
        ]
    )

    if incompatible_preserved_models:
        raise RuntimeError(
            "existing final results use an incompatible protocol for: "
            f"{sorted(incompatible_preserved_models)}. Rerun those models."
        )

    comparison_df = comparison_df[
        compatible & ~comparison_df["model"].isin(models_to_replace)
    ]

    return comparison_df.to_dict(
        orient="records"
    )


def load_pv_quality():
    """PV coverage per interval of the active dataset, indexed by time.

    Datasets built from the 10-minute record carry their own coverage column at
    their own resolution, so the quality mask always lines up with the rows
    being scored. The validated hourly dataset predates that column and reads
    the separate hourly PV audit instead.

    The fully_missing_hour column keeps its historical name at every
    resolution; at 10 minutes it marks a fully missing 10-minute interval.
    """
    if ACTIVE_DATASET.coverage_in_file:
        quality_df = pd.read_csv(
            DATA_FILE_PATH,
            usecols=["time", "observation_fraction"]
        )
        quality_df["time"] = pd.to_datetime(quality_df["time"])
        quality_df = quality_df.set_index("time")
        quality_df["fully_missing_hour"] = quality_df[
            "observation_fraction"
        ].eq(0)
    else:
        if not PV_QUALITY_HOURLY_PATH.exists():
            raise FileNotFoundError(
                "missing PV quality audit. Run "
                "python -m src.analysis.pv_data_quality_analysis first."
            )

        quality_df = pd.read_csv(PV_QUALITY_HOURLY_PATH)
        quality_df["time"] = pd.to_datetime(
            quality_df["time"],
            utc=True
        ).dt.tz_localize(None)
        quality_df = quality_df.set_index("time")

    if quality_df.index.duplicated().any():
        raise ValueError("PV quality data contains duplicate timestamps")

    return quality_df


def attach_pv_quality(final_result, quality_df):
    test_index = pd.DatetimeIndex(final_result["test_index"])
    quality = quality_df.reindex(test_index)

    if quality["observation_fraction"].isna().any():
        raise ValueError("PV quality audit does not cover the full test period")

    final_result["pv_observation_fraction"] = quality[
        "observation_fraction"
    ].to_numpy()
    final_result["pv_fully_missing_hour"] = quality[
        "fully_missing_hour"
    ].astype(bool).to_numpy()
    return final_result


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
    tuning_dir = (
        AR_TUNING_RESULTS_DIR
        if autoregressive
        else NEURAL_TUNING_RESULTS_DIR
    )
    path = tuning_dir / f"{model_name}{suffix}"

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
    test_end,
    capacity_kwh,
    minimum_kwh
):
    y_test = np.asarray(final_result["y_test"])
    primary_prediction = np.asarray(final_result["y_pred"])
    observation_fraction = np.asarray(
        final_result["pv_observation_fraction"]
    )
    high_quality_mask = observation_fraction >= PV_QUALITY_THRESHOLD

    if not high_quality_mask.any():
        raise RuntimeError("no high-quality PV observations in test period")

    # Rolling-origin robustness: score the one-step-ahead predictions per
    # calendar-month block instead of a single pooled aggregate, so the model
    # ranking can be judged against block-to-block variation rather than one
    # slice of the year.
    test_index = pd.DatetimeIndex(final_result["test_index"])
    test_block_metrics = calculate_monthly_validation_metrics(
        index=test_index,
        y_true=y_test,
        y_pred=primary_prediction
    )
    high_quality_block_metrics = calculate_monthly_validation_metrics(
        index=test_index[high_quality_mask],
        y_true=y_test[high_quality_mask],
        y_pred=primary_prediction[high_quality_mask]
    )
    seed_predictions = final_result.get("seed_predictions")

    def seed_mean_metric(metric_function):
        """Aggregate a metric the way every other metric here is.

        Neural models yield one prediction per seed and each reported figure is
        the mean over them, so a metric that aggregated differently would not be
        comparable with the ones sitting beside it in the same row.
        """
        if seed_predictions is None:
            return metric_function(test_index, y_test, primary_prediction)

        return float(
            np.mean(
                [
                    metric_function(
                        test_index,
                        y_test,
                        np.asarray(seed_prediction)
                    )
                    for seed_prediction in seed_predictions
                ]
            )
        )

    # Percentage error is reported over the upwards/downwards hours and the
    # by-day plateau separately, and nowhere else. Production while it rises and
    # falls is small enough that dividing by it inflates the percentage
    # regardless of how accurate the forecast was in kWh, so mixing the two
    # periods hides which is which; over night or over the whole period the
    # denominator is near zero too often for the mean to carry meaning at all.
    upwards_downwards_mape = seed_mean_metric(
        lambda index, true, pred: clock_period_mape(
            index, true, pred, UPWARDS_DOWNWARDS_PERIOD_HOURS
        )
    )
    by_day_mape = seed_mean_metric(
        lambda index, true, pred: clock_period_mape(
            index, true, pred, BY_DAY_PERIOD_HOURS
        )
    )
    upwards_downwards_mape_samples = int(
        (
            clock_hour_mask(test_index, UPWARDS_DOWNWARDS_PERIOD_HOURS)
            & (y_test > MAPE_PRODUCTION_THRESHOLD)
        ).sum()
    )
    by_day_mape_samples = int(
        (
            clock_hour_mask(test_index, BY_DAY_PERIOD_HOURS)
            & (y_test > MAPE_PRODUCTION_THRESHOLD)
        ).sum()
    )

    # Pooled MAE is diluted by the night hours, where every model scores near
    # zero because predicting no production is trivial, so the day and night
    # periods are reported separately.
    day_hours_mask = clock_hour_mask(test_index, DAY_PERIOD_HOURS)

    if seed_predictions is not None:
        day_night = np.asarray(
            [
                clock_period_mae(
                    test_index,
                    y_test,
                    np.asarray(seed_prediction),
                    DAY_PERIOD_HOURS
                )
                for seed_prediction in seed_predictions
            ]
        )
        day_mae, night_mae = day_night.mean(axis=0)
        day_mae_std = float(day_night[:, 0].std(ddof=0))
    else:
        day_mae, night_mae = clock_period_mae(
            test_index,
            y_test,
            primary_prediction,
            DAY_PERIOD_HOURS
        )
        day_mae_std = np.nan

    if seed_predictions is not None:
        high_quality_metric_values = np.asarray(
            [
                evaluate(
                    y_test[high_quality_mask],
                    np.asarray(seed_prediction)[high_quality_mask]
                )
                for seed_prediction in seed_predictions
            ]
        )
        high_quality_metrics = high_quality_metric_values.mean(axis=0)
        high_quality_metric_std = high_quality_metric_values.std(
            axis=0,
            ddof=0
        )
    else:
        high_quality_metrics = np.asarray(
            evaluate(
                y_test[high_quality_mask],
                primary_prediction[high_quality_mask]
            )
        )
        high_quality_metric_std = np.full(3, np.nan)

    # The sensitivity analysis follows the same reporting rule as the main
    # table: percentage error only over the upwards/downwards and by-day hours.
    high_quality_period_mapes = period_mapes(
        test_index,
        y_test,
        (
            seed_predictions
            if seed_predictions is not None
            else [primary_prediction]
        ),
        mask=high_quality_mask,
        prefix="high_quality_"
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
        "weight_decay": final_result.get("weight_decay"),
        "gradient_clip": final_result.get("gradient_clip"),
        "lr_decay": final_result.get("lr_decay"),
        "shuffle_training": final_result.get("shuffle_training"),
        "seeds": final_result.get("seeds"),
        "seed_count": final_result.get("seed_count"),
        "refit_epochs": final_result.get("refit_epochs"),
        "refit_epoch_median": final_result.get("refit_epoch_median"),
        "mae_by_seed": final_result.get("mae_by_seed"),
        "rmse_by_seed": final_result.get("rmse_by_seed"),
        "order": final_result.get("order"),
        "seasonal_order": final_result.get("seasonal_order"),
        "exog_features": final_result.get("exog_features"),
        "input_features": final_result.get("input_features"),
        "feature_count": final_result.get("feature_count"),
        "max_iter": final_result.get("max_iter"),
        "final_aic": final_result.get("final_aic"),
        "final_bic": final_result.get("final_bic"),
        "final_fit_converged": final_result.get("final_fit_converged"),
        "forecast_valid": final_result.get("forecast_valid"),
        "fit_attempt_count": final_result.get("fit_attempt_count"),
        "fit_retry_count": final_result.get("fit_retry_count"),
        "fit_failure_count": final_result.get("fit_failure_count"),
        "forecast_fallback_count": final_result.get(
            "forecast_fallback_count"
        ),
        "state_update_failure_count": final_result.get(
            "state_update_failure_count"
        ),
        "tuning_aic": best_setting.get("aic"),
        "tuning_bic": best_setting.get("bic"),
        "tuning_fit_converged": best_setting.get("fit_converged"),
        "exog_lag_steps": final_result.get("exog_lag_steps"),
        "mape_production_threshold": MAPE_PRODUCTION_THRESHOLD,
        "pv_quality_threshold": PV_QUALITY_THRESHOLD,
        "high_quality_samples": int(high_quality_mask.sum()),
        "high_quality_coverage": float(high_quality_mask.mean()),
        "fully_missing_target_hours": int(
            np.asarray(final_result["pv_fully_missing_hour"]).sum()
        ),
        "high_quality_mae": high_quality_metrics[0],
        "high_quality_mae_std": high_quality_metric_std[0],
        "high_quality_rmse": high_quality_metrics[1],
        "high_quality_rmse_std": high_quality_metric_std[1],
        **high_quality_period_mapes,
        "period_definition": PERIOD_DEFINITION,
        "day_hours": int(day_hours_mask.sum()),
        "night_hours": int((~day_hours_mask).sum()),
        "day_mae": day_mae,
        "day_mae_std": day_mae_std,
        "night_mae": night_mae,
        "upwards_downwards_mape_samples": upwards_downwards_mape_samples,
        "upwards_downwards_mape": upwards_downwards_mape,
        "by_day_mape_samples": by_day_mape_samples,
        "by_day_mape": by_day_mape,
        # NMAE, the min-max scaled MAE. The offset cancels when the two scaled
        # series are subtracted, so this is MAE / (max - min); with a training
        # minimum of zero it coincides with max scaling, MAE / max.
        "capacity_kwh": capacity_kwh,
        "minimum_kwh": minimum_kwh,
        "scaling_range_kwh": capacity_kwh - minimum_kwh,
        "nmae": as_capacity_percentage(
            final_result["mae"],
            capacity_kwh,
            minimum_kwh
        ),
        "day_nmae": as_capacity_percentage(
            day_mae,
            capacity_kwh,
            minimum_kwh
        ),
        "night_nmae": as_capacity_percentage(
            night_mae,
            capacity_kwh,
            minimum_kwh
        ),
        "test_blocks": test_block_metrics["validation_blocks"],
        "test_block_mae_mean": test_block_metrics["val_block_mae_mean"],
        "test_block_mae_std": test_block_metrics["val_block_mae_std"],
        "test_block_rmse_mean": test_block_metrics["val_block_rmse_mean"],
        "test_block_rmse_std": test_block_metrics["val_block_rmse_std"],
        "high_quality_test_blocks": high_quality_block_metrics[
            "validation_blocks"
        ],
        "high_quality_test_block_mae_mean": high_quality_block_metrics[
            "val_block_mae_mean"
        ],
        "high_quality_test_block_mae_std": high_quality_block_metrics[
            "val_block_mae_std"
        ],
        "high_quality_test_block_rmse_mean": high_quality_block_metrics[
            "val_block_rmse_mean"
        ],
        "high_quality_test_block_rmse_std": high_quality_block_metrics[
            "val_block_rmse_std"
        ],
        "mae": final_result["mae"],
        "mae_std": final_result.get("mae_std"),
        "rmse": final_result["rmse"],
        "rmse_std": final_result.get("rmse_std"),
        "ensemble_mae": final_result.get("ensemble_mae"),
        "ensemble_rmse": final_result.get("ensemble_rmse")
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

    # For neural models y_pred is the average of the seed predictions, while the
    # reported metrics are averages of per-seed metrics. The two can differ a
    # lot when one seed is poor, so each seed's own prediction is kept too.
    prediction_df = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(final_result["test_index"]),
            "y_true": np.asarray(final_result["y_test"]),
            "y_pred": np.asarray(final_result["y_pred"])
        }
    )
    seed_predictions = final_result.get("seed_predictions")

    if seed_predictions is not None:
        seeds = str(final_result["seeds"]).split(",")

        for seed, seed_prediction in zip(seeds, seed_predictions):
            prediction_df[f"y_pred_seed_{seed}"] = np.asarray(seed_prediction)

    prediction_df["pv_observation_fraction"] = np.asarray(
        final_result["pv_observation_fraction"]
    )
    prediction_df["pv_fully_missing_hour"] = np.asarray(
        final_result["pv_fully_missing_hour"]
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
        persistence_high_quality_mae = float(
            persistence_rows.iloc[0]["high_quality_mae"]
        )
        persistence_high_quality_rmse = float(
            persistence_rows.iloc[0]["high_quality_rmse"]
        )
        comparison_df["high_quality_mae_skill_vs_persistence"] = (
            1
            - comparison_df["high_quality_mae"]
            / persistence_high_quality_mae
        )
        comparison_df["high_quality_rmse_skill_vs_persistence"] = (
            1
            - comparison_df["high_quality_rmse"]
            / persistence_high_quality_rmse
        )
        persistence_block_mae = float(
            persistence_rows.iloc[0]["test_block_mae_mean"]
        )
        comparison_df["test_block_mae_skill_vs_persistence"] = (
            1 - comparison_df["test_block_mae_mean"] / persistence_block_mae
        )

    comparison_df = comparison_df.sort_values(
        "mae"
    )

    comparison_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False
    )
    # The reported metric set, and nothing else. Percentage error appears only
    # for the upwards/downwards hours (06-09 and 15-17) and the by-day hours
    # (10-14), never for night or the whole period, because a denominator near
    # zero makes the mean meaningless there. Absolute error appears only scaled:
    # over the whole test period, and split into day (06-17) and night (18-05)
    # because pooling lets the trivially-zero dark hours halve it. The divisor
    # and the period definitions stay in final_model_comparison.csv.
    comparison_df[
        [
            "model",
            "upwards_downwards_mape",
            "by_day_mape",
            "nmae",
            "day_nmae",
            "night_nmae"
        ]
    ].sort_values("nmae").to_csv(
        RESULTS_DIR / "final_model_reported_metrics.csv",
        index=False
    )
    comparison_df[
        [
            "model",
            "high_quality_mae",
            "high_quality_rmse",
            "high_quality_upwards_downwards_mape",
            "high_quality_by_day_mape"
        ]
    ].sort_values("high_quality_mae").to_csv(
        RESULTS_DIR / "final_model_high_quality_metrics.csv",
        index=False
    )
    comparison_df[
        [
            "model",
            "test_blocks",
            "test_block_mae_mean",
            "test_block_mae_std",
            "test_block_rmse_mean",
            "test_block_rmse_std",
            "high_quality_test_block_mae_mean",
            "high_quality_test_block_mae_std"
        ]
    ].sort_values("test_block_mae_mean").to_csv(
        RESULTS_DIR / "final_model_block_metrics.csv",
        index=False
    )

def run_baseline(model_name, df_proc, test_start_index, test_steps):
    lags = {
        "persistence": 1,
        "seasonal_naive_24h": hours_to_steps(24)
    }
    lag = lags[model_name]
    target = df_proc["pv_total_kWh"]
    y_test = target.iloc[
        test_start_index:test_start_index + test_steps
    ]
    y_pred = target.shift(lag).loc[y_test.index].to_numpy()
    y_pred = np.clip(y_pred, 0, None)
    mae, rmse, _ = evaluate(y_test, y_pred)

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


def rewrite_tables():
    """Re-derive the summary tables from the saved comparison CSV.

    Every value the tables need is already stored there, so changing which
    columns get written does not have to cost a full refit. save_comparison is
    replayed rather than reimplemented, which keeps the regenerated files
    identical to what a real run would produce.
    """
    comparison_path = RESULTS_DIR / "final_model_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"missing {comparison_path}. Run the comparison first."
        )

    save_comparison(
        read_comparison(comparison_path).to_dict(orient="records")
    )
    print(f"rewrote summary tables from {comparison_path}")


def main():
    args = parse_args()

    if args.tables_only:
        rewrite_tables()
        return

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
    # One fixed divisor for every model and every split, taken from training
    # data only, so scaled errors stay comparable and leak no test information.
    minimum_kwh, capacity_kwh = training_production_range(df_proc)
    print(
        f"scaling reference: training min {minimum_kwh:.1f} kWh, "
        f"max {capacity_kwh:.1f} kWh, range {capacity_kwh - minimum_kwh:.1f} kWh"
    )
    quality_df = load_pv_quality()

    n_raw = len(df_proc)
    _, test_split_start = split_positions(n_raw)
    test_start_index = test_split_start + TEST_OFFSET
    test_df = df_proc.iloc[test_start_index:]

    if FINAL_TEST_STEPS is not None:
        test_df = test_df.iloc[:FINAL_TEST_STEPS]

    if test_df.empty:
        raise RuntimeError("the final test period is empty")

    test_start = test_df.index[0]
    test_end = test_df.index[-1]
    final_test_steps = len(test_df)
    summaries = load_existing_summaries(
        models_to_replace=selected_models,
        test_start=test_start,
        test_end=test_end,
        test_steps=final_test_steps
    )
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
            test_steps=final_test_steps,
            test_offset=TEST_OFFSET
        )
        final_result = attach_pv_quality(final_result, quality_df)

        summary = result_summary(
            final_result=final_result,
            best_setting=best_setting,
            model_family="neural",
            test_start=test_start,
            test_end=test_end,
            capacity_kwh=capacity_kwh,
            minimum_kwh=minimum_kwh
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
            test_steps=final_test_steps,
            test_offset=TEST_OFFSET,
            include_validation_in_training=True,
            refit_interval=AR_REFIT_INTERVAL
        )
        final_result = attach_pv_quality(final_result, quality_df)

        summary = result_summary(
            final_result=final_result,
            best_setting=best_setting,
            model_family="autoregressive",
            test_start=test_start,
            test_end=test_end,
            capacity_kwh=capacity_kwh,
            minimum_kwh=minimum_kwh
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
            test_steps=final_test_steps
        )
        final_result = attach_pv_quality(final_result, quality_df)
        summary = result_summary(
            final_result=final_result,
            best_setting=baseline_setting(),
            model_family="baseline",
            test_start=test_start,
            test_end=test_end,
            capacity_kwh=capacity_kwh,
            minimum_kwh=minimum_kwh
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

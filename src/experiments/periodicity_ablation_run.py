"""Ablation of the explicit time-of-day encoding.

The neural models receive the daily cycle twice: implicitly, through the window
of recent PV history, and explicitly, through the ``hour_sin``/``hour_cos``
components. This experiment removes the explicit encoding and re-runs the model,
reporting the difference.

Both evaluation splits are available so the result can be compared like with
like. ``--split validation`` reproduces the setup used by the feature
significance study (fit on the training split, score the validation window),
while ``--split test`` uses the final protocol (fit on train+validation, score
the held-out test period). Running both isolates whether a difference comes from
the encoding itself or from the evaluation period.

GRU is the default because it is the strongest model in the final comparison and
the cheapest neural model to refit.
"""

import argparse

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    MAPE_PRODUCTION_THRESHOLD,
    PV_QUALITY_THRESHOLD,
    TEST_OFFSET,
    TEST_STEPS,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.experiments.constants import (
    DATA_FILE_PATH,
    PERIODICITY_ABLATION_RESULTS_DIR
)
from src.experiments.final_model_comparison_run import (
    load_best_setting,
    load_pv_quality
)
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.plots import (
    plot_forecast_zoom_all_models,
    plot_periodicity_ablation
)
from src.parameter_tuning.selection import calculate_monthly_validation_metrics
from src.parameter_tuning.tuner import (
    evaluate_on_validation,
    final_test as final_neural_test,
    params_from_best_setting,
    parse_seeds
)
from src.training.evaluation import evaluate


RESULTS_DIR = PERIODICITY_ABLATION_RESULTS_DIR
EXPERIMENT_PROTOCOL = "periodicity_ablation_v2"
TIME_FEATURES = ["hour_sin", "hour_cos"]

VARIANTS = {
    "with_time_features": True,
    "without_time_features": False
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a model with and without the explicit time-of-day "
            "features on a single evaluation split."
        )
    )
    parser.add_argument(
        "--model",
        default="gru",
        choices=["gru", "lstm"],
        help="Neural model to ablate (default: gru, the best final model)."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation"],
        help=(
            "test: fit on train+validation, score the test period (final "
            "protocol). validation: fit on train only, score the validation "
            "window (matches the feature significance study)."
        )
    )
    parser.add_argument(
        "--zoom-hours",
        type=int,
        default=72,
        help="Width of the qualitative comparison window in hours."
    )
    return parser.parse_args()


def prepare_variant(df, keep_time_features):
    df_proc = prepare_dataframe(df)

    if keep_time_features:
        return df_proc

    missing = set(TIME_FEATURES).difference(df_proc.columns)

    if missing:
        raise ValueError(
            f"expected time features are absent: {sorted(missing)}"
        )

    return df_proc.drop(columns=TIME_FEATURES)


def run_variant(model_name, df_proc, best_setting, split):
    """Fit and score one variant, returning a split-independent shape."""
    if split == "test":
        result = final_neural_test(
            model_name=model_name,
            get_model=get_model,
            df_proc=df_proc,
            best_setting=best_setting,
            test_steps=TEST_STEPS,
            test_offset=TEST_OFFSET
        )

        return {
            "index": pd.DatetimeIndex(result["test_index"]),
            "y_true": np.asarray(result["y_test"]),
            "seed_predictions": np.asarray(result["seed_predictions"]),
            "input_features": result["input_features"],
            "feature_count": result["feature_count"],
            "seq_len": result["seq_len"],
            "seeds": result["seeds"],
            "training_data": result["training_data"]
        }

    seeds = parse_seeds(best_setting.get("seeds"))
    result, predictions = evaluate_on_validation(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        seq_len=int(best_setting["seq_len"]),
        params=params_from_best_setting(best_setting),
        validation_steps=VALIDATION_STEPS,
        seeds=seeds,
        return_predictions=True
    )

    return {
        "index": pd.DatetimeIndex(predictions["validation_index"]),
        "y_true": np.asarray(predictions["y_true"]),
        "seed_predictions": np.asarray(predictions["seed_predictions"]),
        "input_features": result["input_features"],
        "feature_count": result["feature_count"],
        "seq_len": result["seq_len"],
        "seeds": result["seeds"],
        "training_data": "train_only"
    }


def observation_fraction_for(index, quality_df):
    quality = quality_df.reindex(index)

    if quality["observation_fraction"].isna().any():
        raise ValueError(
            "the PV quality audit does not cover the full evaluation period"
        )

    return quality["observation_fraction"].to_numpy()


def seed_metric(seed_predictions, y_true, mask=None):
    """Mean and std across seeds of a per-seed metric tuple."""
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    values = np.asarray(
        [
            evaluate(y_true[mask], np.asarray(prediction)[mask])
            for prediction in seed_predictions
        ]
    )

    return values.mean(axis=0), values.std(axis=0, ddof=0)


def hourly_mae(index, y_true, y_pred):
    frame = pd.DataFrame(
        {
            "hour": pd.DatetimeIndex(index).hour,
            "error": np.abs(np.asarray(y_true) - np.asarray(y_pred))
        }
    )

    return frame.groupby("hour")["error"].mean().reindex(range(24)).to_numpy()


def summarize(variant, run, split, quality_df):
    y_true = run["y_true"]
    seed_predictions = run["seed_predictions"]
    ensemble = seed_predictions.mean(axis=0)
    quality_mask = (
        observation_fraction_for(run["index"], quality_df)
        >= PV_QUALITY_THRESHOLD
    )

    metrics, metric_std = seed_metric(seed_predictions, y_true)
    quality_metrics, _ = seed_metric(seed_predictions, y_true, quality_mask)
    ensemble_metrics = evaluate(y_true, ensemble)
    block_metrics = calculate_monthly_validation_metrics(
        index=run["index"],
        y_true=y_true,
        y_pred=ensemble
    )

    return {
        "variant": variant,
        "split": split,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "evaluation_start": run["index"][0],
        "evaluation_end": run["index"][-1],
        "evaluation_steps": len(y_true),
        "training_data": run["training_data"],
        "seq_len": run["seq_len"],
        "input_features": run["input_features"],
        "feature_count": run["feature_count"],
        "seeds": run["seeds"],
        "mape_production_threshold": MAPE_PRODUCTION_THRESHOLD,
        "pv_quality_threshold": PV_QUALITY_THRESHOLD,
        "high_quality_samples": int(quality_mask.sum()),
        "mae": metrics[0],
        "mae_std": metric_std[0],
        "rmse": metrics[1],
        "rmse_std": metric_std[1],
        "mape": metrics[2],
        "ensemble_mae": ensemble_metrics[0],
        "ensemble_rmse": ensemble_metrics[1],
        "high_quality_mae": quality_metrics[0],
        "high_quality_rmse": quality_metrics[1],
        "high_quality_mape": quality_metrics[2],
        "block_mae_mean": block_metrics["val_block_mae_mean"],
        "block_mae_std": block_metrics["val_block_mae_std"]
    }


def add_deltas(summary_df):
    baseline = summary_df[
        summary_df["variant"] == "with_time_features"
    ].iloc[0]

    for metric in ["mae", "high_quality_mae", "block_mae_mean", "rmse"]:
        summary_df[f"delta_{metric}"] = summary_df[metric] - baseline[metric]
        summary_df[f"delta_{metric}_percent"] = (
            100 * (summary_df[metric] - baseline[metric]) / baseline[metric]
        )

    return summary_df


def monthly_delta(runs, index_key="index"):
    """Per-month MAE for each variant, to expose seasonal dependence."""
    frames = []

    for variant, run in runs.items():
        ensemble = run["seed_predictions"].mean(axis=0)
        frame = pd.DataFrame(
            {
                "month": pd.DatetimeIndex(run[index_key]).to_period("M"),
                "error": np.abs(run["y_true"] - ensemble)
            }
        )
        monthly = frame.groupby("month")["error"].mean().rename(variant)
        frames.append(monthly)

    monthly_df = pd.concat(frames, axis=1)
    monthly_df["delta_mae"] = (
        monthly_df["without_time_features"]
        - monthly_df["with_time_features"]
    )
    monthly_df["delta_mae_percent"] = (
        100 * monthly_df["delta_mae"] / monthly_df["with_time_features"]
    )

    return monthly_df.reset_index()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_FILE_PATH)
    quality_df = load_pv_quality()
    best_setting = load_best_setting(args.model)

    runs = {}
    summaries = []
    hourly_errors = []
    predictions = {}

    for variant, keep_time_features in VARIANTS.items():
        print(f"\nrunning {args.model} [{variant}] on the {args.split} split")
        df_proc = prepare_variant(df, keep_time_features)
        print(f"  input features: {list(df_proc.columns)}")

        run = run_variant(
            model_name=args.model,
            df_proc=df_proc,
            best_setting=best_setting,
            split=args.split
        )
        runs[variant] = run
        summaries.append(
            summarize(variant, run, args.split, quality_df)
        )

        ensemble = run["seed_predictions"].mean(axis=0)
        hourly_errors.append(
            hourly_mae(run["index"], run["y_true"], ensemble)
        )
        predictions[variant] = pd.DataFrame(
            {
                "time": run["index"],
                "y_true": run["y_true"],
                "y_pred": ensemble
            }
        )

    prefix = f"{args.model}_periodicity_{args.split}"
    summary_df = add_deltas(pd.DataFrame(summaries))
    summary_path = RESULTS_DIR / f"{prefix}_ablation.csv"
    summary_df.to_csv(summary_path, index=False)

    hourly_df = pd.DataFrame(
        {
            "hour": range(24),
            "mae_with_time_features": hourly_errors[0],
            "mae_without_time_features": hourly_errors[1]
        }
    )
    hourly_df["delta_mae"] = (
        hourly_df["mae_without_time_features"]
        - hourly_df["mae_with_time_features"]
    )
    hourly_df.to_csv(RESULTS_DIR / f"{prefix}_hourly_mae.csv", index=False)

    monthly_df = monthly_delta(runs)
    monthly_df.to_csv(RESULTS_DIR / f"{prefix}_monthly_mae.csv", index=False)

    plot_periodicity_ablation(
        hourly_errors=hourly_errors,
        labels=list(VARIANTS.keys()),
        output_path=str(RESULTS_DIR / f"{prefix}_hourly_mae.png"),
        title=(
            f"{args.model} {args.split} MAE by hour of day, "
            "with and without hour_sin/hour_cos"
        )
    )
    plot_forecast_zoom_all_models(
        predictions_by_model=predictions,
        output_path=str(RESULTS_DIR / f"{prefix}_forecast_zoom.png"),
        start=0,
        hours=args.zoom_hours
    )

    columns = [
        "variant",
        "feature_count",
        "mae",
        "mae_std",
        "high_quality_mae",
        "block_mae_mean",
        "delta_mae_percent"
    ]
    print(
        f"\n{args.split} split "
        f"({summary_df['evaluation_start'].iloc[0]} to "
        f"{summary_df['evaluation_end'].iloc[0]}, "
        f"trained on {summary_df['training_data'].iloc[0]})"
    )
    print(summary_df[columns].round(3).to_string(index=False))
    print("\nper-month delta (positive = worse without time features):")
    print(
        monthly_df[
            ["month", "with_time_features", "without_time_features",
             "delta_mae", "delta_mae_percent"]
        ].round(3).to_string(index=False)
    )
    print(f"\nsaved {summary_path}")


if __name__ == "__main__":
    main()

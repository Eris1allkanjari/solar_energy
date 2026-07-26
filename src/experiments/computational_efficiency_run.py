import argparse
import platform
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.configs.config import ExperimentConfig
from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    PV_QUALITY_THRESHOLD,
    TEST_OFFSET,
    VALIDATION_END_RATIO
)
from src.data.loader import load_dataset
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import DATA_FILE_PATH, EFFICIENCY_RESULTS_DIR
from src.experiments.final_model_comparison_run import (
    AR_MODELS,
    BASELINE_MODELS,
    NEURAL_MODELS,
    attach_pv_quality,
    load_best_setting,
    load_pv_quality,
    run_baseline
)
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.ar_tuner import (
    ARExperimentConfig,
    final_test as final_ar_test,
    params_from_best_setting as ar_params_from_best_setting,
    prepare_ar_data,
    uses_exog
)
from src.parameter_tuning.tuner import (
    final_test as final_neural_test,
    params_from_best_setting as neural_params_from_best_setting,
    parse_seeds
)
from src.training.evaluation import evaluate


RESULTS_DIR = EFFICIENCY_RESULTS_DIR
MODEL_CHOICES = NEURAL_MODELS + AR_MODELS + BASELINE_MODELS
EXPERIMENT_PROTOCOL = "frozen_end_to_end_efficiency_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark frozen final-model protocols on a common test horizon."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=168,
        help="Common forecast horizon used only for computational benchmarking."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Use at least three repeats for thesis-reported timing variability."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help=(
            "RNN replicas included in each timed run. Use 42 123 2026 to "
            "benchmark the exact final ensemble."
        )
    )
    return parser.parse_args()


def validate_args(args):
    if args.test_steps <= 0:
        raise ValueError("test_steps must be positive")

    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    if not args.seeds:
        raise ValueError("at least one RNN seed is required")


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_metadata():
    return {
        "cpu_name": platform.processor() or "unknown_cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__
    }


def neural_parameter_count(model_name, best_setting, input_size):
    config = ExperimentConfig(
        params=neural_params_from_best_setting(best_setting),
        seq_len=int(best_setting["seq_len"])
    )
    model = get_model(
        model_name=model_name,
        input_shape=(int(best_setting["seq_len"]), input_size),
        config=config
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


def ar_parameter_count(model_name, best_setting, df):
    params = ar_params_from_best_setting(best_setting)
    config = ARExperimentConfig(
        params=params,
        window_length=int(best_setting["seq_len"])
    )
    splits = prepare_ar_data(df)
    y_train_final = pd.concat([splits[0], splits[1]], axis=0)
    history = y_train_final.tail(config.MAX_HISTORY)

    if uses_exog(params):
        exog_train_final = pd.concat([splits[3], splits[4]], axis=0)
        exog_history = exog_train_final[
            config.EXOG_FEATURES
        ].tail(config.MAX_HISTORY)
        model = get_ar_model_builder(model_name)(
            history,
            exog_history,
            config
        )
    else:
        model = get_ar_model_builder(model_name)(history, config)

    return int(model.k_params)


def selected_seed_count(best_setting):
    return len(parse_seeds(best_setting.get("seeds")))


def timed_call(function, neural=False):
    peak_gpu_memory_mb = np.nan

    if neural and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    synchronize_cuda()
    start = perf_counter()
    result = function()
    synchronize_cuda()
    elapsed_seconds = perf_counter() - start

    if neural and torch.cuda.is_available():
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    return result, float(elapsed_seconds), float(peak_gpu_memory_mb)


def metric_values(result, ensemble=False):
    if ensemble:
        return {
            "mae": result["ensemble_mae"],
            "rmse": result["ensemble_rmse"],
            "mape": result["ensemble_mape"],
            "smape": result["ensemble_smape"],
            "metric_aggregation": "seed_ensemble_mean"
        }

    return {
        "mae": result["mae"],
        "rmse": result["rmse"],
        "mape": result["mape"],
        "smape": result["smape"],
        "metric_aggregation": result.get(
            "metric_aggregation",
            "deterministic"
        )
    }


def quality_metric_values(result):
    observation_fraction = np.asarray(
        result["pv_observation_fraction"]
    )
    mask = observation_fraction >= PV_QUALITY_THRESHOLD
    y_true = np.asarray(result["y_test"])[mask]
    seed_predictions = result.get("seed_predictions")

    if seed_predictions is not None:
        metrics = np.asarray(
            [
                evaluate(
                    y_true,
                    np.asarray(seed_prediction)[mask]
                )
                for seed_prediction in seed_predictions
            ]
        ).mean(axis=0)
    else:
        metrics = evaluate(
            y_true,
            np.asarray(result["y_pred"])[mask]
        )
    return {
        "high_quality_samples": int(mask.sum()),
        "high_quality_coverage": float(mask.mean()),
        "high_quality_mae": metrics[0],
        "high_quality_rmse": metrics[1],
        "high_quality_mape": metrics[2],
        "high_quality_smape": metrics[3]
    }


def benchmark_neural(
    model_name,
    df_proc,
    quality_df,
    args,
    repeat,
    metadata
):
    selected_setting = load_best_setting(model_name)
    best_setting = dict(selected_setting)
    best_setting["seeds"] = ",".join(str(seed) for seed in args.seeds)

    if len(args.seeds) != selected_seed_count(selected_setting):
        best_setting.pop("best_epochs", None)

    parameter_count = neural_parameter_count(
        model_name,
        best_setting,
        input_size=len(df_proc.columns)
    )
    result, elapsed, peak_memory = timed_call(
        lambda: final_neural_test(
            model_name=model_name,
            get_model=get_model,
            df_proc=df_proc,
            best_setting=best_setting,
            test_steps=args.test_steps,
            test_offset=TEST_OFFSET
        ),
        neural=True
    )
    result = attach_pv_quality(result, quality_df)
    metrics = metric_values(result)

    return {
        "model": model_name,
        "model_family": "neural",
        "execution_device": "cuda" if torch.cuda.is_available() else "cpu",
        "execution_device_name": (
            metadata["cuda_device_name"]
            if torch.cuda.is_available()
            else metadata["cpu_name"]
        ),
        "repeat": repeat,
        "benchmark_seed_count": len(args.seeds),
        "selected_seed_count": selected_seed_count(selected_setting),
        "refit_epochs": result["refit_epochs"],
        "parameter_count": parameter_count,
        "estimated_parameter_storage_mb": parameter_count * 4 / (1024 ** 2),
        "peak_gpu_memory_mb": peak_memory,
        **metrics,
        **quality_metric_values(result),
        **metadata,
        "elapsed_seconds": elapsed
    }


def benchmark_ar(model_name, df, quality_df, args, repeat, metadata):
    best_setting = load_best_setting(model_name, autoregressive=True)
    parameter_count = ar_parameter_count(model_name, best_setting, df)
    result, elapsed, peak_memory = timed_call(
        lambda: final_ar_test(
            model_name=model_name,
            build_model=get_ar_model_builder(model_name),
            df=df,
            best_setting=best_setting,
            test_steps=args.test_steps,
            test_offset=TEST_OFFSET,
            include_validation_in_training=True,
            refit_interval=AR_REFIT_INTERVAL
        )
    )
    result = attach_pv_quality(result, quality_df)

    return {
        "model": model_name,
        "model_family": "autoregressive",
        "execution_device": "cpu",
        "execution_device_name": metadata["cpu_name"],
        "repeat": repeat,
        "benchmark_seed_count": 0,
        "selected_seed_count": 0,
        "refit_epochs": np.nan,
        "parameter_count": parameter_count,
        "estimated_parameter_storage_mb": parameter_count * 8 / (1024 ** 2),
        "peak_gpu_memory_mb": peak_memory,
        **metric_values(result),
        **quality_metric_values(result),
        **metadata,
        "elapsed_seconds": elapsed
    }


def benchmark_baseline(
    model_name,
    df_proc,
    quality_df,
    test_start_index,
    args,
    repeat,
    metadata
):
    result, elapsed, peak_memory = timed_call(
        lambda: run_baseline(
            model_name=model_name,
            df_proc=df_proc,
            test_start_index=test_start_index,
            test_steps=args.test_steps
        )
    )
    result = attach_pv_quality(result, quality_df)

    return {
        "model": model_name,
        "model_family": "baseline",
        "execution_device": "cpu",
        "execution_device_name": metadata["cpu_name"],
        "repeat": repeat,
        "benchmark_seed_count": 0,
        "selected_seed_count": 0,
        "refit_epochs": np.nan,
        "parameter_count": 0,
        "estimated_parameter_storage_mb": 0.0,
        "peak_gpu_memory_mb": peak_memory,
        **metric_values(result),
        **quality_metric_values(result),
        **metadata,
        "elapsed_seconds": elapsed
    }


def complete_row(row, result_index, args):
    row["experiment_protocol"] = EXPERIMENT_PROTOCOL
    row["timed_scope"] = "model_refit_and_forecast"
    row["test_offset"] = TEST_OFFSET
    row["test_steps"] = args.test_steps
    row["test_start"] = result_index[0]
    row["test_end"] = result_index[-1]
    row["seconds_per_forecast"] = (
        row["elapsed_seconds"] / args.test_steps
    )
    row["forecasts_per_second"] = (
        args.test_steps / row["elapsed_seconds"]
        if row["elapsed_seconds"] > 0 else np.inf
    )
    return row


def summarize(raw_df):
    rows = []

    for model_name, model_df in raw_df.groupby("model", sort=False):
        first = model_df.iloc[0]
        rows.append(
            {
                "model": model_name,
                "model_family": first["model_family"],
                "experiment_protocol": first["experiment_protocol"],
                "timed_scope": first["timed_scope"],
                "test_offset": first["test_offset"],
                "test_steps": first["test_steps"],
                "test_start": first["test_start"],
                "test_end": first["test_end"],
                "repeats": len(model_df),
                "benchmark_seed_count": first["benchmark_seed_count"],
                "selected_seed_count": first["selected_seed_count"],
                "refit_epochs": first["refit_epochs"],
                "parameter_count": first["parameter_count"],
                "estimated_parameter_storage_mb": first[
                    "estimated_parameter_storage_mb"
                ],
                "peak_gpu_memory_mb": model_df[
                    "peak_gpu_memory_mb"
                ].max(),
                "elapsed_seconds_mean": model_df[
                    "elapsed_seconds"
                ].mean(),
                "elapsed_seconds_std": model_df[
                    "elapsed_seconds"
                ].std(ddof=0),
                "seconds_per_forecast_mean": model_df[
                    "seconds_per_forecast"
                ].mean(),
                "forecasts_per_second_mean": model_df[
                    "forecasts_per_second"
                ].mean(),
                "mae": model_df["mae"].mean(),
                "rmse": model_df["rmse"].mean(),
                "mape": model_df["mape"].mean(),
                "smape": model_df["smape"].mean(),
                "high_quality_samples": first["high_quality_samples"],
                "high_quality_coverage": first["high_quality_coverage"],
                "high_quality_mae": model_df[
                    "high_quality_mae"
                ].mean(),
                "high_quality_rmse": model_df[
                    "high_quality_rmse"
                ].mean(),
                "high_quality_mape": model_df[
                    "high_quality_mape"
                ].mean(),
                "high_quality_smape": model_df[
                    "high_quality_smape"
                ].mean(),
                "metric_aggregation": first["metric_aggregation"],
                "execution_device": first["execution_device"],
                "execution_device_name": first["execution_device_name"],
                "cpu_name": first["cpu_name"],
                "cuda_available": first["cuda_available"],
                "cuda_device_name": first["cuda_device_name"],
                "platform": first["platform"],
                "python_version": first["python_version"],
                "torch_version": first["torch_version"]
            }
        )

    return pd.DataFrame(rows).sort_values("elapsed_seconds_mean")


def plot_summary(summary_df):
    figure, axis = plt.subplots(figsize=(9, 6))
    families = {
        "neural": "#1565c0",
        "autoregressive": "#ef6c00",
        "baseline": "#607d8b"
    }

    for _, row in summary_df.iterrows():
        axis.scatter(
            max(row["elapsed_seconds_mean"], 1e-6),
            row["mae"],
            color=families[row["model_family"]],
            s=70
        )
        axis.annotate(
            row["model"],
            (max(row["elapsed_seconds_mean"], 1e-6), row["mae"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    axis.set_xscale("log")
    axis.set_xlabel("End-to-end refit and forecast time (seconds, log scale)")
    axis.set_ylabel("MAE on the benchmark horizon")
    axis.set_title("Predictive accuracy versus computational cost")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "computational_efficiency.png",
        dpi=200
    )
    plt.close(figure)


def save_results(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(
        RESULTS_DIR / "computational_efficiency_runs.csv",
        index=False
    )
    summary_df = summarize(raw_df)
    summary_df.to_csv(
        RESULTS_DIR / "computational_efficiency_summary.csv",
        index=False
    )
    plot_summary(summary_df)


def main():
    args = parse_args()
    validate_args(args)
    metadata = device_metadata()
    df = load_dataset(DATA_FILE_PATH)
    df_proc = prepare_dataframe(df)
    quality_df = load_pv_quality()
    test_start_index = int(
        len(df_proc) * VALIDATION_END_RATIO
    ) + TEST_OFFSET
    benchmark_index = df_proc.index[
        test_start_index:test_start_index + args.test_steps
    ]

    if len(benchmark_index) < args.test_steps:
        raise RuntimeError("requested benchmark horizon exceeds the test split")

    rows = []

    for model_name in args.models:
        for repeat in range(1, args.repeats + 1):
            print(
                f"\nbenchmarking {model_name}, "
                f"repeat {repeat}/{args.repeats}"
            )

            if model_name in NEURAL_MODELS:
                row = benchmark_neural(
                    model_name,
                    df_proc,
                    quality_df,
                    args,
                    repeat,
                    metadata
                )
            elif model_name in AR_MODELS:
                row = benchmark_ar(
                    model_name,
                    df,
                    quality_df,
                    args,
                    repeat,
                    metadata
                )
            else:
                row = benchmark_baseline(
                    model_name,
                    df_proc,
                    quality_df,
                    test_start_index,
                    args,
                    repeat,
                    metadata
                )

            rows.append(
                complete_row(row, benchmark_index, args)
            )
            save_results(rows)

    print(f"\nsaved computational-efficiency results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

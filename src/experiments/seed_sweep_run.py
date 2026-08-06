"""Seed sweep for the recurrent models.

The final comparison trains three seeds, which is enough to report a mean but
too few to estimate the spread reliably or to decide whether the LSTM/GRU gap is
real. This script repeats the final test protocol over many seeds.

Each seed runs in its own interpreter via ``subprocess.run``. That does not
change the numbers - ``set_random_seed`` already seeds random, numpy, torch and
cuDNN - but it guarantees no PyTorch or CUDA state carries between runs, frees
GPU memory after every seed, and keeps one failed seed from aborting the sweep.
Completed seeds are cached, so an interrupted sweep resumes where it stopped.

    python -m src.experiments.seed_sweep_run --models gru lstm
    python -m src.experiments.seed_sweep_run --models gru --seeds 1 104 216
    python -m src.experiments.seed_sweep_run --aggregate-only

Each seed reuses the tuned hyperparameters and the median validation-selected
epoch count, because a newly drawn seed has no early-stopping epoch of its own.
The sweep therefore measures the effect of initialisation and batch order, not
of re-running the whole tuning procedure.
"""

import argparse
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd

from src.configs.evaluation import TEST_OFFSET, TEST_STEPS
from src.data.loader import load_dataset
from src.experiments.constants import (
    DATA_FILE_PATH,
    SEED_SWEEP_RESULTS_DIR
)
from src.experiments.final_model_comparison_run import (
    attach_pv_quality,
    load_best_setting,
    load_pv_quality
)
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.plots import plot_seed_sweep
from src.parameter_tuning.selection import calculate_monthly_validation_metrics
from src.parameter_tuning.tuner import final_test as final_neural_test
from src.training.evaluation import daylight_mape, evaluate


RESULTS_DIR = SEED_SWEEP_RESULTS_DIR
EXPERIMENT_PROTOCOL = "seed_sweep_final_test_v1"
DEFAULT_MODELS = ["gru", "lstm"]

# Fixed pool, kept explicit so a sweep is reproducible. Six seeds doubles the
# three used by the final protocol, which is enough to estimate the spread and
# to compare two models, while keeping the sweep short.
DEFAULT_SEEDS = [1, 104, 216, 314, 465, 555]

ENSEMBLE_SUBSET_LIMIT = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeat the final test protocol across many seeds."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["gru", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true",
                        help="Recompute seeds that already have a cached run.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only summarise the runs already on disk.")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def seed_paths(model_name, seed):
    directory = RESULTS_DIR / model_name
    return (
        directory / f"seed_{seed}_metrics.csv",
        directory / f"seed_{seed}_predictions.csv"
    )


def run_single_seed(model_name, seed):
    """Train and score one seed; called inside the worker process."""
    df = load_dataset(DATA_FILE_PATH)
    df_proc = prepare_dataframe(df)
    quality_df = load_pv_quality()

    best_setting = dict(load_best_setting(model_name))
    best_setting["seeds"] = str(seed)
    # A fresh seed has no early-stopping epoch of its own, so fall back to the
    # median tuned epoch count rather than a per-seed value from another seed.
    best_setting.pop("best_epochs", None)

    result = final_neural_test(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        best_setting=best_setting,
        test_steps=TEST_STEPS,
        test_offset=TEST_OFFSET
    )
    result = attach_pv_quality(result, quality_df)

    index = pd.DatetimeIndex(result["test_index"])
    y_true = np.asarray(result["y_test"])
    y_pred = np.asarray(result["y_pred"])
    quality_mask = np.asarray(result["pv_observation_fraction"]) >= 0.5

    mae, rmse, mape = evaluate(y_true, y_pred)
    quality_mae, quality_rmse, _ = evaluate(
        y_true[quality_mask],
        y_pred[quality_mask]
    )
    blocks = calculate_monthly_validation_metrics(index, y_true, y_pred)

    metrics = {
        "model": model_name,
        "seed": seed,
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "seq_len": result["seq_len"],
        "refit_epochs": result["refit_epochs"],
        "feature_count": result["feature_count"],
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "day_mape": daylight_mape(index, y_true, y_pred),
        "high_quality_mae": quality_mae,
        "high_quality_rmse": quality_rmse,
        "block_mae_mean": blocks["val_block_mae_mean"],
        "block_mae_std": blocks["val_block_mae_std"]
    }

    metrics_path, predictions_path = seed_paths(model_name, seed)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(
        {"time": index, "y_true": y_true, "y_pred": y_pred}
    ).to_csv(predictions_path, index=False)

    print(f"\nseed {seed}: MAE {mae:.4f}  RMSE {rmse:.4f}")


def dispatch(model_name, seed):
    """Run one seed in a fresh interpreter."""
    command = [
        sys.executable,
        "-m",
        "src.experiments.seed_sweep_run",
        "--worker",
        "--model",
        model_name,
        "--seed",
        str(seed)
    ]
    completed = subprocess.run(command)

    return completed.returncode == 0


def load_runs(model_name):
    directory = RESULTS_DIR / model_name

    if not directory.exists():
        return pd.DataFrame(), {}

    metrics = [
        pd.read_csv(path)
        for path in sorted(directory.glob("seed_*_metrics.csv"))
    ]
    predictions = {}

    for path in sorted(directory.glob("seed_*_predictions.csv")):
        seed = int(path.name.split("_")[1])
        predictions[seed] = pd.read_csv(path)["y_pred"].to_numpy()

    if not metrics:
        return pd.DataFrame(), predictions

    return pd.concat(metrics, ignore_index=True).sort_values("seed"), predictions


def ensemble_curve(predictions, y_true, rng):
    """Mean ensemble MAE as a function of the number of pooled seeds."""
    seeds = sorted(predictions)
    rows = []

    for size in range(1, len(seeds) + 1):
        combinations = list(itertools.combinations(seeds, size))

        if len(combinations) > ENSEMBLE_SUBSET_LIMIT:
            chosen = rng.choice(
                len(combinations),
                size=ENSEMBLE_SUBSET_LIMIT,
                replace=False
            )
            combinations = [combinations[position] for position in chosen]

        errors = [
            np.abs(
                y_true
                - np.mean([predictions[seed] for seed in combination], axis=0)
            ).mean()
            for combination in combinations
        ]
        rows.append(
            {
                "seeds_pooled": size,
                "ensemble_mae_mean": float(np.mean(errors)),
                "ensemble_mae_std": float(np.std(errors, ddof=0)),
                "subsets_evaluated": len(combinations)
            }
        )

    return pd.DataFrame(rows)


def welch_test(first, second):
    """Welch's t statistic and approximate two-sided p value."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    n1, n2 = len(first), len(second)

    if n1 < 2 or n2 < 2:
        return np.nan, np.nan

    v1, v2 = first.var(ddof=1) / n1, second.var(ddof=1) / n2

    if v1 + v2 == 0:
        return np.nan, np.nan

    t_statistic = (first.mean() - second.mean()) / np.sqrt(v1 + v2)
    degrees = (v1 + v2) ** 2 / (
        v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)
    )

    try:
        from scipy import stats

        p_value = 2 * stats.t.sf(abs(t_statistic), degrees)
    except ImportError:
        # Normal approximation is adequate once the sweep has enough seeds.
        p_value = 2 * (
            1 - 0.5 * (1 + np.math.erf(abs(t_statistic) / np.sqrt(2)))
        )

    return float(t_statistic), float(p_value)


def aggregate(models):
    summaries = []
    seed_maes = {}
    rng = np.random.default_rng(2026)

    for model_name in models:
        runs, predictions = load_runs(model_name)

        if runs.empty:
            print(f"no cached runs for {model_name}")
            continue

        y_true = None
        directory = RESULTS_DIR / model_name
        first_prediction_file = sorted(
            directory.glob("seed_*_predictions.csv")
        )

        if first_prediction_file:
            y_true = pd.read_csv(first_prediction_file[0])["y_true"].to_numpy()

        seed_maes[model_name] = runs["mae"].to_numpy()
        row = {
            "model": model_name,
            "seed_count": len(runs),
            "mae_mean": runs["mae"].mean(),
            "mae_std": runs["mae"].std(ddof=1),
            "mae_min": runs["mae"].min(),
            "mae_max": runs["mae"].max(),
            "mae_sem": runs["mae"].std(ddof=1) / np.sqrt(len(runs)),
            "rmse_mean": runs["rmse"].mean(),
            "rmse_std": runs["rmse"].std(ddof=1),
            "high_quality_mae_mean": runs["high_quality_mae"].mean(),
            "high_quality_mae_std": runs["high_quality_mae"].std(ddof=1),
            "block_mae_mean": runs["block_mae_mean"].mean()
        }

        if y_true is not None and predictions:
            curve = ensemble_curve(predictions, y_true, rng)
            curve.insert(0, "model", model_name)
            curve.to_csv(
                RESULTS_DIR / f"{model_name}_ensemble_curve.csv",
                index=False
            )
            row["full_ensemble_mae"] = curve.iloc[-1]["ensemble_mae_mean"]

        summaries.append(row)

    if not summaries:
        raise RuntimeError("no seed runs found; run the sweep first")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "seed_sweep_summary.csv", index=False)

    print("\n=== seed sweep summary ===")
    print(
        summary_df[
            ["model", "seed_count", "mae_mean", "mae_std", "mae_sem",
             "mae_min", "mae_max", "full_ensemble_mae"]
        ].round(4).to_string(index=False)
    )

    if len(seed_maes) == 2:
        (first_name, first), (second_name, second) = seed_maes.items()
        t_statistic, p_value = welch_test(first, second)
        difference = first.mean() - second.mean()
        print(
            f"\n{first_name} vs {second_name}: "
            f"mean MAE difference {difference:+.4f} "
            f"(t={t_statistic:.3f}, p={p_value:.4f})"
        )
        print(
            "  -> "
            + (
                "significant at 0.05"
                if p_value < 0.05
                else "not distinguishable at 0.05"
            )
        )
        pd.DataFrame(
            [
                {
                    "model_a": first_name,
                    "model_b": second_name,
                    "mae_difference": difference,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "seeds_a": len(first),
                    "seeds_b": len(second)
                }
            ]
        ).to_csv(RESULTS_DIR / "seed_sweep_model_test.csv", index=False)

    plot_seed_sweep(
        seed_maes=seed_maes,
        ensemble_curves={
            model_name: pd.read_csv(
                RESULTS_DIR / f"{model_name}_ensemble_curve.csv"
            )
            for model_name in seed_maes
            if (RESULTS_DIR / f"{model_name}_ensemble_curve.csv").exists()
        },
        output_path=str(RESULTS_DIR / "seed_sweep.png")
    )
    print(f"\nsaved results to {RESULTS_DIR}")


def main():
    args = parse_args()

    if args.worker:
        run_single_seed(args.model, args.seed)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for model_name in args.models:
            for seed in args.seeds:
                metrics_path, _ = seed_paths(model_name, seed)

                if metrics_path.exists() and not args.force:
                    print(f"skipping {model_name} seed {seed} (cached)")
                    continue

                print(f"\n=== {model_name}, seed {seed} ===")

                if not dispatch(model_name, seed):
                    print(
                        f"seed {seed} failed for {model_name}; continuing"
                    )

    aggregate(args.models)


if __name__ == "__main__":
    main()

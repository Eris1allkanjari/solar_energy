"""Compare training losses for the recurrent models.

The tuning grid fixes the loss to Huber with ``delta = 0.1``, so the loss is the
one hyperparameter that was never searched. That default has a concrete
consequence: the target is min-max scaled to [0, 1] over a range of roughly
260 kWh, so ``delta = 0.1`` is about 26 kWh. Every residual larger than that
falls in Huber's linear branch and is treated like MAE, which is precisely the
regime of peak production hours. The models under-predict peaks, and this
experiment tests whether the loss is responsible.

Losses are compared on the validation split, never on the test split, because
choosing a loss is a model-selection decision. The tuned architecture, window
length and seeds are held fixed so the loss is the only thing that varies.

    python -m src.experiments.loss_comparison_run
    python -m src.experiments.loss_comparison_run --models gru --seeds 42
"""

import argparse

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    MAPE_PRODUCTION_THRESHOLD,
    RNN_SEEDS,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.experiments.constants import (
    DATA_FILE_PATH,
    LOSS_COMPARISON_RESULTS_DIR
)
from src.experiments.final_model_comparison_run import load_best_setting
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.plots import plot_loss_comparison
from src.parameter_tuning.tuner import (
    evaluate_on_validation,
    params_from_best_setting
)


RESULTS_DIR = LOSS_COMPARISON_RESULTS_DIR
EXPERIMENT_PROTOCOL = "loss_comparison_validation_v1"

# delta is expressed in scaled target units; 0.1 is roughly 26 kWh and 0.3
# roughly 78 kWh, so the wider setting keeps peak-hour residuals quadratic.
LOSS_VARIANTS = {
    "huber_0.1": {"loss": "huber", "huber_delta": 0.1},
    "huber_0.3": {"loss": "huber", "huber_delta": 0.3},
    "mae": {"loss": "mae"},
    "mse": {"loss": "mse"}
}

PEAK_QUANTILE = 0.90
PRODUCTION_BINS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare training losses on the validation split."
    )
    parser.add_argument("--models", nargs="+", default=["gru", "lstm"],
                        choices=["gru", "lstm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(RNN_SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(LOSS_VARIANTS),
                        choices=list(LOSS_VARIANTS))
    return parser.parse_args()


def peak_diagnostics(y_true, y_pred):
    """Error and signed bias on the highest-production hours."""
    producing = y_true > MAPE_PRODUCTION_THRESHOLD

    if not producing.any():
        return {"peak_mae": np.nan, "peak_bias": np.nan, "overall_bias": np.nan}

    peak_threshold = np.quantile(y_true[producing], PEAK_QUANTILE)
    peak = y_true > peak_threshold

    return {
        "peak_threshold_kwh": float(peak_threshold),
        "peak_hours": int(peak.sum()),
        "peak_mae": float(np.abs(y_true[peak] - y_pred[peak]).mean()),
        # negative means the model systematically under-predicts peaks
        "peak_bias": float((y_pred[peak] - y_true[peak]).mean()),
        "overall_bias": float((y_pred - y_true).mean())
    }


def bias_by_production(y_true, y_pred):
    """Signed bias across production bins, from low output to peak."""
    producing = y_true > MAPE_PRODUCTION_THRESHOLD

    if not producing.any():
        return pd.DataFrame()

    edges = np.quantile(
        y_true[producing],
        np.linspace(0, 1, PRODUCTION_BINS + 1)
    )
    rows = []

    for position in range(PRODUCTION_BINS):
        low, high = edges[position], edges[position + 1]
        selected = (
            (y_true >= low) & (y_true <= high)
            if position == PRODUCTION_BINS - 1
            else (y_true >= low) & (y_true < high)
        )

        if not selected.any():
            continue

        rows.append(
            {
                "bin": position + 1,
                "production_low": float(low),
                "production_high": float(high),
                "hours": int(selected.sum()),
                "bias": float((y_pred[selected] - y_true[selected]).mean()),
                "mae": float(
                    np.abs(y_true[selected] - y_pred[selected]).mean()
                )
            }
        )

    return pd.DataFrame(rows)


def run_variant(model_name, df_proc, best_setting, variant, seeds):
    params = params_from_best_setting(best_setting)
    params.update(LOSS_VARIANTS[variant])

    result, predictions = evaluate_on_validation(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        seq_len=int(best_setting["seq_len"]),
        params=params,
        validation_steps=VALIDATION_STEPS,
        seeds=tuple(seeds),
        return_predictions=True
    )

    y_true = np.asarray(predictions["y_true"])
    y_pred = np.asarray(predictions["ensemble_prediction"])

    row = {
        "model": model_name,
        "loss_variant": variant,
        "loss": params["loss"],
        "huber_delta": params.get("huber_delta"),
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "seq_len": result["seq_len"],
        "seeds": result["seeds"],
        "best_epochs": result["best_epochs"],
        "val_block_mae_mean": result["val_block_mae_mean"],
        "val_block_mae_std": result["val_block_mae_std"],
        "val_mae": result["val_mae"],
        "val_rmse": result["val_rmse"],
        "val_mape": result["val_mape"],
        "val_seed_mae_std": result["val_seed_mae_std"],
        **peak_diagnostics(y_true, y_pred)
    }

    bins = bias_by_production(y_true, y_pred)
    bins.insert(0, "loss_variant", variant)
    bins.insert(0, "model", model_name)

    return row, bins


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_FILE_PATH)
    df_proc = prepare_dataframe(df)

    rows = []
    bin_frames = []

    for model_name in args.models:
        best_setting = load_best_setting(model_name)

        for variant in args.variants:
            print(f"\n=== {model_name}: {variant} ===")
            row, bins = run_variant(
                model_name=model_name,
                df_proc=df_proc,
                best_setting=best_setting,
                variant=variant,
                seeds=args.seeds
            )
            rows.append(row)
            bin_frames.append(bins)

            summary_df = pd.DataFrame(rows)
            summary_df.to_csv(
                RESULTS_DIR / "loss_comparison.csv",
                index=False
            )
            pd.concat(bin_frames, ignore_index=True).to_csv(
                RESULTS_DIR / "loss_comparison_bias_by_production.csv",
                index=False
            )

    summary_df = pd.DataFrame(rows)
    bins_df = pd.concat(bin_frames, ignore_index=True)

    for model_name in args.models:
        model_rows = summary_df[summary_df["model"] == model_name]

        if model_rows.empty:
            continue

        plot_loss_comparison(
            summary_df=model_rows,
            bins_df=bins_df[bins_df["model"] == model_name],
            output_path=str(
                RESULTS_DIR / f"{model_name}_loss_comparison.png"
            ),
            title=f"{model_name}: training loss comparison (validation)"
        )

    columns = [
        "model",
        "loss_variant",
        "val_block_mae_mean",
        "val_mae",
        "val_rmse",
        "peak_mae",
        "peak_bias",
        "overall_bias"
    ]
    print("\n=== loss comparison (validation split) ===")
    print(summary_df[columns].round(3).to_string(index=False))
    print(f"\nsaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_mae_by_l(best_per_l, output_path=None):
    # plot best validation mae for each sequence length

    results_df = pd.DataFrame(best_per_l)
    metric = (
        "val_block_mae_mean"
        if "val_block_mae_mean" in results_df.columns
        else "val_mae"
    )

    results_df = results_df.sort_values(
        "seq_len"
    )

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["seq_len"],
        results_df[metric],
        marker="o"
    )

    plt.xlabel("input window length L")
    plt.ylabel(metric)
    plt.title(f"{metric} by input window length")
    plt.grid(True)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300
        )

    plt.close(figure)


def plot_real_vs_predicted(
    y_true,
    y_pred,
    output_path=None,
    title="actual vs predicted pv production",
    max_points=None,
    show=True
):
    # plot actual and predicted values over time

    if max_points is not None:
        y_true = y_true[:max_points]
        y_pred = y_pred[:max_points]

    figure = plt.figure(figsize=(12, 6))

    plt.plot(
        y_true,
        label="actual"
    )

    plt.plot(
        y_pred,
        label="predicted"
    )

    plt.xlabel("time step")
    plt.ylabel("pv production")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300
        )

    if show:
        plt.show()

    plt.close(figure)


def plot_feature_ablation(results, output_path=None, show=False):
    results_df = pd.DataFrame(results).sort_values(
        "val_block_mae_mean"
    )

    figure = plt.figure(figsize=(10, 6))
    plt.barh(
        results_df["feature_set"],
        results_df["val_block_mae_mean"]
    )
    plt.xlabel("mean monthly validation MAE")
    plt.ylabel("feature set")
    plt.title(
        f"feature ablation for {results_df['model'].iloc[0]}"
    )
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )
        plt.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)

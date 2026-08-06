import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_STYLES = {
    "lstm": ("#1f77b4", "o"),
    "gru": ("#ff7f0e", "s"),
    "arima": ("#2ca02c", "^"),
    "sarima": ("#d62728", "v"),
    "arimax": ("#9467bd", "D"),
    "sarimax": ("#8c564b", "P")
}


def ensure_output_dir(output_path):
    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)


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


def plot_mae_by_l_all_models(
    results_df,
    output_path=None,
    metric="val_block_mae_mean",
    show=False
):
    """Validation MAE against input window length L for every model.

    The neural windows (24-168 h) and the autoregressive windows (168-1440 h)
    differ by an order of magnitude, so L is drawn on a log axis to keep both
    families readable on one set of axes. The selected L per model is marked.
    """
    figure, axes = plt.subplots(figsize=(10, 6))

    for model_name, model_df in results_df.groupby("model", sort=False):
        model_df = model_df.sort_values("seq_len")
        color, marker = MODEL_STYLES.get(model_name, (None, "o"))

        axes.plot(
            model_df["seq_len"],
            model_df[metric],
            marker=marker,
            color=color,
            label=model_name,
            linewidth=1.8,
            markersize=6
        )

        best_row = model_df.loc[model_df[metric].idxmin()]
        axes.scatter(
            best_row["seq_len"],
            best_row[metric],
            marker="*",
            s=260,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=5
        )

    axes.set_xscale("log", base=2)
    axes.set_xticks(sorted(results_df["seq_len"].unique()))
    axes.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{int(value)}")
    )
    axes.set_xlabel("input window length L (hours, log scale)")
    axes.set_ylabel("mean monthly validation MAE (kWh)")
    axes.set_title(
        "validation MAE by input window length (star = selected L)"
    )
    axes.grid(True, alpha=0.3)
    axes.legend(title="model", ncol=2)
    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_forecast_zoom(
    time,
    y_true,
    y_pred,
    output_path=None,
    title="actual vs predicted",
    start=0,
    hours=72,
    show=False
):
    """Zoomed actual-vs-predicted view on the test set.

    The top panel keeps the full test period for context and shades the zoom
    window; the bottom panel is the horizontal zoom (a few days instead of the
    whole test set) with the vertical axis rescaled to that window, so the
    hour-by-hour tracking error is actually visible.
    """
    time = pd.DatetimeIndex(time)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    end = min(start + hours, len(y_true))
    window = slice(start, end)

    figure, (context_axes, zoom_axes) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 2]}
    )

    context_axes.plot(time, y_true, color="#1f77b4", linewidth=0.5)
    context_axes.axvspan(
        time[start],
        time[end - 1],
        color="orange",
        alpha=0.35
    )
    context_axes.set_ylabel("pv production (kWh)")
    context_axes.set_title(f"{title} - full test period, zoom window shaded")
    context_axes.grid(True, alpha=0.3)

    zoom_axes.plot(
        time[window],
        y_true[window],
        label="actual",
        color="#1f77b4",
        linewidth=2,
        marker="o",
        markersize=3
    )
    zoom_axes.plot(
        time[window],
        y_pred[window],
        label="predicted",
        color="#ff7f0e",
        linewidth=2,
        marker="s",
        markersize=3
    )
    zoom_axes.fill_between(
        time[window],
        y_true[window],
        y_pred[window],
        color="red",
        alpha=0.15,
        label="error"
    )

    # vertical zoom: rescale to the visible window instead of the full range
    visible = np.concatenate([y_true[window], y_pred[window]])
    margin = max(0.05 * (visible.max() - visible.min()), 1.0)
    zoom_axes.set_ylim(
        min(visible.min() - margin, 0),
        visible.max() + margin
    )

    zoom_axes.set_xlabel("time")
    zoom_axes.set_ylabel("pv production (kWh)")
    zoom_axes.set_title(
        f"zoom: {time[start]:%Y-%m-%d %H:%M} to {time[end - 1]:%Y-%m-%d %H:%M}"
        f" ({end - start} hours)"
    )
    zoom_axes.grid(True, alpha=0.3)
    zoom_axes.legend()
    figure.autofmt_xdate()
    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_forecast_zoom_all_models(
    predictions_by_model,
    output_path=None,
    start=0,
    hours=72,
    show=False
):
    """One zoom window, every model overlaid against the same actual series."""
    figure, axes = plt.subplots(figsize=(13, 7))
    first = True

    for model_name, frame in predictions_by_model.items():
        time = pd.DatetimeIndex(frame["time"])
        end = min(start + hours, len(frame))
        window = slice(start, end)

        if first:
            axes.plot(
                time[window],
                frame["y_true"].to_numpy()[window],
                label="actual",
                color="black",
                linewidth=3,
                zorder=1
            )
            first = False

        color, marker = MODEL_STYLES.get(model_name, (None, "o"))
        axes.plot(
            time[window],
            frame["y_pred"].to_numpy()[window],
            label=model_name,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.4,
            alpha=0.9
        )

    axes.set_xlabel("time")
    axes.set_ylabel("pv production (kWh)")
    axes.set_title("actual vs predicted on the test set, all models (zoom)")
    axes.grid(True, alpha=0.3)
    axes.legend(ncol=3)
    figure.autofmt_xdate()
    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_periodicity_ablation(
    hourly_errors,
    labels,
    output_path=None,
    title="effect of removing the time-of-day features",
    show=False
):
    """Mean absolute error by hour of day, with and without time features.

    A periodicity ablation should show its effect as a function of the time of
    day, so plotting MAE per hour makes visible whether the loss concentrates
    around sunrise/sunset (when knowing the clock matters most) rather than
    being spread evenly.
    """
    figure, (error_axes, delta_axes) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    hours = np.arange(24)

    for label, errors in zip(labels, hourly_errors):
        error_axes.plot(
            hours,
            errors,
            marker="o",
            markersize=4,
            label=label
        )

    error_axes.set_ylabel("MAE (kWh)")
    error_axes.set_title(title)
    error_axes.grid(True, alpha=0.3)
    error_axes.legend()

    delta = np.asarray(hourly_errors[1]) - np.asarray(hourly_errors[0])
    colors = ["#d62728" if value > 0 else "#2ca02c" for value in delta]
    delta_axes.bar(hours, delta, color=colors)
    delta_axes.axhline(0, color="black", linewidth=0.8)
    delta_axes.set_xlabel("hour of day")
    delta_axes.set_ylabel("MAE change (kWh)")
    delta_axes.set_title(
        "positive = worse without the time features"
    )
    delta_axes.set_xticks(hours)
    delta_axes.grid(True, alpha=0.3, axis="y")

    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_seed_sweep(
    seed_maes,
    ensemble_curves=None,
    output_path=None,
    show=False
):
    """Spread of test MAE across seeds, and the benefit of pooling them.

    The left panel shows every individual seed against the model mean, which is
    what decides whether a gap between two models is larger than the noise from
    initialisation alone. The right panel shows how far the seed ensemble
    improves as more seeds are pooled, which is what justifies the number of
    seeds used in the final protocol.
    """
    panels = 2 if ensemble_curves else 1
    figure, axes = plt.subplots(
        1,
        panels,
        figsize=(6.5 * panels, 5.5),
        squeeze=False
    )
    spread_axes = axes[0][0]

    for position, (model_name, values) in enumerate(seed_maes.items()):
        values = np.asarray(values, dtype=float)
        color, marker = MODEL_STYLES.get(model_name, (None, "o"))
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(values))

        spread_axes.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=color,
            marker=marker,
            s=55,
            alpha=0.85,
            label=f"{model_name} seeds"
        )
        spread_axes.hlines(
            values.mean(),
            position - 0.25,
            position + 0.25,
            color="black",
            linewidth=2
        )

        if len(values) > 1:
            spread_axes.errorbar(
                position,
                values.mean(),
                yerr=values.std(ddof=1),
                color="black",
                capsize=6,
                linewidth=1.4
            )

    spread_axes.set_xticks(range(len(seed_maes)))
    spread_axes.set_xticklabels(list(seed_maes))
    spread_axes.set_ylabel("test MAE (kWh)")
    spread_axes.set_title("per-seed test MAE (bar = mean, whisker = 1 sd)")
    spread_axes.grid(True, alpha=0.3, axis="y")

    if ensemble_curves:
        curve_axes = axes[0][1]

        for model_name, curve in ensemble_curves.items():
            color, marker = MODEL_STYLES.get(model_name, (None, "o"))
            curve_axes.plot(
                curve["seeds_pooled"],
                curve["ensemble_mae_mean"],
                marker=marker,
                color=color,
                label=model_name
            )
            curve_axes.fill_between(
                curve["seeds_pooled"],
                curve["ensemble_mae_mean"] - curve["ensemble_mae_std"],
                curve["ensemble_mae_mean"] + curve["ensemble_mae_std"],
                color=color,
                alpha=0.15
            )

        curve_axes.set_xlabel("seeds pooled in the ensemble")
        curve_axes.set_ylabel("ensemble test MAE (kWh)")
        curve_axes.set_title("ensemble MAE against number of seeds")
        curve_axes.grid(True, alpha=0.3)
        curve_axes.legend()

    spread_axes.legend()
    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_loss_comparison(
    summary_df,
    bins_df,
    output_path=None,
    title="training loss comparison",
    show=False
):
    """Accuracy against peak bias for each candidate training loss.

    The left panel is the selection metric, the middle panel is the signed bias
    on the highest-production hours, and the right panel traces bias across
    production levels. A loss that lowers peak under-prediction without raising
    validation MAE is the one worth adopting.
    """
    figure, (mae_axes, bias_axes, curve_axes) = plt.subplots(
        1,
        3,
        figsize=(17, 5.5)
    )

    variants = summary_df["loss_variant"].tolist()
    positions = np.arange(len(variants))

    mae_axes.bar(
        positions,
        summary_df["val_block_mae_mean"],
        color="#1f77b4"
    )
    mae_axes.set_xticks(positions)
    mae_axes.set_xticklabels(variants, rotation=20)
    mae_axes.set_ylabel("mean monthly validation MAE (kWh)")
    mae_axes.set_title("accuracy (lower is better)")
    mae_axes.grid(True, alpha=0.3, axis="y")

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in summary_df["peak_bias"]]
    bias_axes.bar(positions, summary_df["peak_bias"], color=colors)
    bias_axes.axhline(0, color="black", linewidth=0.8)
    bias_axes.set_xticks(positions)
    bias_axes.set_xticklabels(variants, rotation=20)
    bias_axes.set_ylabel("bias on peak hours (kWh)")
    bias_axes.set_title("negative = under-predicting peaks")
    bias_axes.grid(True, alpha=0.3, axis="y")

    for variant, frame in bins_df.groupby("loss_variant", sort=False):
        curve_axes.plot(
            frame["bin"],
            frame["bias"],
            marker="o",
            label=variant
        )

    curve_axes.axhline(0, color="black", linewidth=0.8)
    curve_axes.set_xlabel("production level (low to peak)")
    curve_axes.set_ylabel("bias (kWh)")
    curve_axes.set_title("bias across production levels")
    curve_axes.set_xticks(sorted(bins_df["bin"].unique()))
    curve_axes.grid(True, alpha=0.3)
    curve_axes.legend()

    figure.suptitle(title)
    figure.tight_layout()

    if output_path is not None:
        ensure_output_dir(output_path)
        figure.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(figure)


def plot_feature_ablation(results, output_path=None, show=False):
    results_df = pd.DataFrame(results)

    if "accepted" in results_df.columns:
        accepted = results_df["accepted"].astype(str).str.lower().eq("true")
        results_df = results_df[accepted]

    if results_df.empty:
        return

    results_df = results_df.sort_values(
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

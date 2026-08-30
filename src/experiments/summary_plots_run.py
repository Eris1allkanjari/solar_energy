import argparse

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.experiments.constants import (
    AR_TUNING_RESULTS_DIR,
    DATA_FILE_PATH,
    FINAL_COMPARISON_RESULTS_DIR,
    NEURAL_TUNING_RESULTS_DIR,
    SUMMARY_RESULTS_DIR
)
from src.experiments.final_model_comparison_run import AR_MODELS, NEURAL_MODELS
from src.parameter_tuning.plots import (
    plot_forecast_zoom,
    plot_forecast_zoom_all_models,
    plot_mae_by_l_all_models
)
from src.parameter_tuning.selection import (
    SELECTION_MAE_KEY,
    SELECTION_RMSE_KEY,
    select_robust_candidate
)
from src.experiments.parameter_tuning_run import prepare_dataframe
from src.utils.capacity import training_peak_capacity


SELECTION_METRIC = SELECTION_MAE_KEY


def scale_variants(capacity_kwh):
    """Filename suffix and divisor for each version of a figure.

    Every figure is produced twice: once in kWh and once scaled to a percentage
    of the training peak. The scaled version makes the error comparable with
    published results from other sites, while the kWh version stays readable
    against the raw data, so neither replaces the other.
    """
    return [
        ("", None),
        ("_scaled", capacity_kwh)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build cross-model summary figures from existing tuning and "
            "final-comparison results. Runs no training."
        )
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=72,
        help="Width of the zoom window in hours (default: 72, i.e. three days)."
    )
    parser.add_argument(
        "--zoom-start",
        default=None,
        help=(
            "Timestamp starting the zoom window, e.g. 2017-06-12. "
            "Defaults to the most variable window in the test period."
        )
    )
    return parser.parse_args()


def load_best_per_l():
    frames = []

    for model_name in NEURAL_MODELS:
        path = NEURAL_TUNING_RESULTS_DIR / f"{model_name}_best_per_l.csv"

        if not path.exists():
            print(f"skipping {model_name}: missing {path}")
            continue

        frame = pd.read_csv(path)
        frame["model"] = model_name
        frames.append(
            frame[["model", "seq_len", SELECTION_METRIC, SELECTION_RMSE_KEY]]
        )

    for model_name in AR_MODELS:
        path = AR_TUNING_RESULTS_DIR / f"{model_name}_ar_best_per_l.csv"

        if not path.exists():
            print(f"skipping {model_name}: missing {path}")
            continue

        frame = pd.read_csv(path)
        frame["model"] = model_name
        frames.append(
            frame[["model", "seq_len", SELECTION_METRIC, SELECTION_RMSE_KEY]]
        )

    if not frames:
        raise FileNotFoundError(
            "no tuning results found. Run the parameter-tuning experiments "
            "first."
        )

    return pd.concat(frames, axis=0, ignore_index=True).dropna(
        subset=[SELECTION_METRIC]
    )


def load_test_predictions():
    predictions = {}

    for model_name in NEURAL_MODELS + AR_MODELS:
        path = (
            FINAL_COMPARISON_RESULTS_DIR
            / f"{model_name}_final_test_predictions.csv"
        )

        if not path.exists():
            print(f"skipping {model_name}: missing {path}")
            continue

        frame = pd.read_csv(path)
        frame["time"] = pd.to_datetime(frame["time"])
        predictions[model_name] = frame

    if not predictions:
        raise FileNotFoundError(
            "no final-test predictions found. Run the final model comparison "
            "first."
        )

    return predictions


def most_variable_window(frame, hours):
    """Index of the window with the largest hour-to-hour movement.

    A flat clear-sky stretch shows nothing interesting, so the default zoom
    lands where the models actually have to work.
    """
    ramp = np.abs(np.diff(frame["y_true"].to_numpy(), prepend=0.0))
    rolling = pd.Series(ramp).rolling(hours).sum()

    if rolling.isna().all():
        return 0

    return max(int(rolling.idxmax()) - hours + 1, 0)


def resolve_zoom_start(frame, args):
    if args.zoom_start is None:
        return most_variable_window(frame, args.hours)

    target = pd.Timestamp(args.zoom_start)
    position = frame["time"].searchsorted(target)

    if position >= len(frame):
        raise ValueError(
            f"zoom start {target} is after the end of the test period"
        )

    return int(position)


def main():
    args = parse_args()
    SUMMARY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Same divisor the final comparison uses, so a percentage read off a figure
    # matches the nmae column in the results.
    capacity_kwh = training_peak_capacity(
        prepare_dataframe(load_dataset(DATA_FILE_PATH))
    )
    variants = scale_variants(capacity_kwh)
    print(f"scaling reference: training peak {capacity_kwh:.1f} kWh")

    best_per_l = load_best_per_l()

    # Use the same rule the final comparison uses, so the marked L is the one
    # actually carried into the results rather than the raw minimum.
    selected_l = {
        model_name: int(
            select_robust_candidate(model_df.to_dict(orient="records"))[
                "seq_len"
            ]
        )
        for model_name, model_df in best_per_l.groupby("model", sort=False)
    }

    neural_family = {
        "title": "recurrent models",
        "xlabel": "input sequence length L (hours, log scale)",
        "models": NEURAL_MODELS
    }
    ar_family = {
        "title": "autoregressive models",
        "xlabel": "rolling fit window (hours, log scale)",
        "models": AR_MODELS
    }
    combined_family = {
        "title": "all models",
        "xlabel": "window length (hours, log scale)",
        "models": NEURAL_MODELS + AR_MODELS
    }

    # The combined figure keeps both families on one axis for a direct accuracy
    # comparison; the per-family figures avoid overloading the x-axis, which
    # means input sequence length for the recurrent models and rolling fit
    # window for the autoregressive ones.
    # The side-by-side variant shares the y-axis, so each family keeps its own
    # correctly labelled x-axis while the accuracy levels stay comparable and
    # the recurrent panel is not auto-zoomed into its own noise.
    window_figures = [
        ("validation_mae_by_window_length.png", [combined_family]),
        (
            "validation_mae_by_window_length_by_family.png",
            [neural_family, ar_family]
        ),
        ("validation_mae_by_window_length_recurrent.png", [neural_family]),
        ("validation_mae_by_window_length_autoregressive.png", [ar_family])
    ]

    for file_name, families in window_figures:
        for suffix, capacity in variants:
            window_path = SUMMARY_RESULTS_DIR / file_name.replace(
                ".png",
                f"{suffix}.png"
            )
            plot_mae_by_l_all_models(
                results_df=best_per_l,
                families=families,
                output_path=str(window_path),
                metric=SELECTION_METRIC,
                selected_l=selected_l,
                capacity_kwh=capacity
            )
            print(f"saved {window_path}")
    print("\nselected window length per model:")
    for model_name, length in selected_l.items():
        row = best_per_l[
            (best_per_l["model"] == model_name)
            & (best_per_l["seq_len"] == length)
        ].iloc[0]
        minimum = best_per_l[best_per_l["model"] == model_name][
            SELECTION_METRIC
        ].min()
        note = "" if np.isclose(row[SELECTION_METRIC], minimum) else \
            f"  (raw minimum is {minimum:.4f}; within tolerance, lower RMSE won)"
        print(
            f"  {model_name:10s} L={length:<5d} "
            f"{SELECTION_METRIC}={row[SELECTION_METRIC]:.4f}{note}"
        )

    predictions = load_test_predictions()
    reference = next(iter(predictions.values()))
    start = resolve_zoom_start(reference, args)
    end = min(start + args.hours, len(reference))
    print(
        f"\nzoom window: {reference['time'].iloc[start]} to "
        f"{reference['time'].iloc[end - 1]} ({end - start} hours)"
    )

    for model_name, frame in predictions.items():
        for suffix, capacity in variants:
            output_path = (
                SUMMARY_RESULTS_DIR
                / f"{model_name}_test_forecast_zoom{suffix}.png"
            )
            plot_forecast_zoom(
                time=frame["time"],
                y_true=frame["y_true"],
                y_pred=frame["y_pred"],
                output_path=str(output_path),
                title=f"{model_name} on the test set",
                start=start,
                hours=args.hours,
                capacity_kwh=capacity
            )
            print(f"saved {output_path}")

    for suffix, capacity in variants:
        combined_path = (
            SUMMARY_RESULTS_DIR
            / f"test_forecast_zoom_all_models{suffix}.png"
        )
        plot_forecast_zoom_all_models(
            predictions_by_model=predictions,
            output_path=str(combined_path),
            start=start,
            hours=args.hours,
            capacity_kwh=capacity
        )
        print(f"saved {combined_path}")


if __name__ == "__main__":
    main()

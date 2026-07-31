import argparse

import numpy as np
import pandas as pd

from src.experiments.constants import (
    AR_TUNING_RESULTS_DIR,
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


SELECTION_METRIC = "val_block_mae_mean"


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
        frames.append(frame[["model", "seq_len", SELECTION_METRIC]])

    for model_name in AR_MODELS:
        path = AR_TUNING_RESULTS_DIR / f"{model_name}_ar_best_per_l.csv"

        if not path.exists():
            print(f"skipping {model_name}: missing {path}")
            continue

        frame = pd.read_csv(path)
        frame["model"] = model_name
        frames.append(frame[["model", "seq_len", SELECTION_METRIC]])

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

    best_per_l = load_best_per_l()
    window_path = SUMMARY_RESULTS_DIR / "validation_mae_by_window_length.png"
    plot_mae_by_l_all_models(
        results_df=best_per_l,
        output_path=str(window_path),
        metric=SELECTION_METRIC
    )
    best_l = (
        best_per_l.loc[best_per_l.groupby("model")[SELECTION_METRIC].idxmin()]
        .sort_values(SELECTION_METRIC)
    )
    print("\nselected window length per model:")
    print(best_l.to_string(index=False))
    print(f"\nsaved {window_path}")

    predictions = load_test_predictions()
    reference = next(iter(predictions.values()))
    start = resolve_zoom_start(reference, args)
    end = min(start + args.hours, len(reference))
    print(
        f"\nzoom window: {reference['time'].iloc[start]} to "
        f"{reference['time'].iloc[end - 1]} ({end - start} hours)"
    )

    for model_name, frame in predictions.items():
        output_path = (
            SUMMARY_RESULTS_DIR / f"{model_name}_test_forecast_zoom.png"
        )
        plot_forecast_zoom(
            time=frame["time"],
            y_true=frame["y_true"],
            y_pred=frame["y_pred"],
            output_path=str(output_path),
            title=f"{model_name} on the test set",
            start=start,
            hours=args.hours
        )
        print(f"saved {output_path}")

    combined_path = SUMMARY_RESULTS_DIR / "test_forecast_zoom_all_models.png"
    plot_forecast_zoom_all_models(
        predictions_by_model=predictions,
        output_path=str(combined_path),
        start=start,
        hours=args.hours
    )
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()

import argparse
import time

import pandas as pd

from src.configs.dataset_selection import ACTIVE_DATASET, add_dataset_argument
from src.configs.evaluation import AR_REFIT_INTERVAL, VALIDATION_STEPS
from src.data.loader import load_dataset
from src.experiments.constants import AR_TUNING_RESULTS_DIR, DATA_FILE_PATH
from src.models.ar.arima import build_arima
from src.models.ar.arimax import build_arimax
from src.models.ar.sarima import build_sarima
from src.models.ar.sarimax import build_sarimax
from src.parameter_tuning.ar_parameter_grid import (
    AR_HYPERPARAMETER_GRIDS,
    AR_WINDOW_LENGTHS_BY_MODEL
)
from src.parameter_tuning.ar_tuner import (
    evaluate_on_validation,
    generate_param_combinations,
    prepare_ar_data,
    select_best_l,
    tune_ar_model
)
from src.parameter_tuning.plots import plot_mae_by_l


RESULTS_DIR = AR_TUNING_RESULTS_DIR
AR_MODEL_CHOICES = ["arima", "sarima", "arimax", "sarimax"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune autoregressive forecasting models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AR_MODEL_CHOICES,
        default=AR_MODEL_CHOICES,
        help="Models to tune; results are saved after each model."
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help=(
            "Time one configuration per model over this many validation "
            "steps and project the full tuning time, writing nothing. Run it "
            "before committing to a long grid."
        )
    )
    add_dataset_argument(parser)
    return parser.parse_args()


def get_ar_model_builder(model_name):
    builders = {
        "arima": build_arima,
        "sarima": build_sarima,
        "arimax": build_arimax,
        "sarimax": build_sarimax
    }

    try:
        return builders[model_name]
    except KeyError as error:
        raise ValueError(
            f"unknown autoregressive model: {model_name}"
        ) from error


def format_duration(seconds):
    if seconds < 120:
        return f"{seconds:.1f} s"

    if seconds < 2 * 3600:
        return f"{seconds / 60:.1f} min"

    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"

    return f"{seconds / 86400:.1f} days"


def probe_model(model_name, df, steps):
    """Time one configuration and project the full grid, writing nothing.

    A fit happens once and every later step is a state update, so the probe
    times a single step and then many: the difference isolates the per-step
    cost, and the projection does not spread one slow fit over a few steps.
    The longest window fits slowest, which keeps the estimate conservative.
    """
    if steps < 2:
        raise ValueError("--probe-steps must be at least 2")

    y_train, y_val, _, exog_train, exog_val, exog_test = prepare_ar_data(df)
    windows = AR_WINDOW_LENGTHS_BY_MODEL[model_name]
    combinations = list(
        generate_param_combinations(AR_HYPERPARAMETER_GRIDS[model_name])
    )
    window_length = max(windows)
    params = combinations[0]

    def elapsed(validation_steps):
        start = time.perf_counter()
        evaluate_on_validation(
            model_name=model_name,
            build_model=get_ar_model_builder(model_name),
            y_train=y_train,
            y_val=y_val,
            exog_train=exog_train,
            exog_val=exog_val,
            exog_test=exog_test,
            window_length=window_length,
            params=params,
            validation_steps=validation_steps,
            refit_interval=AR_REFIT_INTERVAL
        )
        return time.perf_counter() - start

    print(
        f"\nprobing {model_name} on {ACTIVE_DATASET.name}: window "
        f"{window_length} steps, {params}"
    )
    fit_seconds = elapsed(1)
    many_seconds = elapsed(steps)
    per_step = max(many_seconds - fit_seconds, 0.0) / (steps - 1)
    per_configuration = fit_seconds + per_step * VALIDATION_STEPS
    configuration_count = len(windows) * len(combinations)

    print(f"  fit and first step       : {format_duration(fit_seconds)}")
    print(f"  each further step        : {per_step * 1000:.1f} ms")
    print(
        f"  one configuration        : {format_duration(per_configuration)} "
        f"over {VALIDATION_STEPS} validation steps"
    )
    print(
        f"  full grid, {configuration_count:2d} configurations: "
        f"{format_duration(per_configuration * configuration_count)}"
    )


def run_tuning_for_ar_model(model_name, df):
    print(
        f"\nstarting autoregressive tuning for {model_name}"
    )

    checkpoint_path = RESULTS_DIR / f"{model_name}_ar_tuning_checkpoint.csv"
    all_results, best_per_l = tune_ar_model(
        model_name=model_name,
        build_model=get_ar_model_builder(model_name),
        df=df,
        param_grid=AR_HYPERPARAMETER_GRIDS[model_name],
        window_lengths=AR_WINDOW_LENGTHS_BY_MODEL[model_name],
        validation_steps=VALIDATION_STEPS,
        checkpoint_path=checkpoint_path
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results_path = RESULTS_DIR / (
        f"{model_name}_ar_all_tuning_results.csv"
    )
    pd.DataFrame(all_results).to_csv(
        all_results_path,
        index=False
    )
    # The full grid is now saved, so the checkpoint has served its purpose; a
    # later run should start afresh rather than resume a finished one.
    checkpoint_path.unlink(missing_ok=True)

    best_per_l_path = RESULTS_DIR / (
        f"{model_name}_ar_best_per_l.csv"
    )
    pd.DataFrame(best_per_l).to_csv(
        best_per_l_path,
        index=False
    )

    if not best_per_l:
        raise RuntimeError(
            "all autoregressive tuning combinations failed for "
            f"{model_name}; inspect {all_results_path}"
        )

    plot_mae_by_l(
        best_per_l,
        output_path=str(
            RESULTS_DIR / f"{model_name}_ar_mae_by_l.png"
        )
    )

    best_setting = select_best_l(
        best_per_l
    )
    print(
        f"\nbest setting for {model_name}:"
    )
    print(
        best_setting
    )
    print(
        f"saved all tuning results to {all_results_path}"
    )
    print(
        f"saved best per L to {best_per_l_path}"
    )

    return best_setting


def main():
    args = parse_args()
    df = load_dataset(
        DATA_FILE_PATH
    )

    if args.probe_steps is not None:
        for model_name in args.models:
            probe_model(model_name, df, args.probe_steps)
        return

    for model_name in args.models:
        run_tuning_for_ar_model(
            model_name=model_name,
            df=df
        )

    print(
        "\nautoregressive tuning complete; run "
        "final_model_comparison_run once all settings are frozen"
    )


if __name__ == "__main__":
    main()

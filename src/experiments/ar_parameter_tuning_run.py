from pathlib import Path

import pandas as pd

from src.configs.evaluation import VALIDATION_STEPS
from src.data.loader import load_dataset
from src.experiments.constants import DATA_FILE_PATH
from src.models.ar.arima import build_arima
from src.models.ar.arimax import build_arimax
from src.models.ar.sarima import build_sarima
from src.models.ar.sarimax import build_sarimax
from src.parameter_tuning.ar_parameter_grid import (
    AR_HYPERPARAMETER_GRIDS,
    AR_WINDOW_LENGTHS_BY_MODEL
)
from src.parameter_tuning.ar_tuner import select_best_l, tune_ar_model
from src.parameter_tuning.plots import plot_mae_by_l


RESULTS_DIR = Path(__file__).resolve().parent / "results"


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


def run_tuning_for_ar_model(model_name, df):
    print(
        f"\nstarting autoregressive tuning for {model_name}"
    )

    all_results, best_per_l = tune_ar_model(
        model_name=model_name,
        build_model=get_ar_model_builder(model_name),
        df=df,
        param_grid=AR_HYPERPARAMETER_GRIDS[model_name],
        window_lengths=AR_WINDOW_LENGTHS_BY_MODEL[model_name],
        validation_steps=VALIDATION_STEPS
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
    df = load_dataset(
        DATA_FILE_PATH
    )

    for model_name in ["arima", "sarima", "arimax", "sarimax"]:
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

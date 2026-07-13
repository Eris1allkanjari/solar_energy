from pathlib import Path

import pandas as pd

from src.configs.evaluation import AR_REFIT_INTERVAL, VALIDATION_STEPS
from src.data.loader import load_dataset
from src.experiments.constants import DATA_FILE_PATH

from src.models.ar.arima import build_arima
from src.models.ar.arimax import build_arimax
from src.models.ar.sarima import build_sarima
from src.models.ar.sarimax import build_sarimax

from src.parameter_tuning.ar_parameter_grid import (
    AR_HYPERPARAMETER_GRIDS,
    AR_TEST_STEPS,
    AR_VALIDATION_STEPS,
    AR_WINDOW_LENGTHS
)

from src.parameter_tuning.ar_tuner import (
    final_test,
    select_best_l,
    tune_ar_model
)

from src.parameter_tuning.plots import (
    plot_mae_by_l,
    plot_real_vs_predicted
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def get_ar_model_builder(model_name):
    if model_name == "arima":
        return build_arima

    if model_name == "sarima":
        return build_sarima

    if model_name == "arimax":
        return build_arimax

    if model_name == "sarimax":
        return build_sarimax

    raise ValueError(
        f"unknown autoregressive model: {model_name}"
    )


def run_tuning_for_ar_model(model_name, df):
    print(
        f"\nstarting autoregressive tuning for {model_name}"
    )

    build_model = get_ar_model_builder(
        model_name
    )

    param_grid = AR_HYPERPARAMETER_GRIDS[
        model_name
    ]

    all_results, best_per_l = tune_ar_model(
        model_name=model_name,
        build_model=build_model,
        df=df,
        param_grid=param_grid,
        window_lengths=AR_WINDOW_LENGTHS,
        validation_steps=AR_VALIDATION_STEPS
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results_df = pd.DataFrame(
        all_results
    )

    all_results_path = (
        RESULTS_DIR / f"{model_name}_ar_all_tuning_results.csv"
    )

    all_results_df.to_csv(
        all_results_path,
        index=False
    )

    best_per_l_df = pd.DataFrame(
        best_per_l
    )

    best_per_l_path = (
        RESULTS_DIR / f"{model_name}_ar_best_per_l.csv"
    )

    best_per_l_df.to_csv(
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

    final_result = final_test(
        model_name=model_name,
        build_model=build_model,
        df=df,
        best_setting=best_setting,
        test_steps=AR_TEST_STEPS,
        include_validation_in_training=False,
        refit_interval=AR_REFIT_INTERVAL
    )

    final_result_df = pd.DataFrame(
        [
            {
                "model": final_result["model"],
                "seq_len": final_result["seq_len"],
                "order": final_result["order"],
                "seasonal_order": final_result["seasonal_order"],
                "exog_features": final_result["exog_features"],
                "max_iter": final_result["max_iter"],
                "validation_steps": VALIDATION_STEPS,
                "test_steps": len(final_result["y_test"]),
                "training_data": "train_only",
                "ar_refit_interval": AR_REFIT_INTERVAL,
                "mae": final_result["mae"],
                "rmse": final_result["rmse"],
                "mape": final_result["mape"],
                "smape": final_result["smape"]
            }
        ]
    )

    final_result_path = (
        RESULTS_DIR / f"{model_name}_ar_final_test_result.csv"
    )

    final_result_df.to_csv(
        final_result_path,
        index=False
    )

    plot_real_vs_predicted(
        y_true=final_result["y_test"],
        y_pred=final_result["y_pred"],
        output_path=str(
            RESULTS_DIR / f"{model_name}_ar_actual_vs_predicted.png"
        ),
        title=(
            f"actual vs predicted for {model_name}, "
            f"L={final_result['seq_len']}"
        ),
        max_points=500
    )

    print(
        f"\nfinished autoregressive tuning for {model_name}"
    )

    print(
        f"saved all tuning results to {all_results_path}"
    )

    print(
        f"saved best per l to {best_per_l_path}"
    )

    print(
        f"saved final test result to {final_result_path}"
    )

    return {
        "model": model_name,
        "best_setting": best_setting,
        "final_result": final_result
    }


def main():
    df = load_dataset(
        DATA_FILE_PATH
    )

    models = [
        "arima",
        "sarima",
        "arimax",
        "sarimax"
    ]

    summary_results = []

    for model_name in models:
        result = run_tuning_for_ar_model(
            model_name=model_name,
            df=df
        )

        final_result = result[
            "final_result"
        ]

        summary_results.append(
            {
                "model": model_name,
                "best_seq_len": final_result["seq_len"],
                "order": final_result["order"],
                "seasonal_order": final_result["seasonal_order"],
                "exog_features": final_result["exog_features"],
                "max_iter": final_result["max_iter"],
                "mae": final_result["mae"],
                "rmse": final_result["rmse"],
                "mape": final_result["mape"],
                "smape": final_result["smape"]
            }
        )

    summary_df = pd.DataFrame(
        summary_results
    )

    summary_df.to_csv(
        RESULTS_DIR / "ar_tuning_summary.csv",
        index=False
    )

    print(
        "\nsaved autoregressive tuning summary to "
        f"{RESULTS_DIR / 'ar_tuning_summary.csv'}"
    )


if __name__ == "__main__":
    main()

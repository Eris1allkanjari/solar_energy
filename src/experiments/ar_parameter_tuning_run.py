import argparse
from pathlib import Path

import pandas as pd

from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    SELECTION_PROTOCOL,
    TEST_OFFSET,
    VALIDATION_STEPS
)
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
    AR_WINDOW_LENGTHS_BY_MODEL
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune autoregressive forecasting models."
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Run the selected model on the final test after tuning."
    )

    return parser.parse_args()


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


def run_tuning_for_ar_model(
    model_name,
    df,
    run_final_test=False
):
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
        window_lengths=AR_WINDOW_LENGTHS_BY_MODEL[model_name],
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

    if not run_final_test:
        print(
            "final test skipped; use final_model_comparison_run after "
            "all model settings are frozen"
        )
        return {
            "model": model_name,
            "best_setting": best_setting,
            "final_result": None
        }

    final_result = final_test(
        model_name=model_name,
        build_model=build_model,
        df=df,
        best_setting=best_setting,
        test_steps=AR_TEST_STEPS,
        test_offset=TEST_OFFSET,
        include_validation_in_training=True,
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
                "selection_protocol": SELECTION_PROTOCOL,
                "test_offset": TEST_OFFSET,
                "test_steps": len(final_result["y_test"]),
                "training_data": final_result["training_data"],
                "ar_refit_interval": AR_REFIT_INTERVAL,
                "state_context_steps": final_result[
                    "state_context_steps"
                ],
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
    args = parse_args()

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
            df=df,
            run_final_test=args.run_final_test
        )

        final_result = result[
            "final_result"
        ]

        if final_result is None:
            continue

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

    if summary_results:
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

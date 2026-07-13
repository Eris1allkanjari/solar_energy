from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import DATA_FILE_PATH
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.ar_parameter_grid import AR_TEST_STEPS
from src.parameter_tuning.ar_tuner import final_test as final_ar_test
from src.parameter_tuning.plots import plot_real_vs_predicted
from src.parameter_tuning.tuner import final_test as final_neural_test


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINAL_TEST_STEPS = AR_TEST_STEPS
AR_REFIT_INTERVAL = 0

NEURAL_MODELS = [
    "lstm",
    "gru"
]

AR_MODELS = [
    "arima",
    "sarima",
    "arimax",
    "sarimax"
]


def load_best_setting(model_name, autoregressive=False):
    suffix = "_ar_best_per_l.csv" if autoregressive else "_best_per_l.csv"
    path = RESULTS_DIR / f"{model_name}{suffix}"

    if not path.exists():
        raise FileNotFoundError(
            f"missing tuning results for {model_name}: {path}"
        )

    results_df = pd.read_csv(path)
    valid_results = results_df.dropna(
        subset=["val_mae"]
    )

    if valid_results.empty:
        raise RuntimeError(
            f"no successful tuning result found for {model_name}"
        )

    best_index = valid_results["val_mae"].astype(float).idxmin()

    return valid_results.loc[best_index].to_dict()


def result_summary(
    final_result,
    model_family,
    test_start,
    test_end
):
    return {
        "model": final_result["model"],
        "model_family": model_family,
        "seq_len": final_result["seq_len"],
        "test_start": test_start,
        "test_end": test_end,
        "test_steps": len(final_result["y_test"]),
        "training_data": "train_only",
        "ar_refit_interval": (
            AR_REFIT_INTERVAL
            if model_family == "autoregressive"
            else None
        ),
        "hidden_units_1": final_result.get("hidden_units_1"),
        "hidden_units_2": final_result.get("hidden_units_2"),
        "dropout": final_result.get("dropout"),
        "learning_rate": final_result.get("learning_rate"),
        "batch_size": final_result.get("batch_size"),
        "order": final_result.get("order"),
        "seasonal_order": final_result.get("seasonal_order"),
        "exog_features": final_result.get("exog_features"),
        "max_iter": final_result.get("max_iter"),
        "mae": final_result["mae"],
        "rmse": final_result["rmse"],
        "mape": final_result["mape"],
        "smape": final_result["smape"]
    }


def save_model_result(final_result, summary):
    model_name = final_result["model"]
    output_path = RESULTS_DIR / (
        f"{model_name}_comparison_final_test_result.csv"
    )

    pd.DataFrame([summary]).to_csv(
        output_path,
        index=False
    )

    plot_real_vs_predicted(
        y_true=final_result["y_test"],
        y_pred=final_result["y_pred"],
        output_path=str(
            RESULTS_DIR
            / f"{model_name}_comparison_actual_vs_predicted.png"
        ),
        title=(
            f"final comparison for {model_name}, "
            f"L={final_result['seq_len']}"
        ),
        max_points=500,
        show=False
    )


def save_comparison(summaries):
    comparison_df = pd.DataFrame(summaries)
    comparison_df = comparison_df.sort_values(
        "mae"
    )

    comparison_df.to_csv(
        RESULTS_DIR / "final_model_comparison.csv",
        index=False
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_dataset(
        DATA_FILE_PATH
    )
    df_proc = prepare_dataframe(
        df
    )

    n_raw = len(df_proc)
    test_start_index = int(n_raw * 0.80)
    test_df = df_proc.iloc[
        test_start_index:test_start_index + FINAL_TEST_STEPS
    ]

    if len(test_df) < FINAL_TEST_STEPS:
        raise RuntimeError(
            f"requested {FINAL_TEST_STEPS} test steps, "
            f"but only {len(test_df)} are available"
        )

    test_start = test_df.index[0]
    test_end = test_df.index[-1]
    summaries = []

    for model_name in NEURAL_MODELS:
        print(
            f"\nrunning final comparison for {model_name}"
        )

        best_setting = load_best_setting(
            model_name=model_name
        )

        final_result = final_neural_test(
            model_name=model_name,
            get_model=get_model,
            df_proc=df_proc,
            best_setting=best_setting,
            test_steps=FINAL_TEST_STEPS,
            align_test_start=True
        )

        summary = result_summary(
            final_result=final_result,
            model_family="neural",
            test_start=test_start,
            test_end=test_end
        )
        summaries.append(summary)

        save_model_result(
            final_result=final_result,
            summary=summary
        )
        save_comparison(
            summaries
        )

    for model_name in AR_MODELS:
        print(
            f"\nrunning final comparison for {model_name}"
        )

        best_setting = load_best_setting(
            model_name=model_name,
            autoregressive=True
        )

        final_result = final_ar_test(
            model_name=model_name,
            build_model=get_ar_model_builder(model_name),
            df=df,
            best_setting=best_setting,
            test_steps=FINAL_TEST_STEPS,
            include_validation_in_training=False,
            refit_interval=AR_REFIT_INTERVAL
        )

        summary = result_summary(
            final_result=final_result,
            model_family="autoregressive",
            test_start=test_start,
            test_end=test_end
        )
        summaries.append(summary)

        save_model_result(
            final_result=final_result,
            summary=summary
        )
        save_comparison(
            summaries
        )

    print(
        "\nsaved final model comparison to "
        f"{RESULTS_DIR / 'final_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()

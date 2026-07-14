import argparse
from pathlib import Path

import pandas as pd

from src.configs.evaluation import (
    NEURAL_SELECTION_PROTOCOL,
    TEST_OFFSET,
    TEST_STEPS,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.data.preprocessing import (
    add_time_features,
    select_features,
    clean_data
)

from src.experiments.constants import DATA_FILE_PATH

from src.models.rnn.gru import build_gru
from src.models.rnn.lstm import build_lstm
from src.models.rnn.lstm_attention import build_lstm_attention

from src.parameter_tuning.parameter_grid import (
    SEQ_LENGTHS,
    HYPERPARAMETER_GRIDS
)

from src.parameter_tuning.tuner import (
    tune_model,
    select_best_l,
    final_test
)

from src.parameter_tuning.plots import (
    plot_mae_by_l,
    plot_real_vs_predicted
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune neural forecasting models."
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="Run the selected model on the final test after tuning."
    )

    return parser.parse_args()


def get_model(model_name, input_shape, config):
    if model_name == "lstm":
        return build_lstm(
            input_shape=input_shape,
            config=config
        )

    if model_name == "gru":
        return build_gru(
            input_shape=input_shape,
            config=config
        )

    if model_name == "attention":
        return build_lstm_attention(
            input_shape=input_shape,
            config=config
        )

    raise ValueError(
        f"unknown model: {model_name}"
    )


def prepare_dataframe(df):
    # preprocessing

    df_proc = add_time_features(
        df.copy()
    )

    df_proc = select_features(
        df_proc
    )

    df_proc = clean_data(
        df_proc
    )

    return df_proc


def run_tuning_for_model(
    model_name,
    df_proc,
    run_final_test=False
):
    # run hyperparameter tuning for one model

    print(
        f"\nstarting tuning for {model_name}"
    )

    param_grid = HYPERPARAMETER_GRIDS[
        model_name
    ]

    all_results, best_per_l = tune_model(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        param_grid=param_grid,
        seq_lengths=SEQ_LENGTHS
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # save all validation results

    all_results_df = pd.DataFrame(
        all_results
    )

    all_results_path = (
        RESULTS_DIR / f"{model_name}_all_tuning_results.csv"
    )

    all_results_df.to_csv(
        all_results_path,
        index=False
    )

    # save best setting per l

    best_per_l_df = pd.DataFrame(
        best_per_l
    )

    best_per_l_path = (
        RESULTS_DIR / f"{model_name}_best_per_l.csv"
    )

    best_per_l_df.to_csv(
        best_per_l_path,
        index=False
    )

    # plot validation mae by l

    plot_mae_by_l(
        best_per_l,
        output_path=str(
            RESULTS_DIR / f"{model_name}_mae_by_l.png"
        )
    )

    # select best l and hyperparameters

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

    # final test using only best l and best hyperparameters

    final_result = final_test(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        best_setting=best_setting,
        test_steps=TEST_STEPS,
        test_offset=TEST_OFFSET,
        align_test_start=True,
        validation_steps=VALIDATION_STEPS
    )

    # save final test metrics

    final_result_df = pd.DataFrame(
        [
            {
                "model": final_result["model"],
                "seq_len": final_result["seq_len"],
                "hidden_units_1": final_result["hidden_units_1"],
                "hidden_units_2": final_result["hidden_units_2"],
                "dropout": final_result["dropout"],
                "learning_rate": final_result["learning_rate"],
                "batch_size": final_result["batch_size"],
                "loss": final_result["loss"],
                "huber_delta": final_result["huber_delta"],
                "weight_decay": final_result["weight_decay"],
                "gradient_clip": final_result["gradient_clip"],
                "shuffle_training": final_result["shuffle_training"],
                "seeds": final_result["seeds"],
                "seed_count": final_result["seed_count"],
                "refit_epochs": final_result["refit_epochs"],
                "mae_by_seed": final_result["mae_by_seed"],
                "rmse_by_seed": final_result["rmse_by_seed"],
                "validation_steps": VALIDATION_STEPS,
                "selection_protocol": NEURAL_SELECTION_PROTOCOL,
                "test_offset": TEST_OFFSET,
                "test_steps": len(final_result["y_test"]),
                "training_data": final_result["training_data"],
                "mae": final_result["mae"],
                "mae_std": final_result["mae_std"],
                "rmse": final_result["rmse"],
                "rmse_std": final_result["rmse_std"],
                "mape": final_result["mape"],
                "mape_std": final_result["mape_std"],
                "smape": final_result["smape"],
                "smape_std": final_result["smape_std"]
            }
        ]
    )

    final_result_path = (
        RESULTS_DIR / f"{model_name}_final_test_result.csv"
    )

    final_result_df.to_csv(
        final_result_path,
        index=False
    )

    # plot actual vs predicted

    plot_real_vs_predicted(
        y_true=final_result["y_test"],
        y_pred=final_result["y_pred"],
        output_path=str(
            RESULTS_DIR / f"{model_name}_actual_vs_predicted.png"
        ),
        title=f"actual vs predicted for {model_name}, L={final_result['seq_len']}",
        max_points=500
    )

    print(
        f"\nfinished tuning for {model_name}"
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

    # load dataset

    df = load_dataset(
        DATA_FILE_PATH
    )

    # prepare dataframe once

    df_proc = prepare_dataframe(
        df
    )

    # define models to tune

    models = [
        "lstm",
        "gru",
        # "attention"
    ]

    summary_results = []

    for model_name in models:

        result = run_tuning_for_model(
            model_name=model_name,
            df_proc=df_proc,
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
                "hidden_units_1": final_result["hidden_units_1"],
                "hidden_units_2": final_result["hidden_units_2"],
                "dropout": final_result["dropout"],
                "learning_rate": final_result["learning_rate"],
                "batch_size": final_result["batch_size"],
                "mae": final_result["mae"],
                "mae_std": final_result["mae_std"],
                "rmse": final_result["rmse"],
                "rmse_std": final_result["rmse_std"],
                "mape": final_result["mape"],
                "smape": final_result["smape"]
            }
        )

    # save model comparison summary

    if summary_results:
        summary_df = pd.DataFrame(
            summary_results
        )

        summary_df.to_csv(
            RESULTS_DIR / "tuning_summary.csv",
            index=False
        )

        print(
            "\nsaved tuning summary to "
            f"{RESULTS_DIR / 'tuning_summary.csv'}"
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import add_time_features, clean_data, select_features
from src.experiments.constants import DATA_FILE_PATH
from src.models.rnn.gru import build_gru
from src.models.rnn.lstm import build_lstm
from src.models.rnn.lstm_attention import build_lstm_attention
from src.parameter_tuning.parameter_grid import (
    HYPERPARAMETER_GRIDS,
    SEQ_LENGTHS
)
from src.parameter_tuning.plots import plot_mae_by_l
from src.parameter_tuning.tuner import select_best_l, tune_model


RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_CHOICES = ["lstm", "gru", "attention"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune recurrent forecasting models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["lstm", "gru"],
        help="Attention is optional and excluded from the default run."
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
    df_proc = add_time_features(
        df.copy()
    )
    df_proc = select_features(
        df_proc
    )
    return clean_data(
        df_proc
    )


def run_tuning_for_model(model_name, df_proc):
    print(
        f"\nstarting tuning for {model_name}"
    )

    all_results, best_per_l = tune_model(
        model_name=model_name,
        get_model=get_model,
        df_proc=df_proc,
        param_grid=HYPERPARAMETER_GRIDS[model_name],
        seq_lengths=SEQ_LENGTHS
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results_path = RESULTS_DIR / (
        f"{model_name}_all_tuning_results.csv"
    )
    pd.DataFrame(all_results).to_csv(
        all_results_path,
        index=False
    )

    best_per_l_path = RESULTS_DIR / (
        f"{model_name}_best_per_l.csv"
    )
    pd.DataFrame(best_per_l).to_csv(
        best_per_l_path,
        index=False
    )

    plot_mae_by_l(
        best_per_l,
        output_path=str(
            RESULTS_DIR / f"{model_name}_mae_by_l.png"
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
    df_proc = prepare_dataframe(
        df
    )

    for model_name in args.models:
        run_tuning_for_model(
            model_name=model_name,
            df_proc=df_proc
        )

    print(
        "\nparameter tuning complete; run final_model_comparison_run "
        "once all settings are frozen"
    )


if __name__ == "__main__":
    main()

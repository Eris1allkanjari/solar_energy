import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from src.configs.config import (
    BaseConfig,
    LargeModelConfig,
    HighDropoutConfig,
    SmallModelConfig
)

from src.data.loader import load_dataset
from src.data.preprocessing import (
    add_time_features,
    select_features,
    clean_data
)
from src.data.sequences import create_sequences

from src.experiments.constants import DATA_FILE_PATH

from src.models.rnn.gru import build_gru
from src.models.rnn.lstm import build_lstm
from src.models.rnn.lstm_attention import build_lstm_attention

from src.training.evaluation import evaluate
from src.training.trainer import train_model
from src.training.predict import predict_model

from src.utils.scaler import inverse_target


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


def run_experiment(model_name, config, df):

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

    # split before scaling

    n_raw = len(df_proc)

    train_end = int(n_raw * 0.65)
    val_end = int(n_raw * 0.80)

    train_df = df_proc.iloc[:train_end]
    val_df = df_proc.iloc[train_end:val_end]
    test_df = df_proc.iloc[val_end:]

    # scale using training data only

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train_df
    )

    val_scaled = scaler.transform(
        val_df
    )

    test_scaled = scaler.transform(
        test_df
    )

    # create sequences separately for each split

    X_train, y_train = create_sequences(
        train_scaled,
        config.SEQ_LEN
    )

    X_val, y_val = create_sequences(
        val_scaled,
        config.SEQ_LEN
    )

    X_test, y_test = create_sequences(
        test_scaled,
        config.SEQ_LEN
    )

    # build model

    model = get_model(
        model_name=model_name,
        input_shape=(config.SEQ_LEN, X_train.shape[2]),
        config=config
    )

    # train

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    # predict

    y_pred = predict_model(
        model=model,
        X_test=X_test,
        batch_size=config.BATCH_SIZE
    )

    # inverse scaling

    y_test_rescaled = inverse_target(
        scaler,
        y_test,
        X_train.shape[2]
    )

    y_pred_rescaled = inverse_target(
        scaler,
        y_pred.flatten(),
        X_train.shape[2]
    )

    y_pred_rescaled = np.clip(
        y_pred_rescaled,
        0,
        None
    )

    # evaluate

    mae, rmse, mape, smape = evaluate(
        y_test_rescaled,
        y_pred_rescaled
    )

    return {
        "model": model_name,
        "config": config.__class__.__name__,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape
    }


def main():

    # load dataset

    df = load_dataset(
        DATA_FILE_PATH
    )

    # define experiments

    models = [
        "gru",
        "lstm",
        # "attention"
    ]

    configs = [
        BaseConfig(),
        SmallModelConfig(),
        LargeModelConfig(),
        HighDropoutConfig()
    ]

    results = []

    for model_name in models:

        for cfg in configs:

            print(
                f"\nrunning {model_name} with {cfg.__class__.__name__}"
            )

            res = run_experiment(
                model_name=model_name,
                config=cfg,
                df=df
            )

            results.append(
                res
            )

            print(
                f"mae: {res['mae']:.3f}, "
                f"rmse: {res['rmse']:.3f}, "
                f"mape: {res['mape']:.2f}%, "
                f"smape: {res['smape']:.2f}%"
            )

    # save results

    os.makedirs(
        "results",
        exist_ok=True
    )

    df_results = pd.DataFrame(
        results
    )

    df_results.to_csv(
        "results/experiment_results.csv",
        index=False
    )

    print(
        "\nsaved results to results/experiment_results.csv"
    )


if __name__ == "__main__":
    main()
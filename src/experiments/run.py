import os
import pandas as pd

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

    # scaling

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(
        df_proc
    )

    # sequences

    X, y = create_sequences(
        data_scaled,
        config.SEQ_LEN
    )

    # split

    split = int(
        len(X) * 0.8
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # build model

    model = get_model(
        model_name=model_name,
        input_shape=(config.SEQ_LEN, X.shape[2]),
        config=config
    )

    # train

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
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
        X.shape[2]
    )

    y_pred_rescaled = inverse_target(
        scaler,
        y_pred.flatten(),
        X.shape[2]
    )

    # evaluate

    mae, rmse = evaluate(
        y_test_rescaled,
        y_pred_rescaled
    )

    return {
        "model": model_name,
        "config": config.__class__.__name__,
        "mae": mae,
        "rmse": rmse
    }


def main():

    # load dataset

    df = load_dataset(
        DATA_FILE_PATH
    )

    # define experiments

    models = [
        # "gru",
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
                f"rmse: {res['rmse']:.3f}"
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
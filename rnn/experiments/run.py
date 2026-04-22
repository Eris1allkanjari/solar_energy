import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from rnn.configs.config import BaseConfig, LargeModelConfig, HighDropoutConfig, SmallModelConfig
from rnn.data.loader import load_dataset
from rnn.data.preprocessing import add_time_features, select_features, clean_data
from rnn.data.sequences import create_sequences
from rnn.experiments.constants import DATA_FILE_PATH
from rnn.models.gru import build_gru
from rnn.models.lstm import build_lstm
from rnn.models.lstm_attention import build_lstm_attention
from rnn.training.evaluation import evaluate
from rnn.training.trainer import train_model
from rnn.utils.scaler import inverse_target



# Model factory
def get_model(model_name, input_shape, config):
    if model_name == "lstm":
        return build_lstm(input_shape, config)
    elif model_name == "gru":
        return build_gru(input_shape, config)
    elif model_name == "attention":
        return build_lstm_attention(input_shape, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Run single experiment
def run_experiment(model_name, config, df):

    # preprocessing
    df_proc = add_time_features(df.copy())
    df_proc = select_features(df_proc)
    df_proc = clean_data(df_proc)

    # scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_proc)

    # sequences
    X, y = create_sequences(data_scaled, config.SEQ_LEN)

    # split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # build model
    model = get_model(
        model_name,
        input_shape=(config.SEQ_LEN, X.shape[2]),
        config=config
    )

    # train
    train_model(model, X_train, y_train, config)

    # predict
    y_pred = model.predict(X_test)

    # inverse scaling
    y_test_rescaled = inverse_target(scaler, y_test, X.shape[2])
    y_pred_rescaled = inverse_target(scaler, y_pred.flatten(), X.shape[2])

    # evaluate
    mae, rmse = evaluate(y_test_rescaled, y_pred_rescaled)

    return {
        "model": model_name,
        "config": config.__class__.__name__,
        "mae": mae,
        "rmse": rmse
    }


# Main experiment loop
def main():
    # load dataset once
    df = load_dataset(DATA_FILE_PATH)

    # define experiments
    models = ["gru"]
    configs = [BaseConfig(), SmallModelConfig(), LargeModelConfig(), HighDropoutConfig()]

    results = []

    for model_name in models:
        for cfg in configs:
            print(f"\nRunning {model_name} with {cfg.__class__.__name__}")

            res = run_experiment(model_name, cfg, df)
            results.append(res)

            print(f"MAE: {res['mae']:.3f}, RMSE: {res['rmse']:.3f}")

    # save results
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/experiment_results.csv", index=False)

    print("\nSaved results to results/experiment_results.csv")


if __name__ == "__main__":
    main()
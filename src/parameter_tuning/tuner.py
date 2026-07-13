import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.data.sequences import create_sequences
from src.configs.config import ExperimentConfig
from src.training.trainer import train_model
from src.training.predict import predict_model
from src.training.evaluation import evaluate
from src.utils.scaler import inverse_target


def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def prepare_data_for_l(df_proc, seq_len):
    n_raw = len(df_proc)

    train_end = int(n_raw * 0.65)
    val_end = int(n_raw * 0.80)

    train_df = df_proc.iloc[:train_end]
    val_df = df_proc.iloc[train_end:val_end]
    test_df = df_proc.iloc[val_end:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)
    X_test, y_test = create_sequences(test_scaled, seq_len)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def evaluate_on_validation(
    model_name,
    get_model,
    df_proc,
    seq_len,
    params
):
    config = ExperimentConfig(
        params=params,
        seq_len=seq_len
    )

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data_for_l(
        df_proc=df_proc,
        seq_len=seq_len
    )

    model = get_model(
        model_name=model_name,
        input_shape=(seq_len, X_train.shape[2]),
        config=config
    )

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    y_val_pred = predict_model(
        model=model,
        X_test=X_val,
        batch_size=config.BATCH_SIZE
    )

    y_val_rescaled = inverse_target(
        scaler,
        y_val,
        X_train.shape[2]
    )

    y_val_pred_rescaled = inverse_target(
        scaler,
        y_val_pred.flatten(),
        X_train.shape[2]
    )

    y_val_pred_rescaled = np.clip(
        y_val_pred_rescaled,
        0,
        None
    )

    mae, rmse, mape, smape = evaluate(
        y_val_rescaled,
        y_val_pred_rescaled
    )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "hidden_units_1": config.HIDDEN_UNITS_1,
        "hidden_units_2": config.HIDDEN_UNITS_2,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_mape": mape,
        "val_smape": smape
    }
def tune_model(model_name, get_model, df_proc, param_grid, seq_lengths):
    all_results = []
    best_per_l = []

    for seq_len in seq_lengths:
        print(f"\nsearching {model_name} with L={seq_len}")

        best_result_for_l = None

        for params in generate_param_combinations(param_grid):
            result = evaluate_on_validation(
                model_name=model_name,
                get_model=get_model,
                df_proc=df_proc,
                seq_len=seq_len,
                params=params
            )

            all_results.append(result)

            if (
                best_result_for_l is None
                or result["val_mae"] < best_result_for_l["val_mae"]
            ):
                best_result_for_l = result

        best_per_l.append(best_result_for_l)

    return all_results, best_per_l


def select_best_l(best_per_l):
    return min(
        best_per_l,
        key=lambda result: result["val_mae"]
    )


def final_test(
    model_name,
    get_model,
    df_proc,
    best_setting,
    test_steps=None,
    align_test_start=False
):
    seq_len = best_setting["seq_len"]

    params = {
        "hidden_units_1": best_setting["hidden_units_1"],
        "hidden_units_2": best_setting["hidden_units_2"],
        "dropout": best_setting["dropout"],
        "learning_rate": best_setting["learning_rate"],
        "batch_size": best_setting["batch_size"],
        "epochs": best_setting["epochs"],
        "patience": best_setting["patience"]
    }

    config = ExperimentConfig(
        params=params,
        seq_len=seq_len
    )

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data_for_l(
        df_proc=df_proc,
        seq_len=seq_len
    )

    if align_test_start:
        n_raw = len(df_proc)
        val_end = int(n_raw * 0.80)

        test_df = df_proc.iloc[val_end:]

        if test_steps is not None:
            test_df = test_df.iloc[:test_steps]

        test_context = pd.concat(
            [
                df_proc.iloc[val_end - seq_len:val_end],
                test_df
            ],
            axis=0
        )

        test_context_scaled = scaler.transform(
            test_context
        )

        X_test, y_test = create_sequences(
            test_context_scaled,
            seq_len
        )
    elif test_steps is not None:
        X_test = X_test[:test_steps]
        y_test = y_test[:test_steps]

    model = get_model(
        model_name=model_name,
        input_shape=(seq_len, X_train.shape[2]),
        config=config
    )

    model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    y_pred = predict_model(
        model=model,
        X_test=X_test,
        batch_size=config.BATCH_SIZE
    )

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

    mae, rmse, mape, smape = evaluate(
        y_test_rescaled,
        y_pred_rescaled
    )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "hidden_units_1": config.HIDDEN_UNITS_1,
        "hidden_units_2": config.HIDDEN_UNITS_2,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "y_test": y_test_rescaled,
        "y_pred": y_pred_rescaled
    }

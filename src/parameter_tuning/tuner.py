import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from src.configs.evaluation import (
    NEURAL_SELECTION_PROTOCOL,
    RNN_SEEDS,
    TRAIN_RATIO,
    VALIDATION_END_RATIO,
    VALIDATION_STEPS
)
from src.data.sequences import create_sequences
from src.configs.config import ExperimentConfig
from src.training.trainer import (
    set_random_seed,
    train_model
)
from src.training.predict import predict_model
from src.training.evaluation import evaluate
from src.parameter_tuning.selection import (
    calculate_monthly_validation_metrics,
    select_robust_candidate
)
from src.utils.scaler import inverse_target


def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def prepare_data_for_l(
    df_proc,
    seq_len,
    validation_steps=None
):
    n_raw = len(df_proc)

    train_end = int(n_raw * TRAIN_RATIO)
    val_end = int(n_raw * VALIDATION_END_RATIO)

    train_df = df_proc.iloc[:train_end]
    val_df = df_proc.iloc[train_end:val_end]
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)

    if validation_steps is not None:
        if validation_steps <= 0:
            raise ValueError(
                "validation_steps must be a positive integer"
            )

        if validation_steps > len(val_df):
            raise ValueError(
                f"requested {validation_steps} validation steps, "
                f"but only {len(val_df)} are available"
            )

        val_context = pd.concat(
            [
                train_df.tail(seq_len),
                val_df.iloc[:validation_steps]
            ],
            axis=0
        )

        val_scaled = scaler.transform(
            val_context
        )
    else:
        val_scaled = scaler.transform(
            val_df
        )

    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_val, y_val = create_sequences(val_scaled, seq_len)

    return X_train, y_train, X_val, y_val, scaler


def evaluate_on_validation(
    model_name,
    get_model,
    df_proc,
    seq_len,
    params,
    validation_steps=VALIDATION_STEPS,
    seeds=RNN_SEEDS
):
    config = ExperimentConfig(
        params=params,
        seq_len=seq_len
    )

    X_train, y_train, X_val, y_val, scaler = prepare_data_for_l(
        df_proc=df_proc,
        seq_len=seq_len,
        validation_steps=validation_steps
    )

    y_val_rescaled = inverse_target(
        scaler,
        y_val,
        X_train.shape[2]
    )

    n_raw = len(df_proc)
    train_end = int(n_raw * TRAIN_RATIO)

    if validation_steps is None:
        validation_start_index = train_end + seq_len
    else:
        validation_start_index = train_end

    validation_index = df_proc.index[
        validation_start_index:validation_start_index + len(y_val)
    ]
    seed_results = []
    best_epochs = []

    seeds = tuple(seeds)

    if not seeds:
        raise ValueError("at least one random seed is required")

    for seed in seeds:
        print(
            f"training {model_name}, L={seq_len}, seed={seed}"
        )
        set_random_seed(seed)

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
            config=config,
            seed=seed
        )

        y_val_pred = predict_model(
            model=model,
            X_test=X_val,
            batch_size=config.BATCH_SIZE
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
        block_metrics = calculate_monthly_validation_metrics(
            index=validation_index,
            y_true=y_val_rescaled,
            y_pred=y_val_pred_rescaled
        )
        seed_results.append(
            {
                **block_metrics,
                "val_mae": mae,
                "val_rmse": rmse,
                "val_mape": mape,
                "val_smape": smape
            }
        )
        best_epochs.append(
            model.best_epoch
        )

    def metric_mean(metric):
        return float(np.mean([
            result[metric]
            for result in seed_results
        ]))

    def metric_std(metric):
        return float(np.std([
            result[metric]
            for result in seed_results
        ], ddof=0))

    def metric_values_text(metric):
        return ",".join(
            f"{result[metric]:.10g}"
            for result in seed_results
        )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "input_features": ",".join(df_proc.columns),
        "feature_count": len(df_proc.columns),
        "hidden_units_1": config.HIDDEN_UNITS_1,
        "hidden_units_2": config.HIDDEN_UNITS_2,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "loss": config.LOSS,
        "huber_delta": config.HUBER_DELTA,
        "weight_decay": config.WEIGHT_DECAY,
        "gradient_clip": config.GRADIENT_CLIP,
        "shuffle_training": True,
        "seeds": ",".join(str(seed) for seed in seeds),
        "seed_count": len(seeds),
        "best_epochs": ",".join(
            str(epoch)
            for epoch in best_epochs
        ),
        "best_epoch": int(round(np.median(best_epochs))),
        "selection_protocol": NEURAL_SELECTION_PROTOCOL,
        "validation_start": validation_index[0],
        "validation_end": validation_index[-1],
        "validation_steps": len(y_val),
        "validation_refit_interval": None,
        "validation_blocks": seed_results[0]["validation_blocks"],
        "val_block_mae_mean": metric_mean("val_block_mae_mean"),
        "val_block_mae_std": metric_mean("val_block_mae_std"),
        "val_block_rmse_mean": metric_mean("val_block_rmse_mean"),
        "val_block_rmse_std": metric_mean("val_block_rmse_std"),
        "val_seed_mae_std": metric_std("val_mae"),
        "val_seed_rmse_std": metric_std("val_rmse"),
        "val_seed_mae_values": metric_values_text("val_mae"),
        "val_seed_rmse_values": metric_values_text("val_rmse"),
        "val_mae": metric_mean("val_mae"),
        "val_rmse": metric_mean("val_rmse"),
        "val_mape": metric_mean("val_mape"),
        "val_smape": metric_mean("val_smape")
    }


def tune_model(
    model_name,
    get_model,
    df_proc,
    param_grid,
    seq_lengths,
    validation_steps=VALIDATION_STEPS
):
    all_results = []
    best_per_l = []

    for seq_len in seq_lengths:
        print(f"\nsearching {model_name} with L={seq_len}")

        results_for_l = []

        for params in generate_param_combinations(param_grid):
            result = evaluate_on_validation(
                model_name=model_name,
                get_model=get_model,
                df_proc=df_proc,
                seq_len=seq_len,
                params=params,
                validation_steps=validation_steps
            )

            all_results.append(result)
            results_for_l.append(result)

        best_per_l.append(
            select_robust_candidate(
                results_for_l
            )
        )

    return all_results, best_per_l


def select_best_l(best_per_l):
    return select_robust_candidate(
        best_per_l
    )


def parse_seeds(value):
    if value is None:
        return RNN_SEEDS

    if isinstance(value, (tuple, list)):
        return tuple(int(seed) for seed in value)

    if pd.isna(value):
        return RNN_SEEDS

    return tuple(
        int(seed.strip())
        for seed in str(value).split(",")
        if seed.strip()
    )


def params_from_best_setting(best_setting):
    return {
        "hidden_units_1": int(best_setting["hidden_units_1"]),
        "hidden_units_2": int(best_setting["hidden_units_2"]),
        "dropout": float(best_setting["dropout"]),
        "learning_rate": float(best_setting["learning_rate"]),
        "batch_size": int(best_setting["batch_size"]),
        "epochs": int(best_setting["epochs"]),
        "patience": int(best_setting["patience"]),
        "loss": best_setting.get("loss", "huber"),
        "huber_delta": float(best_setting.get("huber_delta", 0.1)),
        "weight_decay": float(best_setting.get("weight_decay", 1e-5)),
        "gradient_clip": float(best_setting.get("gradient_clip", 1.0))
    }


def final_test(
    model_name,
    get_model,
    df_proc,
    best_setting,
    test_steps=None,
    test_offset=0
):
    seq_len = int(best_setting["seq_len"])

    params = params_from_best_setting(best_setting)

    config = ExperimentConfig(
        params=params,
        seq_len=seq_len
    )

    n_raw = len(df_proc)
    val_end = int(n_raw * VALIDATION_END_RATIO)
    train_validation_df = df_proc.iloc[:val_end]
    scaler = MinMaxScaler()
    train_validation_scaled = scaler.fit_transform(
        train_validation_df
    )
    X_train, y_train = create_sequences(
        train_validation_scaled,
        seq_len
    )

    test_start = val_end + test_offset
    test_df = df_proc.iloc[test_start:]

    if test_steps is not None:
        test_df = test_df.iloc[:test_steps]

    test_context = pd.concat(
        [
            df_proc.iloc[test_start - seq_len:test_start],
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

    y_test_rescaled = inverse_target(
        scaler,
        y_test,
        X_train.shape[2]
    )

    seeds = parse_seeds(
        best_setting.get("seeds")
    )
    refit_epochs = int(
        best_setting.get(
            "best_epoch",
            config.EPOCHS
        )
    )
    seed_predictions = []
    seed_metrics = []

    for seed in seeds:
        print(
            f"refitting {model_name} on train+validation, seed={seed}"
        )
        set_random_seed(seed)
        model = get_model(
            model_name=model_name,
            input_shape=(seq_len, X_train.shape[2]),
            config=config
        )
        model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=None,
            y_val=None,
            config=config,
            seed=seed,
            epochs=refit_epochs
        )
        y_pred = predict_model(
            model=model,
            X_test=X_test,
            batch_size=config.BATCH_SIZE
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
        seed_predictions.append(
            y_pred_rescaled
        )
        seed_metrics.append(
            evaluate(
                y_test_rescaled,
                y_pred_rescaled
            )
        )

    metric_values = np.asarray(
        seed_metrics
    )
    y_pred_rescaled = np.mean(
        seed_predictions,
        axis=0
    )
    mae, rmse, mape, smape = np.mean(
        metric_values,
        axis=0
    )
    mae_std, rmse_std, mape_std, smape_std = np.std(
        metric_values,
        axis=0,
        ddof=0
    )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "input_features": ",".join(df_proc.columns),
        "feature_count": len(df_proc.columns),
        "hidden_units_1": config.HIDDEN_UNITS_1,
        "hidden_units_2": config.HIDDEN_UNITS_2,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "loss": config.LOSS,
        "huber_delta": config.HUBER_DELTA,
        "weight_decay": config.WEIGHT_DECAY,
        "gradient_clip": config.GRADIENT_CLIP,
        "shuffle_training": True,
        "seeds": ",".join(str(seed) for seed in seeds),
        "seed_count": len(seeds),
        "refit_epochs": refit_epochs,
        "mae_by_seed": ",".join(
            f"{value:.10g}"
            for value in metric_values[:, 0]
        ),
        "rmse_by_seed": ",".join(
            f"{value:.10g}"
            for value in metric_values[:, 1]
        ),
        "training_data": "train_validation",
        "state_context_steps": seq_len,
        "mae": mae,
        "mae_std": mae_std,
        "rmse": rmse,
        "rmse_std": rmse_std,
        "mape": mape,
        "mape_std": mape_std,
        "smape": smape,
        "smape_std": smape_std,
        "y_test": y_test_rescaled,
        "y_pred": y_pred_rescaled
    }

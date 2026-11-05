import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    NEURAL_SELECTION_PROTOCOL,
    RNN_SEEDS,
    VALIDATION_STEPS,
    split_positions
)
from src.data.sequences import WindowDataset
from src.configs.config import ExperimentConfig
from src.training.trainer import (
    set_random_seed,
    train_model
)
from src.training.predict import predict_model
from src.training.evaluation import evaluate
from src.parameter_tuning.ar_tuner import key_text
from src.parameter_tuning.selection import (
    calculate_monthly_validation_metrics,
    select_robust_candidate
)
from src.utils.scaler import TargetFeatureScaler, inverse_target


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

    train_end, val_end = split_positions(n_raw)

    train_df = df_proc.iloc[:train_end]
    val_df = df_proc.iloc[train_end:val_end]
    scaler = TargetFeatureScaler()

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

    train_windows = WindowDataset(train_scaled, seq_len)
    val_windows = WindowDataset(val_scaled, seq_len)

    return train_windows, val_windows, scaler


def evaluate_on_validation(
    model_name,
    get_model,
    df_proc,
    seq_len,
    params,
    validation_steps=VALIDATION_STEPS,
    seeds=RNN_SEEDS,
    return_predictions=False
):
    config = ExperimentConfig(
        params=params,
        seq_len=seq_len
    )

    train_windows, val_windows, scaler = prepare_data_for_l(
        df_proc=df_proc,
        seq_len=seq_len,
        validation_steps=validation_steps
    )

    n_features = train_windows.n_features
    y_val = val_windows.targets
    y_val_rescaled = inverse_target(
        scaler,
        y_val,
        n_features
    )

    n_raw = len(df_proc)
    train_end, _ = split_positions(n_raw)

    if validation_steps is None:
        validation_start_index = train_end + seq_len
    else:
        validation_start_index = train_end

    validation_index = df_proc.index[
        validation_start_index:validation_start_index + len(y_val)
    ]
    seed_results = []
    best_epochs = []
    seed_predictions = []

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
            input_shape=(seq_len, n_features),
            config=config
        )

        model = train_model(
            model=model,
            train_windows=train_windows,
            val_windows=val_windows,
            config=config,
            seed=seed
        )

        y_val_pred = predict_model(
            model=model,
            windows=val_windows,
            batch_size=config.BATCH_SIZE
        )

        y_val_pred_rescaled = inverse_target(
            scaler,
            y_val_pred.flatten(),
            n_features
        )
        y_val_pred_rescaled = np.clip(
            y_val_pred_rescaled,
            0,
            None
        )

        # MAPE over the whole period is not recorded: its denominator is too
        # often near zero there. It is reported only for the final test, over
        # the upwards/downwards and by-day hours.
        mae, rmse, _ = evaluate(
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
                "val_rmse": rmse
            }
        )
        seed_predictions.append(y_val_pred_rescaled)
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

    result = {
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
        "weight_decay": config.WEIGHT_DECAY,
        "gradient_clip": config.GRADIENT_CLIP,
        "lr_decay": config.LR_DECAY,
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
        "val_rmse": metric_mean("val_rmse")
    }

    if return_predictions:
        prediction_array = np.asarray(seed_predictions)
        return result, {
            "validation_index": validation_index,
            "y_true": np.asarray(y_val_rescaled),
            "seed_predictions": prediction_array,
            "ensemble_prediction": prediction_array.mean(axis=0)
        }

    return result


def neural_configuration_key(seq_len, values, parameter_names):
    """Identify a configuration by its window length and every grid value."""
    return (
        key_text(seq_len),
        *(key_text(values.get(name)) for name in parameter_names)
    )


def load_completed_neural_results(results_path, parameter_names):
    """Results of earlier runs under the current protocol, by configuration.

    Rows from another protocol or dataset describe a different experiment and
    are recomputed rather than reused.
    """
    if results_path is None or not Path(results_path).exists():
        return {}

    rows = pd.read_csv(results_path).to_dict(orient="records")

    return {
        neural_configuration_key(row["seq_len"], row, parameter_names): row
        for row in rows
        if row.get("selection_protocol") == NEURAL_SELECTION_PROTOCOL
    }


def refinement_params(incumbent, param_grid, extension_values):
    """Single-parameter variations of the incumbent, one per extension value.

    The incumbent's values are read back through params_from_best_setting, which
    restores the types a reloaded CSV row loses, so the variations train exactly
    like grid configurations.
    """
    typed = params_from_best_setting(incumbent)
    base = {name: typed[name] for name in param_grid}

    for name, values in extension_values.items():
        for value in values:
            if key_text(base[name]) == key_text(value):
                continue

            yield {**base, name: value}


def tune_model(
    model_name,
    get_model,
    df_proc,
    param_grid,
    seq_lengths,
    validation_steps=VALIDATION_STEPS,
    extension_values=None,
    results_path=None,
    dry_run=False
):
    """Evaluate the grid for every window length, then refine around its best.

    With `extension_values`, each window length's best grid configuration is
    varied one parameter at a time through the extra values, and the final
    choice is made over grid and variations together. With `results_path`,
    configurations finished in an earlier run under the current protocol are
    reused rather than retrained, and the file is rewritten after every newly
    trained configuration so an interruption loses at most one. With
    `dry_run`, nothing is trained: the counts of reused and new configurations
    are printed instead.
    """
    parameter_names = sorted(param_grid)
    completed = load_completed_neural_results(results_path, parameter_names)
    all_results = []
    best_per_l = []
    visited = set()
    counts = {"reused": 0, "new": 0, "unknown_refinements": 0}

    def save_progress():
        if results_path is None or dry_run:
            return

        pending = [
            row for key, row in completed.items() if key not in visited
        ]
        pd.DataFrame(all_results + pending).to_csv(results_path, index=False)

    def run_or_reuse(seq_len, params):
        key = neural_configuration_key(seq_len, params, parameter_names)

        if key in visited:
            return None

        visited.add(key)

        if key in completed:
            counts["reused"] += 1
            result = completed[key]
        else:
            counts["new"] += 1

            if dry_run:
                return None

            result = evaluate_on_validation(
                model_name=model_name,
                get_model=get_model,
                df_proc=df_proc,
                seq_len=seq_len,
                params=params,
                validation_steps=validation_steps
            )

        all_results.append(result)

        if key not in completed:
            save_progress()

        return result

    for seq_len in seq_lengths:
        if not dry_run:
            print(f"\nsearching {model_name} with L={seq_len}")

        results_for_l = []
        grid_complete = True

        for params in generate_param_combinations(param_grid):
            result = run_or_reuse(seq_len, params)

            if result is None:
                grid_complete = False
            else:
                results_for_l.append(result)

        if extension_values:
            if grid_complete:
                incumbent = select_robust_candidate(results_for_l)

                for params in refinement_params(
                    incumbent,
                    param_grid,
                    extension_values
                ):
                    result = run_or_reuse(seq_len, params)

                    if result is not None:
                        results_for_l.append(result)
            else:
                # Without the full grid the incumbent is not known yet, so the
                # variations it will get can only be bounded.
                counts["unknown_refinements"] += sum(
                    len(values) for values in extension_values.values()
                )

        if results_for_l and not dry_run:
            best_per_l.append(select_robust_candidate(results_for_l))

    if dry_run:
        print(
            f"{model_name}: {counts['reused']} configurations reused, "
            f"{counts['new']} to train, plus up to "
            f"{counts['unknown_refinements']} refinements that depend on "
            "grid results not yet available"
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


def parse_refit_epochs(value, seed_count, fallback):
    if value is None:
        return (fallback,) * seed_count

    if isinstance(value, (tuple, list)):
        epochs = tuple(int(epoch) for epoch in value)
    else:
        if pd.isna(value):
            return (fallback,) * seed_count

        epochs = tuple(
            int(epoch.strip())
            for epoch in str(value).split(",")
            if epoch.strip()
        )

    if len(epochs) != seed_count:
        raise ValueError(
            "best_epochs must contain one value per random seed"
        )

    return epochs


def params_from_best_setting(best_setting):
    return {
        "hidden_units_1": int(best_setting["hidden_units_1"]),
        "hidden_units_2": int(best_setting["hidden_units_2"]),
        "dropout": float(best_setting["dropout"]),
        "learning_rate": float(best_setting["learning_rate"]),
        "batch_size": int(best_setting["batch_size"]),
        "epochs": int(best_setting["epochs"]),
        "patience": int(best_setting["patience"]),
        "loss": best_setting.get("loss", "mae"),
        "weight_decay": float(best_setting.get("weight_decay", 1e-5)),
        "gradient_clip": float(best_setting.get("gradient_clip", 1.0)),
        "lr_decay": float(best_setting["lr_decay"])
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
    _, val_end = split_positions(n_raw)
    train_validation_df = df_proc.iloc[:val_end]
    scaler = TargetFeatureScaler()
    train_validation_scaled = scaler.fit_transform(
        train_validation_df
    )
    train_windows = WindowDataset(
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
    test_windows = WindowDataset(
        test_context_scaled,
        seq_len
    )

    n_features = train_windows.n_features
    y_test_rescaled = inverse_target(
        scaler,
        test_windows.targets,
        n_features
    )

    seeds = parse_seeds(
        best_setting.get("seeds")
    )
    fallback_refit_epoch = int(
        best_setting.get(
            "best_epoch",
            config.EPOCHS
        )
    )
    refit_epochs = parse_refit_epochs(
        best_setting.get("best_epochs"),
        seed_count=len(seeds),
        fallback=fallback_refit_epoch
    )
    seed_predictions = []
    seed_metrics = []

    for seed, refit_epoch in zip(seeds, refit_epochs):
        print(
            f"refitting {model_name} on train+validation, seed={seed}, "
            f"epochs={refit_epoch}"
        )
        set_random_seed(seed)
        model = get_model(
            model_name=model_name,
            input_shape=(seq_len, n_features),
            config=config
        )
        model = train_model(
            model=model,
            train_windows=train_windows,
            val_windows=None,
            config=config,
            seed=seed,
            epochs=refit_epoch
        )
        y_pred = predict_model(
            model=model,
            windows=test_windows,
            batch_size=config.BATCH_SIZE
        )
        y_pred_rescaled = inverse_target(
            scaler,
            y_pred.flatten(),
            n_features
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
    ensemble_mae, ensemble_rmse, _ = evaluate(
        y_test_rescaled,
        y_pred_rescaled
    )
    mae, rmse, _ = np.mean(
        metric_values,
        axis=0
    )
    mae_std, rmse_std, _ = np.std(
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
        "weight_decay": config.WEIGHT_DECAY,
        "gradient_clip": config.GRADIENT_CLIP,
        "lr_decay": config.LR_DECAY,
        "shuffle_training": True,
        "seeds": ",".join(str(seed) for seed in seeds),
        "seed_count": len(seeds),
        "refit_epochs": ",".join(
            str(epoch) for epoch in refit_epochs
        ),
        "refit_epoch_median": int(round(np.median(refit_epochs))),
        "mae_by_seed": ",".join(
            f"{value:.10g}"
            for value in metric_values[:, 0]
        ),
        "rmse_by_seed": ",".join(
            f"{value:.10g}"
            for value in metric_values[:, 1]
        ),
        "training_data": "train_validation",
        "metric_aggregation": "mean_across_seeds",
        "state_context_steps": seq_len,
        "mae": mae,
        "mae_std": mae_std,
        "rmse": rmse,
        "rmse_std": rmse_std,
        "ensemble_mae": ensemble_mae,
        "ensemble_rmse": ensemble_rmse,
        "seed_predictions": np.asarray(seed_predictions),
        "test_index": test_df.index,
        "y_test": y_test_rescaled,
        "y_pred": y_pred_rescaled
    }

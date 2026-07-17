import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.configs.config import ExperimentConfig
from src.configs.evaluation import (
    TRAIN_RATIO,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.data.preprocessing import (
    add_target_time_features,
    add_time_features,
    clean_data
)
from src.experiments.constants import DATA_FILE_PATH
from src.experiments.final_model_comparison_run import load_best_setting
from src.experiments.parameter_tuning_run import get_model, prepare_dataframe
from src.parameter_tuning.plots import plot_feature_ablation
from src.parameter_tuning.selection import calculate_monthly_validation_metrics
from src.parameter_tuning.tuner import (
    evaluate_on_validation,
    params_from_best_setting,
    prepare_data_for_l
)
from src.training.evaluation import evaluate
from src.training.predict import predict_model
from src.training.trainer import set_random_seed, train_model
from src.utils.scaler import inverse_target


RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPERIMENT_PROTOCOL = "daily_periodicity_validation_v1"
DEFAULT_SEED = 42
SPECIALIST_INTERVAL_HOURS = 4

BASE_WEATHER_FEATURES = [
    "solar_radiation_Wm2",
    "cloud_cover_okta",
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent"
]

TIME_FEATURE_SETS = {
    "no_time": [],
    "historical_trigonometric": ["hour_sin", "hour_cos"],
    "target_linear": ["target_hour_linear"],
    "target_cosine": ["target_hour_cosine"],
    "target_trigonometric": ["target_hour_sin", "target_hour_cos"]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation-only daily-periodicity experiments."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["summary", "time_features", "specialists"],
        default=["summary", "time_features", "specialists"]
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )
    return parser.parse_args()


def save_dataframe(rows, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def save_existing_evidence():
    lstm_results = pd.read_csv(
        RESULTS_DIR / "lstm_best_per_l.csv"
    )
    window_rows = lstm_results[
        lstm_results["seq_len"].astype(int).isin([48, 96])
    ].copy()
    window_rows["experiment_protocol"] = EXPERIMENT_PROTOCOL
    window_path = RESULTS_DIR / "lstm_daily_window_comparison.csv"
    window_rows.to_csv(window_path, index=False)

    arima_results = pd.read_csv(
        RESULTS_DIR / "arima_ar_best_per_l.csv"
    )
    sarima_results = pd.read_csv(
        RESULTS_DIR / "sarima_ar_best_per_l.csv"
    )
    same_window_rows = pd.concat(
        [
            arima_results[arima_results["seq_len"].astype(int) == 720],
            sarima_results[sarima_results["seq_len"].astype(int) == 720]
        ],
        ignore_index=True
    )
    same_window_rows["experiment_protocol"] = EXPERIMENT_PROTOCOL
    same_window_rows["feature_set"] = same_window_rows["model"]
    same_window_rows["model"] = "ar_daily_periodicity"
    ar_path = RESULTS_DIR / "arima_sarima_same_window_comparison.csv"
    same_window_rows.to_csv(ar_path, index=False)
    plot_feature_ablation(
        same_window_rows.to_dict(orient="records"),
        output_path=str(
            RESULTS_DIR / "arima_sarima_same_window_comparison.png"
        )
    )

    print(f"saved existing window evidence to {window_path}")
    print(f"saved same-window AR evidence to {ar_path}")


def prepare_time_feature_dataframe(feature_df, time_features):
    columns = [
        "pv_total_kWh",
        *BASE_WEATHER_FEATURES,
        *time_features
    ]
    return clean_data(feature_df[columns].copy())


def run_time_feature_experiment(df, seed):
    best_setting = load_best_setting("lstm")
    params = params_from_best_setting(best_setting)
    seq_len = int(best_setting["seq_len"])
    feature_df = add_target_time_features(
        add_time_features(df.copy())
    )
    results = []

    for feature_set, time_features in TIME_FEATURE_SETS.items():
        print(f"\nevaluating target-time encoding: {feature_set}")
        model_df = prepare_time_feature_dataframe(
            feature_df,
            time_features
        )
        result = evaluate_on_validation(
            model_name="lstm",
            get_model=get_model,
            df_proc=model_df,
            seq_len=seq_len,
            params=params,
            validation_steps=VALIDATION_STEPS,
            seeds=(seed,)
        )
        result["feature_set"] = feature_set
        result["time_features"] = ",".join(time_features)
        result["experiment_protocol"] = EXPERIMENT_PROTOCOL
        results.append(result)
        save_dataframe(
            results,
            "lstm_target_time_feature_comparison.csv"
        )

    plot_feature_ablation(
        results,
        output_path=str(
            RESULTS_DIR / "lstm_target_time_feature_comparison.png"
        )
    )
    return results


def load_time_feature_results(seed):
    path = RESULTS_DIR / "lstm_target_time_feature_comparison.csv"

    if not path.exists():
        return None

    results_df = pd.read_csv(path)
    required_feature_sets = set(TIME_FEATURE_SETS)
    available_feature_sets = set(results_df.get("feature_set", []))
    protocols = set(results_df.get("experiment_protocol", []))
    seeds = set(
        pd.to_numeric(
            results_df.get("seeds", pd.Series(dtype=float)),
            errors="coerce"
        ).dropna().astype(int)
    )

    if (
        available_feature_sets != required_feature_sets
        or protocols != {EXPERIMENT_PROTOCOL}
        or seeds != {seed}
    ):
        return None

    return results_df.to_dict(orient="records")


def evaluate_specialist_models(df_proc, best_setting, seed):
    seq_len = int(best_setting["seq_len"])
    params = params_from_best_setting(best_setting)
    config = ExperimentConfig(params=params, seq_len=seq_len)
    X_train, y_train, X_val, y_val, scaler = prepare_data_for_l(
        df_proc=df_proc,
        seq_len=seq_len,
        validation_steps=VALIDATION_STEPS
    )

    train_end = int(len(df_proc) * TRAIN_RATIO)
    train_target_index = df_proc.index[seq_len:train_end]
    validation_index = df_proc.index[
        train_end:train_end + len(y_val)
    ]

    if len(train_target_index) != len(y_train):
        raise RuntimeError("training timestamps do not align with sequences")

    predictions_scaled = np.full(len(y_val), np.nan)
    block_rows = []
    train_blocks = train_target_index.hour // SPECIALIST_INTERVAL_HOURS
    validation_blocks = validation_index.hour // SPECIALIST_INTERVAL_HOURS
    specialist_count = 24 // SPECIALIST_INTERVAL_HOURS

    for block in range(specialist_count):
        train_mask = np.asarray(train_blocks == block)
        validation_mask = np.asarray(validation_blocks == block)

        if not train_mask.any() or not validation_mask.any():
            raise RuntimeError(f"time block {block} has no samples")

        print(
            f"\ntraining LSTM specialist {block + 1}/{specialist_count} "
            f"for hours {block * 4:02d}-{block * 4 + 3:02d}"
        )
        set_random_seed(seed)
        model = get_model(
            model_name="lstm",
            input_shape=(seq_len, X_train.shape[2]),
            config=config
        )
        model = train_model(
            model=model,
            X_train=X_train[train_mask],
            y_train=y_train[train_mask],
            X_val=X_val[validation_mask],
            y_val=y_val[validation_mask],
            config=config,
            seed=seed
        )
        block_predictions = predict_model(
            model=model,
            X_test=X_val[validation_mask],
            batch_size=config.BATCH_SIZE
        ).flatten()
        predictions_scaled[validation_mask] = block_predictions

        block_true = inverse_target(
            scaler,
            y_val[validation_mask],
            X_train.shape[2]
        )
        block_pred = inverse_target(
            scaler,
            block_predictions,
            X_train.shape[2]
        )
        block_pred = np.clip(block_pred, 0, None)
        mae, rmse, mape, smape = evaluate(block_true, block_pred)
        block_rows.append(
            {
                "block": block,
                "start_hour": block * SPECIALIST_INTERVAL_HOURS,
                "end_hour": block * SPECIALIST_INTERVAL_HOURS + 3,
                "train_samples": int(train_mask.sum()),
                "validation_samples": int(validation_mask.sum()),
                "best_epoch": model.best_epoch,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "smape": smape,
                "seed": seed,
                "experiment_protocol": EXPERIMENT_PROTOCOL
            }
        )

    if np.isnan(predictions_scaled).any():
        raise RuntimeError("specialist predictions do not cover validation")

    y_true = inverse_target(
        scaler,
        y_val,
        X_train.shape[2]
    )
    y_pred = inverse_target(
        scaler,
        predictions_scaled,
        X_train.shape[2]
    )
    y_pred = np.clip(y_pred, 0, None)
    mae, rmse, mape, smape = evaluate(y_true, y_pred)
    monthly_metrics = calculate_monthly_validation_metrics(
        index=validation_index,
        y_true=y_true,
        y_pred=y_pred
    )
    summary = {
        "model": "lstm",
        "feature_set": "six_4h_specialists",
        "seq_len": seq_len,
        "specialist_interval_hours": SPECIALIST_INTERVAL_HOURS,
        "specialist_count": specialist_count,
        "seed": seed,
        "validation_steps": len(y_true),
        **monthly_metrics,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_mape": mape,
        "val_smape": smape,
        "experiment_protocol": EXPERIMENT_PROTOCOL
    }
    return summary, block_rows


def run_specialist_experiment(df, seed, time_feature_results=None):
    best_setting = load_best_setting("lstm")
    df_proc = prepare_dataframe(df)
    specialist_result, block_rows = evaluate_specialist_models(
        df_proc=df_proc,
        best_setting=best_setting,
        seed=seed
    )

    if time_feature_results is None:
        time_feature_results = load_time_feature_results(seed)

    if time_feature_results is None:
        time_feature_results = run_time_feature_experiment(df, seed)

    baseline = next(
        result
        for result in time_feature_results
        if result["feature_set"] == "historical_trigonometric"
    ).copy()
    baseline["feature_set"] = "single_lstm"
    comparison = [baseline, specialist_result]

    save_dataframe(
        comparison,
        "lstm_time_block_model_comparison.csv"
    )
    save_dataframe(
        block_rows,
        "lstm_time_block_details.csv"
    )
    plot_feature_ablation(
        comparison,
        output_path=str(
            RESULTS_DIR / "lstm_time_block_model_comparison.png"
        )
    )


def main():
    args = parse_args()
    selected_experiments = set(args.experiments)
    df = load_dataset(DATA_FILE_PATH)
    time_feature_results = None

    if "summary" in selected_experiments:
        save_existing_evidence()

    if "time_features" in selected_experiments:
        time_feature_results = run_time_feature_experiment(
            df=df,
            seed=args.seed
        )

    if "specialists" in selected_experiments:
        run_specialist_experiment(
            df=df,
            seed=args.seed,
            time_feature_results=time_feature_results
        )

    print(f"\nsaved daily-periodicity results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

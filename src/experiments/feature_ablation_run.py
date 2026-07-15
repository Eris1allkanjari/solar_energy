import argparse
from pathlib import Path

import pandas as pd

from src.configs.evaluation import AR_REFIT_INTERVAL, VALIDATION_STEPS
from src.data.loader import load_dataset
from src.data.preprocessing import add_time_features, clean_data
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import DATA_FILE_PATH
from src.experiments.final_model_comparison_run import load_best_setting
from src.experiments.parameter_tuning_run import get_model
from src.parameter_tuning.ar_tuner import (
    evaluate_on_validation as evaluate_ar_validation,
    params_from_best_setting as ar_params_from_best_setting,
    prepare_ar_data
)
from src.parameter_tuning.plots import plot_feature_ablation
from src.parameter_tuning.tuner import (
    evaluate_on_validation as evaluate_neural_validation,
    params_from_best_setting as neural_params_from_best_setting
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_CHOICES = ["lstm", "gru", "arimax", "sarimax"]

NEURAL_FEATURE_SETS = {
    "target_history": [],
    "temporal": ["hour_sin", "hour_cos"],
    "radiation": ["solar_radiation_Wm2"],
    "radiation_cloud": [
        "solar_radiation_Wm2",
        "cloud_cover_okta"
    ],
    "core_weather": [
        "solar_radiation_Wm2",
        "cloud_cover_okta",
        "temperature_C"
    ],
    "expanded_weather": [
        "solar_radiation_Wm2",
        "cloud_cover_okta",
        "temperature_C",
        "wind_speed_ms",
        "humidity_percent",
        "hour_sin",
        "hour_cos"
    ],
    "all_available": [
        "solar_radiation_Wm2",
        "cloud_cover_okta",
        "temperature_C",
        "wind_speed_ms",
        "humidity_percent",
        "pressure_hpa",
        "sunshine_hours",
        "wind_direction_sin",
        "wind_direction_cos",
        "hour_sin",
        "hour_cos"
    ]
}

AR_FEATURE_SETS = {
    name: features
    for name, features in NEURAL_FEATURE_SETS.items()
    if name not in {"target_history", "temporal"}
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen models with different feature sets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES
    )
    return parser.parse_args()


def prepare_feature_dataframe(df, features):
    required_columns = ["pv_total_kWh", *features]
    missing_columns = set(required_columns).difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"missing feature columns: {sorted(missing_columns)}"
        )

    return clean_data(df[required_columns].copy())


def save_results(model_name, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_name}_feature_ablation.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    plot_feature_ablation(
        results,
        output_path=str(
            RESULTS_DIR / f"{model_name}_feature_ablation.png"
        )
    )


def run_neural_ablation(model_name, feature_df):
    best_setting = load_best_setting(model_name)
    params = neural_params_from_best_setting(best_setting)
    seq_len = int(best_setting["seq_len"])
    results = []

    for feature_set, features in NEURAL_FEATURE_SETS.items():
        print(f"\nevaluating {model_name} with {feature_set}")
        model_df = prepare_feature_dataframe(feature_df, features)
        result = evaluate_neural_validation(
            model_name=model_name,
            get_model=get_model,
            df_proc=model_df,
            seq_len=seq_len,
            params=params,
            validation_steps=VALIDATION_STEPS
        )
        result["feature_set"] = feature_set
        result["feature_count"] = len(features) + 1
        result["features"] = ",".join(
            ["pv_total_kWh", *features]
        )
        results.append(result)
        save_results(model_name, results)


def run_ar_ablation(model_name, feature_df):
    best_setting = load_best_setting(
        model_name,
        autoregressive=True
    )
    base_params = ar_params_from_best_setting(best_setting)
    window_length = int(best_setting["seq_len"])
    results = []

    for feature_set, features in AR_FEATURE_SETS.items():
        print(f"\nevaluating {model_name} with {feature_set}")
        model_df = prepare_feature_dataframe(feature_df, features)
        params = dict(base_params)
        params["exog_features"] = features
        splits = prepare_ar_data(model_df)
        result = evaluate_ar_validation(
            model_name=model_name,
            build_model=get_ar_model_builder(model_name),
            y_train=splits[0],
            y_val=splits[1],
            exog_train=splits[3],
            exog_val=splits[4],
            exog_test=splits[5],
            window_length=window_length,
            params=params,
            validation_steps=VALIDATION_STEPS,
            refit_interval=AR_REFIT_INTERVAL
        )
        result["feature_set"] = feature_set
        result["feature_count"] = len(features)
        result["features"] = ",".join(features)
        results.append(result)
        save_results(model_name, results)


def main():
    args = parse_args()
    raw_df = load_dataset(DATA_FILE_PATH)
    feature_df = add_time_features(raw_df.copy())

    for model_name in args.models:
        if model_name in {"lstm", "gru"}:
            run_neural_ablation(model_name, feature_df)
        else:
            run_ar_ablation(model_name, feature_df)

    print(f"\nsaved feature ablation results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

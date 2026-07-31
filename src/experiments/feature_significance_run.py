import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    PV_QUALITY_THRESHOLD,
    RNN_SEEDS,
    VALIDATION_STEPS
)
from src.data.loader import load_dataset
from src.data.preprocessing import (
    DEFAULT_NEURAL_FEATURES,
    add_time_features,
    clean_data
)
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import (
    DATA_FILE_PATH,
    FEATURE_SIGNIFICANCE_RESULTS_DIR
)
from src.experiments.final_model_comparison_run import (
    load_best_setting,
    load_pv_quality
)
from src.experiments.parameter_tuning_run import get_model
from src.parameter_tuning.ar_tuner import (
    evaluate_on_validation as evaluate_ar_validation,
    params_from_best_setting as ar_params_from_best_setting,
    prepare_ar_data
)
from src.parameter_tuning.tuner import (
    evaluate_on_validation as evaluate_neural_validation,
    params_from_best_setting as neural_params_from_best_setting
)
from src.training.evaluation import evaluate


RESULTS_DIR = FEATURE_SIGNIFICANCE_RESULTS_DIR
MODEL_CHOICES = ["lstm", "gru", "arimax", "sarimax"]
EXPERIMENT_PROTOCOL = "lofo_quality_seed_mean_bootstrap_holm_v2"

FEATURE_GROUPS = {
    "solar_radiation": ["solar_radiation_Wm2"],
    "cloud_cover": ["cloud_cover_okta"],
    "temperature": ["temperature_C"],
    "wind_speed": ["wind_speed_ms"],
    "humidity": ["humidity_percent"],
    "hour_of_day": ["hour_sin", "hour_cos"]
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure feature contribution with leave-one-feature-group-out "
            "validation experiments."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=MODEL_CHOICES
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(RNN_SEEDS),
        help="Fixed seeds used for recurrent models."
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--block-hours",
        type=int,
        default=24
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026
    )
    return parser.parse_args()


def prepare_feature_dataframe(df, features):
    columns = ["pv_total_kWh", *features]
    missing = set(columns).difference(df.columns)

    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")

    return clean_data(df[columns].copy())


def available_feature_groups(features):
    feature_set = set(features)

    return {
        group_name: group_features
        for group_name, group_features in FEATURE_GROUPS.items()
        if set(group_features).issubset(feature_set)
    }


def moving_block_bootstrap(
    loss_difference,
    samples,
    block_hours,
    seed
):
    values = np.asarray(loss_difference, dtype=float)
    n_values = len(values)

    if n_values == 0:
        raise ValueError("loss_difference cannot be empty")

    if samples <= 0 or block_hours <= 0:
        raise ValueError("bootstrap samples and block hours must be positive")

    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(n_values / block_hours))
    offsets = np.arange(block_hours)
    bootstrap_means = np.empty(samples, dtype=float)

    for sample_index in range(samples):
        starts = generator.integers(0, n_values, size=block_count)
        indices = (
            starts[:, None] + offsets[None, :]
        ) % n_values
        sampled = values[indices.reshape(-1)[:n_values]]
        bootstrap_means[sample_index] = sampled.mean()

    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    probability_nonpositive = (
        np.count_nonzero(bootstrap_means <= 0) + 1
    ) / (samples + 1)
    probability_nonnegative = (
        np.count_nonzero(bootstrap_means >= 0) + 1
    ) / (samples + 1)
    p_value = min(
        1.0,
        2 * min(probability_nonpositive, probability_nonnegative)
    )

    return float(lower), float(upper), float(p_value)


def apply_holm_correction(rows):
    corrected_rows = [dict(row) for row in rows]
    candidates = [
        (index, float(row["p_value_raw"]))
        for index, row in enumerate(corrected_rows)
        if pd.notna(row.get("p_value_raw"))
    ]
    candidates.sort(key=lambda item: item[1])
    running_adjusted = 0.0
    total = len(candidates)

    for rank, (row_index, p_value) in enumerate(candidates):
        adjusted = min(1.0, (total - rank) * p_value)
        running_adjusted = max(running_adjusted, adjusted)
        corrected_rows[row_index]["p_value_holm"] = running_adjusted

    for row in corrected_rows:
        adjusted = row.get("p_value_holm")
        lower = row.get("delta_mae_ci_lower")
        row["significant_difference"] = bool(
            pd.notna(adjusted) and adjusted < 0.05
        )
        row["significant_benefit"] = bool(
            row["significant_difference"]
            and pd.notna(lower)
            and lower > 0
        )

    return corrected_rows


def evaluate_neural_variant(
    model_name,
    feature_df,
    features,
    best_setting,
    seeds
):
    result, predictions = evaluate_neural_validation(
        model_name=model_name,
        get_model=get_model,
        df_proc=prepare_feature_dataframe(feature_df, features),
        seq_len=int(best_setting["seq_len"]),
        params=neural_params_from_best_setting(best_setting),
        validation_steps=VALIDATION_STEPS,
        seeds=seeds,
        return_predictions=True
    )
    return result, predictions


def evaluate_ar_variant(
    model_name,
    feature_df,
    features,
    best_setting
):
    params = ar_params_from_best_setting(best_setting)
    params["exog_features"] = features
    model_df = prepare_feature_dataframe(feature_df, features)
    splits = prepare_ar_data(model_df)
    result, predictions = evaluate_ar_validation(
        model_name=model_name,
        build_model=get_ar_model_builder(model_name),
        y_train=splits[0],
        y_val=splits[1],
        exog_train=splits[3],
        exog_val=splits[4],
        exog_test=splits[5],
        window_length=int(best_setting["seq_len"]),
        params=params,
        validation_steps=VALIDATION_STEPS,
        refit_interval=AR_REFIT_INTERVAL,
        return_predictions=True
    )
    return result, predictions


def prediction_metrics(predictions, mask=None):
    y_true = np.asarray(predictions["y_true"])

    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    seed_predictions = predictions.get("seed_predictions")

    if seed_predictions is not None:
        metric_values = np.asarray(
            [
                evaluate(
                    y_true[mask],
                    np.asarray(seed_prediction)[mask]
                )
                for seed_prediction in seed_predictions
            ]
        )
        return tuple(metric_values.mean(axis=0))

    return evaluate(
        y_true[mask],
        np.asarray(predictions["ensemble_prediction"])[mask]
    )


def mean_absolute_error_by_timestamp(predictions):
    y_true = np.asarray(predictions["y_true"])
    seed_predictions = predictions.get("seed_predictions")

    if seed_predictions is not None:
        return np.mean(
            np.abs(np.asarray(seed_predictions) - y_true[None, :]),
            axis=0
        )

    return np.abs(
        np.asarray(predictions["ensemble_prediction"]) - y_true
    )


def attach_validation_quality(predictions, quality_df):
    validation_index = pd.DatetimeIndex(
        predictions["validation_index"]
    )
    quality = quality_df.reindex(validation_index)

    if quality["observation_fraction"].isna().any():
        raise ValueError(
            "PV quality audit does not cover the full validation period"
        )

    predictions["pv_observation_fraction"] = quality[
        "observation_fraction"
    ].to_numpy()
    predictions["high_quality_mask"] = (
        predictions["pv_observation_fraction"] >= PV_QUALITY_THRESHOLD
    )
    return predictions


def base_row(
    model_name,
    best_setting,
    features,
    dropped_group,
    dropped_features,
    result,
    predictions,
    seeds
):
    all_mae, all_rmse, all_mape = prediction_metrics(
        predictions
    )
    quality_mask = predictions["high_quality_mask"]
    mae, rmse, mape = prediction_metrics(
        predictions,
        mask=quality_mask
    )

    return {
        "model": model_name,
        "model_family": (
            "neural" if model_name in {"lstm", "gru"}
            else "autoregressive"
        ),
        "experiment_protocol": EXPERIMENT_PROTOCOL,
        "selection_protocol": best_setting["selection_protocol"],
        "seq_len": int(best_setting["seq_len"]),
        "dropped_group": dropped_group,
        "dropped_features": ",".join(dropped_features),
        "remaining_features": ",".join(features),
        "remaining_feature_count": len(features),
        "validation_start": predictions["validation_index"][0],
        "validation_end": predictions["validation_index"][-1],
        "validation_steps": len(predictions["y_true"]),
        "pv_quality_threshold": PV_QUALITY_THRESHOLD,
        "inference_samples": int(quality_mask.sum()),
        "inference_coverage": float(quality_mask.mean()),
        "seeds": (
            ",".join(str(seed) for seed in seeds)
            if model_name in {"lstm", "gru"}
            else None
        ),
        "fit_converged": result.get("fit_converged"),
        "accepted": bool(result.get("fit_converged", True)),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "all_hour_mae": all_mae,
        "all_hour_rmse": all_rmse,
        "all_hour_mape": all_mape,
        "delta_mae": np.nan,
        "delta_mae_percent": np.nan,
        "delta_mae_ci_lower": np.nan,
        "delta_mae_ci_upper": np.nan,
        "p_value_raw": np.nan,
        "p_value_holm": np.nan,
        "effect_direction": "reference"
    }


def compare_with_reference(
    row,
    reference_row,
    predictions,
    reference_predictions,
    bootstrap_samples,
    block_hours,
    bootstrap_seed
):
    if not row["accepted"] or not reference_row["accepted"]:
        row["effect_direction"] = "rejected_non_converged"
        return row

    reference_error = mean_absolute_error_by_timestamp(
        reference_predictions
    )
    variant_error = mean_absolute_error_by_timestamp(predictions)
    quality_mask = reference_predictions["high_quality_mask"]

    if not np.array_equal(
        quality_mask,
        predictions["high_quality_mask"]
    ):
        raise ValueError("feature variants use different quality masks")

    loss_difference = (
        variant_error - reference_error
    )[quality_mask]
    lower, upper, p_value = moving_block_bootstrap(
        loss_difference=loss_difference,
        samples=bootstrap_samples,
        block_hours=block_hours,
        seed=bootstrap_seed
    )
    delta_mae = float(loss_difference.mean())
    row["delta_mae"] = delta_mae
    row["delta_mae_percent"] = (
        100 * delta_mae / reference_row["mae"]
    )
    row["delta_mae_ci_lower"] = lower
    row["delta_mae_ci_upper"] = upper
    row["p_value_raw"] = p_value
    row["effect_direction"] = (
        "beneficial" if delta_mae > 0 else "detrimental"
    )
    return row


def plot_results(model_name, rows):
    plot_rows = [
        row for row in rows
        if row["dropped_group"] != "none"
        and pd.notna(row["delta_mae"])
    ]

    if not plot_rows:
        return

    labels = [row["dropped_group"] for row in plot_rows]
    values = np.asarray([row["delta_mae"] for row in plot_rows])
    lower = np.asarray([row["delta_mae_ci_lower"] for row in plot_rows])
    upper = np.asarray([row["delta_mae_ci_upper"] for row in plot_rows])
    errors = np.vstack([
        np.maximum(values - lower, 0),
        np.maximum(upper - values, 0)
    ])
    colors = [
        "#2e7d32" if row["significant_benefit"] else "#607d8b"
        for row in plot_rows
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color=colors, yerr=errors, capsize=4)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("MAE increase when feature group is removed")
    axis.set_title(f"Feature contribution significance: {model_name.upper()}")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / f"{model_name}_feature_significance.png",
        dpi=200
    )
    plt.close(figure)


def save_rows(model_name, rows, create_plot=False):
    corrected_rows = apply_holm_correction(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{model_name}_feature_significance.csv"
    pd.DataFrame(corrected_rows).to_csv(output_path, index=False)

    if create_plot:
        plot_results(model_name, corrected_rows)

    return corrected_rows, output_path


def add_inference_metadata(row, args):
    row["bootstrap_samples"] = args.bootstrap_samples
    row["bootstrap_block_hours"] = args.block_hours
    row["bootstrap_seed"] = args.bootstrap_seed
    row["confidence_level"] = 0.95
    row["multiple_testing_correction"] = "holm"
    return row


def run_model(model_name, feature_df, quality_df, args):
    autoregressive = model_name in {"arimax", "sarimax"}
    best_setting = load_best_setting(
        model_name,
        autoregressive=autoregressive
    )
    full_features = (
        ar_params_from_best_setting(best_setting)["exog_features"]
        if autoregressive
        else list(DEFAULT_NEURAL_FEATURES)
    )
    evaluator = evaluate_ar_variant if autoregressive else evaluate_neural_variant

    print(f"\nevaluating {model_name} full-feature reference")
    if autoregressive:
        reference_result, reference_predictions = evaluator(
            model_name, feature_df, full_features, best_setting
        )
    else:
        reference_result, reference_predictions = evaluator(
            model_name, feature_df, full_features, best_setting, args.seeds
        )
    reference_predictions = attach_validation_quality(
        reference_predictions,
        quality_df
    )

    reference_row = add_inference_metadata(
        base_row(
            model_name=model_name,
            best_setting=best_setting,
            features=full_features,
            dropped_group="none",
            dropped_features=[],
            result=reference_result,
            predictions=reference_predictions,
            seeds=args.seeds
        ),
        args
    )

    if not reference_row["accepted"]:
        raise RuntimeError(f"full-feature {model_name} fit did not converge")

    rows = [reference_row]
    save_rows(model_name, rows)

    for group_index, (group_name, group_features) in enumerate(
        available_feature_groups(full_features).items()
    ):
        remaining_features = [
            feature for feature in full_features
            if feature not in group_features
        ]
        print(f"\nevaluating {model_name} without {group_name}")

        if autoregressive:
            result, predictions = evaluator(
                model_name,
                feature_df,
                remaining_features,
                best_setting
            )
        else:
            result, predictions = evaluator(
                model_name,
                feature_df,
                remaining_features,
                best_setting,
                args.seeds
            )
        predictions = attach_validation_quality(predictions, quality_df)

        row = add_inference_metadata(
            base_row(
                model_name=model_name,
                best_setting=best_setting,
                features=remaining_features,
                dropped_group=group_name,
                dropped_features=group_features,
                result=result,
                predictions=predictions,
                seeds=args.seeds
            ),
            args
        )
        rows.append(
            compare_with_reference(
                row=row,
                reference_row=reference_row,
                predictions=predictions,
                reference_predictions=reference_predictions,
                bootstrap_samples=args.bootstrap_samples,
                block_hours=args.block_hours,
                bootstrap_seed=args.bootstrap_seed + group_index
            )
        )
        save_rows(model_name, rows)

    rows, output_path = save_rows(
        model_name,
        rows,
        create_plot=True
    )
    print(f"saved {output_path}")


def main():
    args = parse_args()
    raw_df = load_dataset(DATA_FILE_PATH)
    feature_df = add_time_features(raw_df.copy())
    quality_df = load_pv_quality()

    for model_name in args.models:
        run_model(model_name, feature_df, quality_df, args)


if __name__ == "__main__":
    main()

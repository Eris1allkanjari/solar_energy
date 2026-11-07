"""Test-period metrics of ARIMAX and SARIMAX with and without cloud cover.

Each model keeps its selected order and window and is run on the test period
under the final protocol, scored by the same code as the final comparison. Only
the exogenous inputs change:

- radiation + cloud cover (the input set of the current final results),
- radiation only (cloud cover removed),
- radiation + daylight cloud cover (cloud cover weighted by the solar elevation,
  so it is zero at night).

The upwards/downwards and by-day MAPE are reported as in the final comparison,
together with the day and night NMAE, since the cost of plain cloud cover was
found at night on validation.
"""

import argparse

import pandas as pd

from src.configs.dataset_selection import add_dataset_argument
from src.configs.evaluation import (
    AR_REFIT_INTERVAL,
    TEST_OFFSET,
    split_positions
)
from src.data.loader import load_dataset
from src.data.preprocessing import DAYLIGHT_CLOUD_FEATURE
from src.experiments.ar_parameter_tuning_run import get_ar_model_builder
from src.experiments.constants import (
    ANALYSIS_RESULTS_DIR,
    DATA_FILE_PATH,
    FINAL_COMPARISON_PATH
)
from src.experiments.final_model_comparison_run import (
    attach_pv_quality,
    load_best_setting,
    load_pv_quality,
    result_summary
)
from src.experiments.parameter_tuning_run import prepare_dataframe
from src.parameter_tuning.ar_tuner import final_test
from src.utils.capacity import training_production_range


RADIATION = "solar_radiation_Wm2"
CLOUD = "cloud_cover_okta"

INPUT_SETS = {
    "radiation + cloud": [RADIATION, CLOUD],
    "radiation only": [RADIATION],
    "radiation + daylight cloud": [RADIATION, DAYLIGHT_CLOUD_FEATURE]
}

METRICS = [
    "upwards_downwards_mape",
    "by_day_mape",
    "nmae",
    "day_nmae",
    "night_nmae",
    "high_quality_upwards_downwards_mape",
    "high_quality_by_day_mape"
]

# Autoregressive models without inputs, shown for reference.
REFERENCE_MODELS = ["sarima", "arima"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare ARIMAX and SARIMAX test metrics with and without cloud "
            "cover, keeping each model's selected order and window."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sarimax", "arimax"],
        default=["sarimax", "arimax"]
    )
    add_dataset_argument(parser)
    return parser.parse_args()


def setting_text(best_setting):
    seasonal_order = best_setting.get("seasonal_order")
    seasonal = "" if pd.isna(seasonal_order) else f" {seasonal_order}"

    return (
        f"{best_setting['order']}{seasonal}, "
        f"L={int(best_setting['seq_len'])}"
    )


def main():
    args = parse_args()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_FILE_PATH)
    df_proc = prepare_dataframe(df)
    minimum_kwh, capacity_kwh = training_production_range(df_proc)
    quality_df = load_pv_quality()
    _, test_split_start = split_positions(len(df_proc))
    test_index = df_proc.index[test_split_start + TEST_OFFSET:]

    rows = []

    for model_name in args.models:
        best_setting = dict(
            load_best_setting(model_name, autoregressive=True)
        )

        for input_name, features in INPUT_SETS.items():
            print(f"\nrunning {model_name} with {input_name}")
            setting = {
                **best_setting,
                "exog_features": ",".join(features)
            }
            final_result = final_test(
                model_name=model_name,
                build_model=get_ar_model_builder(model_name),
                df=df,
                best_setting=setting,
                test_offset=TEST_OFFSET,
                include_validation_in_training=True,
                refit_interval=AR_REFIT_INTERVAL
            )
            summary = result_summary(
                final_result=attach_pv_quality(final_result, quality_df),
                best_setting=setting,
                model_family="autoregressive",
                test_start=test_index[0],
                test_end=test_index[-1],
                capacity_kwh=capacity_kwh,
                minimum_kwh=minimum_kwh
            )
            rows.append(
                {
                    "model": model_name,
                    "inputs": input_name,
                    "exog_features": ",".join(features),
                    "selected_inputs": best_setting["exog_features"],
                    "setting": setting_text(best_setting),
                    **{metric: summary[metric] for metric in METRICS}
                }
            )

    if FINAL_COMPARISON_PATH.exists():
        final_df = pd.read_csv(FINAL_COMPARISON_PATH).set_index("model")

        for model_name in REFERENCE_MODELS:
            if model_name in final_df.index:
                rows.append(
                    {
                        "model": model_name,
                        "inputs": "none",
                        "exog_features": None,
                        "selected_inputs": None,
                        "setting": setting_text(final_df.loc[model_name]),
                        **{
                            metric: final_df.loc[model_name, metric]
                            for metric in METRICS
                        }
                    }
                )

    table = pd.DataFrame(rows)
    output_path = ANALYSIS_RESULTS_DIR / "cloud_cover_comparison.csv"
    table.to_csv(output_path, index=False)

    print()
    print(
        table[["model", "inputs", "setting", *METRICS[:5]]]
        .round(2)
        .to_string(index=False)
    )
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

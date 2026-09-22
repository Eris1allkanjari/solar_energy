"""How much of the MAPE comes from dawn, dusk, cloud and technical failure.

Every hour that MAPE actually scores is assigned to exactly one category, and
the table reports what each category contributes to the final percentage. The
contribution is the quantity that matters: an hour with a huge percentage error
barely moves the mean if there are only a handful of them, so mean APE alone
answers the wrong question.

Categories are assigned in priority order, since an hour can qualify for more
than one. A data outage is a measurement problem rather than a weather one, so
it takes precedence; solar geometry comes next; cloud is judged only for hours
where the sun is high enough that geometry is not already the explanation.
"""

import argparse

import numpy as np
import pandas as pd

from src.configs.evaluation import (
    DAY_PERIOD_ELEVATION_DEGREES,
    MAPE_PRODUCTION_THRESHOLD,
    PV_QUALITY_THRESHOLD,
    SITE_LATITUDE,
    SITE_LONGITUDE
)
from src.data.loader import load_dataset
from src.experiments.constants import (
    ANALYSIS_RESULTS_DIR,
    DATA_FILE_PATH,
    FINAL_COMPARISON_RESULTS_DIR,
    PV_QUALITY_HOURLY_PATH
)
from src.utils.solar import solar_elevation


CLOUDY_OKTA_THRESHOLD = 5.0

CATEGORY_ORDER = [
    "technical_failure",
    "dawn",
    "dusk",
    "cloudy_day",
    "clear_day"
]

CATEGORY_DESCRIPTION = {
    "technical_failure": (
        f"fewer than {PV_QUALITY_THRESHOLD:.0%} of expected PV readings present"
    ),
    "dawn": (
        f"sun below {DAY_PERIOD_ELEVATION_DEGREES:.0f} deg and rising"
    ),
    "dusk": (
        f"sun below {DAY_PERIOD_ELEVATION_DEGREES:.0f} deg and setting"
    ),
    "cloudy_day": (
        f"sun above {DAY_PERIOD_ELEVATION_DEGREES:.0f} deg, "
        f"cloud cover at least {CLOUDY_OKTA_THRESHOLD:.0f} okta"
    ),
    "clear_day": (
        f"sun above {DAY_PERIOD_ELEVATION_DEGREES:.0f} deg, "
        f"cloud cover below {CLOUDY_OKTA_THRESHOLD:.0f} okta"
    )
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Break the test-period MAPE down by dawn, dusk, cloud cover and "
            "technical failure."
        )
    )
    parser.add_argument("--model", default="gru")
    return parser.parse_args()


def load_hours(model_name):
    path = (
        FINAL_COMPARISON_RESULTS_DIR
        / f"{model_name}_final_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run the final model comparison first."
        )

    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time")

    weather = load_dataset(DATA_FILE_PATH)
    frame = frame.join(weather[["cloud_cover_okta"]], how="left")

    quality = pd.read_csv(PV_QUALITY_HOURLY_PATH)
    quality["time"] = pd.to_datetime(
        quality["time"],
        utc=True
    ).dt.tz_localize(None)
    frame = frame.join(
        quality.set_index("time")[["observation_fraction"]],
        how="left"
    )

    index = pd.DatetimeIndex(frame.index)
    frame["elevation"] = solar_elevation(
        index=index,
        latitude=SITE_LATITUDE,
        longitude=SITE_LONGITUDE
    )
    # Rising or setting, rather than a clock time, so the split holds across a
    # year in which day length varies by more than eight hours.
    frame["rising"] = frame["elevation"].diff().fillna(0) > 0
    frame["abs_error"] = (frame["y_pred"] - frame["y_true"]).abs()

    return frame


def assign_category(frame):
    conditions = [
        frame["observation_fraction"] < PV_QUALITY_THRESHOLD,
        (frame["elevation"] <= DAY_PERIOD_ELEVATION_DEGREES)
        & frame["rising"],
        frame["elevation"] <= DAY_PERIOD_ELEVATION_DEGREES,
        frame["cloud_cover_okta"] >= CLOUDY_OKTA_THRESHOLD
    ]
    choices = ["technical_failure", "dawn", "dusk", "cloudy_day"]
    frame["category"] = np.select(conditions, choices, default="clear_day")

    return frame


def build_table(scored):
    total = len(scored)
    mape = scored["abs_pct_error"].mean()
    rows = []

    for category in CATEGORY_ORDER:
        rows_in = scored[scored["category"] == category]

        if rows_in.empty:
            contribution = 0.0
            rows.append(
                {
                    "category": category,
                    "definition": CATEGORY_DESCRIPTION[category],
                    "hours": 0,
                    "share_of_scored_hours_pct": 0.0,
                    "mean_actual_kWh": np.nan,
                    "mean_abs_error_kWh": np.nan,
                    "mean_ape_pct": np.nan,
                    "mape_contribution_points": contribution,
                    "share_of_mape_pct": 0.0
                }
            )
            continue

        # Each hour contributes its own APE divided by the total hour count, so
        # the contributions sum exactly to the reported MAPE.
        contribution = rows_in["abs_pct_error"].sum() / total
        rows.append(
            {
                "category": category,
                "definition": CATEGORY_DESCRIPTION[category],
                "hours": len(rows_in),
                "share_of_scored_hours_pct": 100 * len(rows_in) / total,
                "mean_actual_kWh": rows_in["y_true"].mean(),
                "mean_abs_error_kWh": rows_in["abs_error"].mean(),
                "mean_ape_pct": rows_in["abs_pct_error"].mean(),
                "mape_contribution_points": contribution,
                "share_of_mape_pct": 100 * contribution / mape
            }
        )

    table = pd.DataFrame(rows)
    table.loc[len(table)] = {
        "category": "TOTAL",
        "definition": "all hours scored by MAPE",
        "hours": total,
        "share_of_scored_hours_pct": 100.0,
        "mean_actual_kWh": scored["y_true"].mean(),
        "mean_abs_error_kWh": scored["abs_error"].mean(),
        "mean_ape_pct": mape,
        "mape_contribution_points": table["mape_contribution_points"].sum(),
        "share_of_mape_pct": 100.0
    }

    return table


def main():
    args = parse_args()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = assign_category(load_hours(args.model))
    scored = frame[frame["y_true"] > MAPE_PRODUCTION_THRESHOLD].copy()
    scored["abs_pct_error"] = (
        100 * scored["abs_error"] / scored["y_true"]
    )

    table = build_table(scored).round(3)
    output_path = (
        ANALYSIS_RESULTS_DIR / f"mape_category_breakdown_{args.model}.csv"
    )
    table.to_csv(output_path, index=False)

    excluded = len(frame) - len(scored)
    print(f"model                 : {args.model}")
    print(f"test hours            : {len(frame)}")
    print(
        f"scored by MAPE        : {len(scored)} "
        f"(actual > {MAPE_PRODUCTION_THRESHOLD} kWh)"
    )
    print(f"excluded as too small : {excluded}")
    print()
    print(
        table[
            [
                "category",
                "hours",
                "share_of_scored_hours_pct",
                "mean_actual_kWh",
                "mean_abs_error_kWh",
                "mean_ape_pct",
                "mape_contribution_points",
                "share_of_mape_pct"
            ]
        ].to_string(index=False)
    )
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

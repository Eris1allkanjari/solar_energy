"""What drives large forecast errors during the day period?

Tests whether high-error daytime hours track cloud cover, missing PV readings,
or a reduced number of reporting systems.

The whole question is confounded by production level: errors are larger when
production is larger, and both cloud cover and the reporting-systems count move
with time of day and season. A raw correlation between error and any of these
would therefore be partly a restatement of "it was sunny". Every association
here is reported both raw and partial, holding actual production fixed, and the
gap between the two is the interesting quantity.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from src.configs.evaluation import (
    DAY_PERIOD_ELEVATION_DEGREES,
    MAPE_PRODUCTION_THRESHOLD,
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


WEATHER_DRIVERS = [
    "cloud_cover_okta",
    "solar_radiation_Wm2",
    "sunshine_hours",
    "humidity_percent"
]

QUALITY_DRIVERS = [
    "observation_fraction",
    "mean_reporting_systems",
    "fully_missing_minutes"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Correlate daytime forecast error against weather and PV data "
            "quality, holding production level fixed."
        )
    )
    parser.add_argument("--model", default="gru")
    parser.add_argument(
        "--elevation",
        type=float,
        default=DAY_PERIOD_ELEVATION_DEGREES,
        help="Solar elevation defining the day period."
    )
    return parser.parse_args()


def load_quality():
    quality = pd.read_csv(PV_QUALITY_HOURLY_PATH)
    quality["time"] = pd.to_datetime(
        quality["time"],
        utc=True
    ).dt.tz_localize(None)
    return quality.set_index("time")


def build_frame(model_name, elevation_threshold):
    path = (
        FINAL_COMPARISON_RESULTS_DIR
        / f"{model_name}_final_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"missing {path}")

    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time")

    weather = load_dataset(DATA_FILE_PATH)
    available = [c for c in WEATHER_DRIVERS if c in weather.columns]
    frame = frame.join(weather[available], how="left")
    frame = frame.join(
        load_quality()[QUALITY_DRIVERS],
        how="left"
    )

    index = pd.DatetimeIndex(frame.index)
    frame["elevation"] = solar_elevation(
        index=index,
        latitude=SITE_LATITUDE,
        longitude=SITE_LONGITUDE
    )
    frame["abs_error"] = (frame["y_pred"] - frame["y_true"]).abs()
    frame["signed_error"] = frame["y_pred"] - frame["y_true"]

    day = frame[frame["elevation"] > elevation_threshold].copy()
    scored = day["y_true"] > MAPE_PRODUCTION_THRESHOLD
    day["abs_pct_error"] = np.where(
        scored,
        100 * day["abs_error"] / day["y_true"],
        np.nan
    )

    return day, available


def partial_spearman(x, y, control):
    """Spearman correlation of x and y with `control` held fixed.

    Computed on ranks, by removing the linear dependence of each variable on the
    ranked control and correlating the residuals. Ranks first, so a monotone but
    non-linear relationship with production is still removed properly.
    """
    frame = pd.DataFrame(
        {"x": x, "y": y, "c": control}
    ).dropna()

    if len(frame) < 30:
        return np.nan, np.nan, len(frame)

    ranked = frame.rank()
    control_matrix = np.column_stack(
        [np.ones(len(ranked)), ranked["c"].to_numpy()]
    )

    def residual(column):
        values = ranked[column].to_numpy()
        coefficients, *_ = np.linalg.lstsq(
            control_matrix,
            values,
            rcond=None
        )
        return values - control_matrix @ coefficients

    correlation, p_value = stats.pearsonr(
        residual("x"),
        residual("y")
    )
    return correlation, p_value, len(frame)


def association_table(day, drivers, target, control):
    rows = []

    for driver in drivers:
        frame = pd.DataFrame(
            {"d": day[driver], "t": day[target], "c": day[control]}
        ).dropna()

        if len(frame) < 30:
            continue

        raw, raw_p = stats.spearmanr(frame["d"], frame["t"])
        partial, partial_p, n = partial_spearman(
            frame["d"],
            frame["t"],
            frame["c"]
        )
        rows.append(
            {
                "driver": driver,
                "n": n,
                "raw_rho": raw,
                "raw_p": raw_p,
                "partial_rho": partial,
                "partial_p": partial_p
            }
        )

    return pd.DataFrame(rows)


def stratified_means(day, driver, target, bins=4):
    """Mean target by driver quartile, within production bands.

    Holding production roughly fixed inside a band, a driver that genuinely
    matters should still separate the quartiles. One that only looked important
    because it tracks production will flatten out.
    """
    frame = day[["y_true", driver, target]].dropna().copy()

    if frame.empty:
        return pd.DataFrame()

    frame["production_band"] = pd.cut(
        frame["y_true"],
        bins=[0, 20, 60, 120, np.inf],
        labels=["<20", "20-60", "60-120", "120+"],
        right=False
    )

    try:
        frame["driver_quartile"] = pd.qcut(
            frame[driver],
            bins,
            labels=[f"Q{i + 1}" for i in range(bins)],
            duplicates="drop"
        )
    except ValueError:
        return pd.DataFrame()

    return frame.pivot_table(
        index="production_band",
        columns="driver_quartile",
        values=target,
        aggfunc="mean",
        observed=True
    ).round(2)


def main():
    args = parse_args()
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    day, weather_available = build_frame(args.model, args.elevation)
    drivers = weather_available + QUALITY_DRIVERS

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"Error drivers during the day period: {args.model}")
    emit("=" * 78)
    emit(f"day period      : solar elevation > {args.elevation} degrees")
    emit(f"day hours       : {len(day)}")
    emit(
        "scored hours    : "
        f"{int(day['abs_pct_error'].notna().sum())} "
        f"(actual > {MAPE_PRODUCTION_THRESHOLD} kWh)"
    )
    emit()
    # observation_fraction and mean_reporting_systems are the same measurement
    # in different units, so they cannot be told apart as separate causes.
    coverage = day[["observation_fraction", "mean_reporting_systems"]].dropna()

    if not coverage.empty:
        rho, _ = stats.spearmanr(
            coverage["observation_fraction"],
            coverage["mean_reporting_systems"]
        )
        emit(
            "observation_fraction vs mean_reporting_systems: "
            f"Spearman rho = {rho:.4f}"
        )
        emit()

    emit("Spread of the quality variables over these hours:")

    for driver in QUALITY_DRIVERS:
        series = day[driver].dropna()

        if series.empty:
            continue

        emit(
            f"  {driver:24s} min={series.min():8.3f} "
            f"p25={series.quantile(.25):8.3f} "
            f"median={series.median():8.3f} "
            f"p75={series.quantile(.75):8.3f} "
            f"max={series.max():8.3f}"
        )

    for target, label in [
        ("abs_error", "ABSOLUTE ERROR (kWh)"),
        ("abs_pct_error", "ABSOLUTE PERCENTAGE ERROR (%)")
    ]:
        emit()
        emit("=" * 78)
        emit(f"{label} vs each driver, Spearman rho")
        emit("  raw     = plain association")
        emit("  partial = holding actual production fixed")
        emit("=" * 78)
        table = association_table(day, drivers, target, "y_true")

        if table.empty:
            continue

        emit(
            f"  {'driver':24s} {'n':>5} {'raw_rho':>9} {'raw_p':>10} "
            f"{'partial_rho':>12} {'partial_p':>10}"
        )

        for _, row in table.sort_values(
            "partial_rho",
            key=abs,
            ascending=False
        ).iterrows():
            emit(
                f"  {row['driver']:24s} {int(row['n']):5d} "
                f"{row['raw_rho']:9.3f} {row['raw_p']:10.3g} "
                f"{row['partial_rho']:12.3f} {row['partial_p']:10.3g}"
            )

    emit()
    emit("=" * 78)
    emit("OUTAGE HOURS AND THE WORST ERRORS")
    emit("=" * 78)

    # An aggregate correlation can hide a small number of catastrophic hours, so
    # the two extremes are checked directly: hours with no reporting systems at
    # all, and the hours with the largest errors.
    outage = day[day["observation_fraction"] <= 0.001]
    emit(f"  day-period hours with no reporting systems: {len(outage)}")

    if not outage.empty:
        emit(
            f"    their actual production : mean {outage['y_true'].mean():.2f} "
            f"kWh, max {outage['y_true'].max():.2f} kWh"
        )
        emit(
            f"    their absolute error    : {outage['abs_error'].mean():.2f} kWh"
            f"  (day-period overall {day['abs_error'].mean():.2f} kWh)"
        )

    threshold = day["abs_error"].quantile(0.99)
    worst = day[day["abs_error"] >= threshold]
    rest = day[day["abs_error"] < threshold]

    if not worst.empty and not rest.empty:
        _, p_value = stats.mannwhitneyu(
            worst["observation_fraction"].dropna(),
            rest["observation_fraction"].dropna()
        )
        emit(
            f"  worst 1% of errors (n={len(worst)}): mean coverage "
            f"{worst['observation_fraction'].mean():.4f}"
        )
        emit(
            f"  all other hours    (n={len(rest)}): mean coverage "
            f"{rest['observation_fraction'].mean():.4f}"
        )
        emit(f"  Mann-Whitney p on coverage: {p_value:.4g}")

    emit()
    emit("=" * 78)
    emit("MEAN ABSOLUTE ERROR (kWh) BY DRIVER QUARTILE, WITHIN PRODUCTION BAND")
    emit("=" * 78)

    for driver in drivers:
        table = stratified_means(day, driver, "abs_error")

        if table.empty:
            continue

        emit()
        emit(f"{driver}:")
        emit(table.to_string())

    output_path = (
        ANALYSIS_RESULTS_DIR / f"error_drivers_{args.model}.log"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

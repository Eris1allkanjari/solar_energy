from src.configs.dataset_selection import ACTIVE_DATASET, STEPS_PER_HOUR

TRAIN_RATIO = 0.65
VALIDATION_END_RATIO = 0.80

# Durations are fixed in hours and converted to rows of the active dataset, so
# an hourly and a 10-minute run cover the same stretch of time.
VALIDATION_HOURS = 5260
VALIDATION_STEPS = VALIDATION_HOURS * STEPS_PER_HOUR
TEST_OFFSET = 0
TEST_STEPS = None

AR_REFIT_INTERVAL = 0
# One step is the latest completed interval at either resolution.
EXOG_LAG_STEPS = 1
SELECTION_MAE_TOLERANCE = 0.02
MIN_SEASONAL_WINDOW = 720 * STEPS_PER_HOUR
SEASONAL_PERIOD = 24 * STEPS_PER_HOUR
RNN_SEEDS = (42, 123, 2026)
# 5 kWh per hour, the same production rate expressed per interval.
MAPE_PRODUCTION_THRESHOLD = 5.0 / STEPS_PER_HOUR
PV_QUALITY_THRESHOLD = 0.50


def split_positions(n_rows):
    """Row positions where training ends and where validation ends.

    Both boundaries fall on whole hours, so every dataset splits at the same
    timestamps whatever its resolution. Taking int(n * ratio) directly on
    10-minute rows would land half an hour away from the hourly split. For
    hourly data this is exactly int(n * ratio).
    """
    if n_rows % STEPS_PER_HOUR:
        raise ValueError(
            f"{n_rows} rows is not a whole number of hours at "
            f"{STEPS_PER_HOUR} steps per hour"
        )

    n_hours = n_rows // STEPS_PER_HOUR

    return (
        int(n_hours * TRAIN_RATIO) * STEPS_PER_HOUR,
        int(n_hours * VALIDATION_END_RATIO) * STEPS_PER_HOUR
    )

# Utrecht, the site of the PV systems. Used only to split scored hours into
# day and night via solar elevation, never as a model input.
SITE_LATITUDE = 52.0907
SITE_LONGITUDE = 5.1214

# Sun above the horizon. Separates lit hours from dark ones, and is what the
# day/night MAE split uses to show how much the dark half dilutes a pooled MAE.
DAYLIGHT_ELEVATION_DEGREES = 0.0

# Sun high enough that dawn and dusk are excluded as well as night. Used by the
# diagnostic scripts in src/analysis, which explain the physics behind the error
# and need a mask that follows the sun rather than the clock.
DAY_PERIOD_ELEVATION_DEGREES = 10.0

# Reported metrics split the day by clock hour rather than solar elevation. The
# hours are fixed and need no site geometry to interpret, which makes a reported
# figure reproducible from the timestamp alone.
#
# The upwards and downwards hours are the morning rise and afternoon fall, where
# production is small and the percentage error is correspondingly inflated; the
# by-day hours are the high-production plateau. Percentage error is reported
# over these two periods separately and nowhere else, since a denominator near
# zero makes it meaningless over night or over the period as a whole.
UPWARDS_DOWNWARDS_PERIOD_HOURS = (6, 7, 8, 9, 15, 16, 17)
BY_DAY_PERIOD_HOURS = (10, 11, 12, 13, 14)

# Day is the union of the two; night is derived as its complement rather than
# listed, so the two cannot drift apart.
DAY_PERIOD_HOURS = tuple(range(6, 18))

PERIOD_DEFINITION = (
    "day=06-17, night=18-05, upwards_downwards=06-09+15-17, by_day=10-14"
)

# Bumped when the inputs or the scoring change, so results produced under the
# old version are rejected rather than mixed in. The AR tag was last bumped for
# the realignment of KNMI weather from interval-end to interval-start labels;
# the neural tag since then for the per-epoch learning-rate decay. The final tag
# was left alone for that change: final rows record their tuning protocol, and
# the AR and baseline final results it would have discarded did not change.
#
# The dataset name is part of every tag, so results from one dataset can never
# be accepted as another's, even if files are copied between result folders.
SELECTION_PROTOCOL = (
    f"monthly_blocks_converged_bic_lag1_aligned_v7_{ACTIVE_DATASET.name}"
)
NEURAL_SELECTION_PROTOCOL = (
    f"monthly_blocks_seeded_mae_loss_lr_decay_v9_{ACTIVE_DATASET.name}"
)
FINAL_COMPARISON_PROTOCOL = (
    f"rolling_origin_seed_mean_quality_aligned_v10_{ACTIVE_DATASET.name}"
)

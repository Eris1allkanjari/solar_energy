TRAIN_RATIO = 0.65
VALIDATION_END_RATIO = 0.80

VALIDATION_STEPS = 5260
TEST_OFFSET = 0
TEST_STEPS = None

AR_REFIT_INTERVAL = 0
EXOG_LAG_STEPS = 1
SELECTION_MAE_TOLERANCE = 0.02
MIN_SEASONAL_WINDOW = 720
RNN_SEEDS = (42, 123, 2026)
MAPE_PRODUCTION_THRESHOLD = 5.0
PV_QUALITY_THRESHOLD = 0.50

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
# The ramp hours are the morning rise and afternoon fall, where production is
# small and the percentage error is correspondingly inflated; midday is the
# high-production plateau. Percentage error is reported over these two periods
# separately and nowhere else, since a denominator near zero makes it
# meaningless over night or over the period as a whole.
RAMP_PERIOD_HOURS = (6, 7, 8, 9, 15, 16, 17)
MIDDAY_PERIOD_HOURS = (10, 11, 12, 13, 14)

# Day is the union of the two; night is derived as its complement rather than
# listed, so the two cannot drift apart.
DAY_PERIOD_HOURS = tuple(range(6, 18))

PERIOD_DEFINITION = (
    "day=06-17, night=18-05, ramp=06-09+15-17, midday=10-14"
)

SELECTION_PROTOCOL = "monthly_blocks_converged_bic_lag1_causal_v6"
NEURAL_SELECTION_PROTOCOL = "monthly_blocks_seeded_mae_loss_v7"
FINAL_COMPARISON_PROTOCOL = "rolling_origin_seed_mean_quality_v9"

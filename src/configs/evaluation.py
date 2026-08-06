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
DAYLIGHT_ELEVATION_DEGREES = 0.0

SELECTION_PROTOCOL = "monthly_blocks_converged_bic_lag1_causal_v6"
NEURAL_SELECTION_PROTOCOL = "monthly_blocks_seeded_mae_loss_v7"
FINAL_COMPARISON_PROTOCOL = "rolling_origin_seed_mean_quality_v9"

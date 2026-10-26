from src.configs.dataset_selection import (
    ACTIVE_DATASET,
    EXPERIMENTS_DIR,
    PROJECT_ROOT
)


# The data file and every results folder follow the active dataset, so runs on
# different datasets never overwrite each other. See dataset_selection.
DATA_FILE_PATH = ACTIVE_DATASET.path
RAW_PV_FILE_PATH = (
    PROJECT_ROOT
    / "utrecht"
    / "pv_data"
    / "filtered_pv_power_measurements_ac.csv"
)

RESULTS_ROOT = ACTIVE_DATASET.results_root
ANALYSIS_RESULTS_DIR = RESULTS_ROOT / "analysis"

# The PV quality audit describes the PV record alone, which is the same for
# every dataset, so it stays in one place instead of following the active one.
PV_QUALITY_RESULTS_DIR = EXPERIMENTS_DIR / "results" / "analysis"
PV_QUALITY_HOURLY_PATH = PV_QUALITY_RESULTS_DIR / "pv_data_quality_hourly.csv"

PARAMETER_TUNING_RESULTS_DIR = RESULTS_ROOT / "parameter_tuning"
NEURAL_TUNING_RESULTS_DIR = PARAMETER_TUNING_RESULTS_DIR / "neural"
AR_TUNING_RESULTS_DIR = PARAMETER_TUNING_RESULTS_DIR / "autoregressive"
FEATURE_ABLATION_RESULTS_DIR = RESULTS_ROOT / "feature_ablation"
FEATURE_SIGNIFICANCE_RESULTS_DIR = RESULTS_ROOT / "feature_significance"
DAILY_PERIODICITY_RESULTS_DIR = RESULTS_ROOT / "daily_periodicity"
EFFICIENCY_RESULTS_DIR = RESULTS_ROOT / "computational_efficiency"
FINAL_COMPARISON_RESULTS_DIR = RESULTS_ROOT / "final_comparison"
SUMMARY_RESULTS_DIR = RESULTS_ROOT / "summary"
PERIODICITY_ABLATION_RESULTS_DIR = RESULTS_ROOT / "periodicity_ablation"
SEED_SWEEP_RESULTS_DIR = RESULTS_ROOT / "seed_sweep"
LOSS_COMPARISON_RESULTS_DIR = RESULTS_ROOT / "loss_comparison"

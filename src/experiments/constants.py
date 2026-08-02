from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_PATH = PROJECT_ROOT / "utrecht" / "processed_data" / "utrecht_pv_data.csv"
RAW_PV_FILE_PATH = (
    PROJECT_ROOT
    / "utrecht"
    / "pv_data"
    / "filtered_pv_power_measurements_ac.csv"
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
ANALYSIS_RESULTS_DIR = RESULTS_ROOT / "analysis"
PV_QUALITY_HOURLY_PATH = ANALYSIS_RESULTS_DIR / "pv_data_quality_hourly.csv"
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

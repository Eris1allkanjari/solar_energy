from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE_PATH = PROJECT_ROOT / "utrecht" / "processed_data" / "utrecht_pv_data.csv"

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
ANALYSIS_RESULTS_DIR = RESULTS_ROOT / "analysis"
PARAMETER_TUNING_RESULTS_DIR = RESULTS_ROOT / "parameter_tuning"
NEURAL_TUNING_RESULTS_DIR = PARAMETER_TUNING_RESULTS_DIR / "neural"
AR_TUNING_RESULTS_DIR = PARAMETER_TUNING_RESULTS_DIR / "autoregressive"
FEATURE_ABLATION_RESULTS_DIR = RESULTS_ROOT / "feature_ablation"
DAILY_PERIODICITY_RESULTS_DIR = RESULTS_ROOT / "daily_periodicity"
DAYLIGHT_WEIGHTING_RESULTS_DIR = RESULTS_ROOT / "daylight_weighting"
FINAL_COMPARISON_RESULTS_DIR = RESULTS_ROOT / "final_comparison"

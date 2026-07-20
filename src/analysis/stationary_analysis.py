import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.configs.evaluation import TRAIN_RATIO
from src.data.loader import load_dataset
from src.data.preprocessing import clean_data
from src.experiments.constants import ANALYSIS_RESULTS_DIR, DATA_FILE_PATH


RESULTS_DIR = ANALYSIS_RESULTS_DIR


def run_adf_test(series):
    result = adfuller(series)
    critical_values = result[4]

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_observations": result[3],
        "critical_value_1_percent": critical_values["1%"],
        "critical_value_5_percent": critical_values["5%"],
        "critical_value_10_percent": critical_values["10%"],
        "stationary_at_5_percent": result[1] < 0.05
    }


def print_adf_result(result):

    print("\nadf test results")

    print(f"adf statistic: {result['adf_statistic']:.6f}")

    print(f"p-value: {result['p_value']:.6f}")

    print("\ncritical values:")

    for level in (1, 5, 10):
        value = result[f"critical_value_{level}_percent"]
        print(f"{level}%: {value:.6f}")

    if result["stationary_at_5_percent"]:

        print("\nseries is likely stationary")

    else:

        print("\nseries is likely non-stationary")


def main():

    # load dataset

    df = load_dataset(
       DATA_FILE_PATH
    )

    train_end = int(len(df) * TRAIN_RATIO)
    training_df = clean_data(
        df.iloc[:train_end][["pv_total_kWh"]]
    )
    y = training_df["pv_total_kWh"]

    result = run_adf_test(y)
    result["series"] = "pv_total_kWh"
    result["data_scope"] = "training_only"
    result["start"] = y.index[0]
    result["end"] = y.index[-1]
    result["observations"] = len(y)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "stationarity_adf.csv"
    pd.DataFrame([result]).to_csv(output_path, index=False)

    print_adf_result(result)
    print(f"\nsaved results to {output_path}")


if __name__ == "__main__":

    main()

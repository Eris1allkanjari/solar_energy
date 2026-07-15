from statsmodels.tsa.stattools import adfuller

from src.data.loader import load_dataset
from src.experiments.constants import DATA_FILE_PATH


def run_adf_test(series):

    result = adfuller(series)

    print("\nadf test results")

    print(f"adf statistic: {result[0]:.6f}")

    print(f"p-value: {result[1]:.6f}")

    print("\ncritical values:")

    for key, value in result[4].items():

        print(f"{key}: {value:.6f}")

    if result[1] < 0.05:

        print("\nseries is likely stationary")

    else:

        print("\nseries is likely non-stationary")


def main():

    # load dataset

    df = load_dataset(
       DATA_FILE_PATH
    )

    # target variable

    y = df["pv_total_kWh"].clip(lower=0)

    # fill missing values

    y = y.interpolate().bfill().ffill()

    # run adf test

    run_adf_test(y)


if __name__ == "__main__":

    main()

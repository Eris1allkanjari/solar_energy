import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.data.loader import load_dataset
from src.experiments.ar_experiment_definition import experiments, run_experiment
from src.experiments.constants import DATA_FILE_PATH


def main():

    # load data

    df = load_dataset(
        DATA_FILE_PATH
    )

    # target variable

    y = df["pv_total_kWh"].clip(lower=0)

    y = y.interpolate().bfill().ffill()

    # train test split

    train_size = int(len(y) * 0.8)

    y_train = y[:train_size]
    y_test = y[train_size:]

    # run experiments

    results = []

    for exp in experiments:

        print(f"\nrunning: {exp['name']}")

        try:

            # exogenous features

            exog_train = None
            exog_test = None

            if exp["use_exog"]:

                exog = df[
                    exp["config"].EXOG_FEATURES
                ]

                exog = exog.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                exog = exog.interpolate().bfill().ffill()

                # split before scaling

                exog_train = exog[:train_size]

                exog_test = exog[train_size:]

                # fit scaler on training data only

                scaler = StandardScaler()

                exog_train = pd.DataFrame(

                    scaler.fit_transform(exog_train),

                    index=exog_train.index,

                    columns=exog_train.columns
                )

                exog_test = pd.DataFrame(

                    scaler.transform(exog_test),

                    index=exog_test.index,

                    columns=exog_test.columns
                )

                # align datasets

                y_train_aligned, exog_train = y_train.align(
                    exog_train,
                    join="inner"
                )

                y_test_aligned, exog_test = y_test.align(
                    exog_test,
                    join="inner"
                )

            else:

                y_train_aligned = y_train
                y_test_aligned = y_test

            # run experiment

            start = time.time()

            res = run_experiment(
                build_model=exp["builder"],
                model_name=exp["name"],
                y_train=y_train_aligned,
                y_test=y_test_aligned,
                config=exp["config"],
                exog_train=exog_train,
                exog_test=exog_test,
                forecasting_strategy=exp["forecasting"]
            )

            duration = time.time() - start

            res["runtime_sec"] = duration

            results.append(res)

            # print results

            print(
                f"mae: {res['mae']:.4f}"
            )

            print(
                f"rmse: {res['rmse']:.4f}"
            )

            print(
                f"runtime: {duration:.2f} sec"
            )

        except Exception as e:

            print(
                f"experiment failed: {e}"
            )

    # save results

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/statistical_results.csv",
        index=False
    )

    print("\nsaved results to results/statistical_results.csv")


if __name__ == "__main__":
    main()
from pathlib import Path

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from src.data.loader import load_dataset
from src.experiments.constants import DATA_FILE_PATH

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"


def analyze_acf_pacf(
    series,
    lags=48,
    output_path=None,
    show=False
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # acf plot

    plot_acf(
        series,
        lags=lags,
        ax=axes[0]
    )

    axes[0].set_title(
        "autocorrelation function"
    )

    # pacf plot

    plot_pacf(
        series,
        lags=lags,
        ax=axes[1]
    )

    axes[1].set_title(
        "partial autocorrelation function"
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def main():

    # load dataset

    df = load_dataset(
        DATA_FILE_PATH
    )

    # target variable

    y = df["pv_total_kWh"].clip(lower=0)

    # fill missing values

    y = y.interpolate().bfill().ffill()

    # analyze acf pacf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "acf_pacf.png"

    analyze_acf_pacf(
        y,
        lags=168,
        output_path=output_path
    )

    print(f"saved plot to {output_path}")


if __name__ == "__main__":

    main()

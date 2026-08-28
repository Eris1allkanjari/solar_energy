import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from src.data.loader import load_dataset
from src.data.preprocessing import clean_data
from src.configs.evaluation import TRAIN_RATIO
from src.experiments.constants import ANALYSIS_RESULTS_DIR, DATA_FILE_PATH

RESULTS_DIR = ANALYSIS_RESULTS_DIR


def analyze_acf_pacf(
    series,
    lags=48,
    output_path=None,
    show=False
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # acf plot

    plot_acf(
        series,
        lags=lags,
        ax=axes[0]
    )

    axes[0].set_title(
        "training-set autocorrelation function (ACF)"
    )
    axes[0].set_xlabel(
        "lag k (hours before the current hour)"
    )
    axes[0].set_ylabel(
        "correlation of pv_total_kWh with itself k hours earlier"
    )

    # pacf plot

    plot_pacf(
        series,
        lags=lags,
        ax=axes[1]
    )

    axes[1].set_title(
        "training-set partial autocorrelation function (PACF)"
    )
    axes[1].set_xlabel(
        "lag k (hours before the current hour)"
    )
    axes[1].set_ylabel(
        "direct correlation at lag k, shorter lags removed"
    )

    for axis in axes:
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, alpha=0.3)

    # The shaded band is the 95% interval for zero correlation: spikes inside
    # it are indistinguishable from noise, spikes outside carry real signal.
    fig.suptitle(
        f"{series.name} over {lags} lags. "
        "Shaded band = 95% confidence interval for zero correlation; "
        "spikes inside it are not significant."
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

    train_end = int(len(df) * TRAIN_RATIO)
    training_df = clean_data(
        df.iloc[:train_end][["pv_total_kWh"]]
    )
    y = training_df["pv_total_kWh"]

    # analyze acf pacf

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "acf_pacf.png"

    analyze_acf_pacf(
        y,
        lags=168,
        output_path=output_path
    )

    metadata_path = RESULTS_DIR / "acf_pacf_metadata.csv"
    pd.DataFrame(
        [
            {
                "series": "pv_total_kWh",
                "data_scope": "training_only",
                "start": y.index[0],
                "end": y.index[-1],
                "observations": len(y),
                "lags": 168
            }
        ]
    ).to_csv(metadata_path, index=False)

    print(f"saved plot to {output_path}")
    print(f"saved metadata to {metadata_path}")


if __name__ == "__main__":

    main()

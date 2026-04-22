import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)


def analyze_acf_pacf(
    series,
    lags=48
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    plot_acf(
        series,
        lags=lags,
        ax=axes[0]
    )

    axes[0].set_title(
        "acf"
    )

    plot_pacf(
        series,
        lags=lags,
        ax=axes[1]
    )

    axes[1].set_title(
        "pacf"
    )

    plt.tight_layout()

    plt.show()
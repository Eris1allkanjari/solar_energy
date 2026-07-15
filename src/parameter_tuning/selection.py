import numpy as np
import pandas as pd

from src.configs.evaluation import SELECTION_MAE_TOLERANCE
from src.training.evaluation import evaluate


SELECTION_MAE_KEY = "val_block_mae_mean"
SELECTION_RMSE_KEY = "val_block_rmse_mean"


def calculate_monthly_validation_metrics(
    index,
    y_true,
    y_pred
):
    metrics_df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred)
        },
        index=pd.DatetimeIndex(index)
    )
    metrics_df["month"] = metrics_df.index.to_period("M")

    monthly_metrics = []

    for _, month_df in metrics_df.groupby("month"):
        mae, rmse, _, _ = evaluate(
            month_df["y_true"],
            month_df["y_pred"]
        )
        monthly_metrics.append(
            {
                "mae": mae,
                "rmse": rmse
            }
        )

    monthly_df = pd.DataFrame(
        monthly_metrics
    )

    return {
        "validation_blocks": len(monthly_df),
        "val_block_mae_mean": monthly_df["mae"].mean(),
        "val_block_mae_std": monthly_df["mae"].std(ddof=0),
        "val_block_rmse_mean": monthly_df["rmse"].mean(),
        "val_block_rmse_std": monthly_df["rmse"].std(ddof=0)
    }


def select_robust_candidate(
    results,
    mae_tolerance=SELECTION_MAE_TOLERANCE
):
    valid_results = [
        result
        for result in results
        if np.isfinite(float(result[SELECTION_MAE_KEY]))
    ]

    if not valid_results:
        raise ValueError(
            "no successful validation results are available"
        )

    best_mae = min(
        float(result[SELECTION_MAE_KEY])
        for result in valid_results
    )
    mae_limit = best_mae * (1 + mae_tolerance)

    eligible_results = [
        result
        for result in valid_results
        if float(result[SELECTION_MAE_KEY]) <= mae_limit
    ]

    return min(
        eligible_results,
        key=lambda result: (
            float(result[SELECTION_RMSE_KEY]),
            -int(result["seq_len"])
        )
    )

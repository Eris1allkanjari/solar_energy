import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred, production_threshold=5.0, epsilon=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    # percentage metrics only for meaningful pv production
    mask = y_true > production_threshold

    if np.any(mask):
        mape = np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask]) / y_true[mask]
            )
        ) * 100

        smape = np.mean(
            2 * np.abs(y_pred[mask] - y_true[mask]) /
            (
                np.abs(y_true[mask]) +
                np.abs(y_pred[mask]) +
                epsilon
            )
        ) * 100

    else:
        mape = np.nan
        smape = np.nan

    return mae, rmse, mape, smape
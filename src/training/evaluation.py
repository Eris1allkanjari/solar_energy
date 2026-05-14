import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true, y_pred, epsilon=1e-6, mape_threshold=1e-3):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    # mape only where actual production is not near zero
    mask = np.abs(y_true) > mape_threshold

    if np.any(mask):
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    else:
        mape = np.nan

    # symmetric mape, more stable around small values
    smape = np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + epsilon)
    ) * 100

    return mae, rmse, mape, smape
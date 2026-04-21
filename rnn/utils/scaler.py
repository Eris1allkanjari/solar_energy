import numpy as np


def inverse_target(scaler, y, num_features):
    """
    Inverse transform only the target variable (first column)
    """
    return scaler.inverse_transform(
        np.concatenate(
            [y.reshape(-1, 1), np.zeros((len(y), num_features - 1))],
            axis=1
        )
    )[:, 0]
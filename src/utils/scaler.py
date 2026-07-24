import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TargetFeatureScaler:
    """Scale the forecast target and exogenous features separately.

    The target (column 0, ``pv_total_kWh``) is scaled to ``[0, 1]`` with MinMax
    so the network output stays bounded, while the remaining feature columns are
    standardized. A shared MinMax fit across every column lets a few extreme
    values (e.g. solar-radiation spikes) compress the useful range of the other
    features; standardizing the features is far less sensitive to those tails.
    This also matches the ``StandardScaler`` treatment the autoregressive
    exogenous path already uses, keeping the two model families comparable.

    Pass ``feature_scaler=RobustScaler()`` to make feature scaling robust to
    outliers instead of standardizing.
    """

    def __init__(self, feature_scaler=None):
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = (
            feature_scaler
            if feature_scaler is not None
            else StandardScaler()
        )
        self._has_features = False

    def fit(self, data):
        values = np.asarray(data, dtype=float)
        self.target_scaler.fit(values[:, :1])
        self._has_features = values.shape[1] > 1

        if self._has_features:
            self.feature_scaler.fit(values[:, 1:])

        return self

    def transform(self, data):
        values = np.asarray(data, dtype=float)
        target = self.target_scaler.transform(values[:, :1])

        if not self._has_features:
            return target

        features = self.feature_scaler.transform(values[:, 1:])
        return np.concatenate([target, features], axis=1)

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_target(self, y):
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y)[:, 0]


def inverse_target(scaler, y, num_features=None):
    """
    Inverse transform only the target variable (first column).

    Works with a ``TargetFeatureScaler`` (which scales the target on its own)
    as well as a plain scikit-learn scaler fit on every column, where the
    other feature columns are padded with zeros before inverting.
    """
    if isinstance(scaler, TargetFeatureScaler):
        return scaler.inverse_target(y)

    if num_features is None:
        raise ValueError(
            "num_features is required to invert a whole-frame scaler"
        )

    return scaler.inverse_transform(
        np.concatenate(
            [y.reshape(-1, 1), np.zeros((len(y), num_features - 1))],
            axis=1
        )
    )[:, 0]

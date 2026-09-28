import copy

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Sliding input windows over a 2-D array, cut out on demand.

    Window i covers rows s to s+seq_len-1 of the data, s being the i-th start
    position, and its target is column 0 of the row that follows. Building all
    windows up front costs rows x seq_len x features of memory, which reaches
    several gigabytes at 10-minute resolution with a week of history. Only the
    source array is held here: once in float32 for the model, and once in its
    original precision for the targets that metrics are computed from.

    The windows, their order and their float32 values are exactly those an
    up-front construction produces, so training on either is identical.
    """

    def __init__(self, data, seq_len):
        values = np.asarray(data)

        if values.ndim != 2:
            raise ValueError("window data must be a 2-D array")

        if len(values) <= seq_len:
            raise ValueError(
                f"{len(values)} rows cannot hold a window of {seq_len} plus "
                "a target"
            )

        self.values = values
        self.inputs = torch.as_tensor(values, dtype=torch.float32)
        self.seq_len = seq_len
        self.starts = np.arange(len(values) - seq_len)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        end = start + self.seq_len
        return self.inputs[start:end], self.inputs[end, :1]

    @property
    def n_features(self):
        return self.values.shape[1]

    @property
    def targets(self):
        return self.values[self.starts + self.seq_len, 0]

    def subset(self, mask):
        """The windows a boolean mask selects, sharing the source array."""
        selected = copy.copy(self)
        selected.starts = self.starts[np.asarray(mask, dtype=bool)]
        return selected

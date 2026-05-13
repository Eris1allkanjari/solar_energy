import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()

        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=config.HIDDEN_UNITS_1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(config.DROPOUT)

        self.gru2 = nn.GRU(
            input_size=config.HIDDEN_UNITS_1,
            hidden_size=config.HIDDEN_UNITS_2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(config.DROPOUT)

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_UNITS_2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)

        x, _ = self.gru2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]

        return self.head(x)


def build_gru(input_shape, config):
    input_size = input_shape[1]
    return GRUModel(input_size, config)
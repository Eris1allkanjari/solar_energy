import torch.nn as nn


class BaseLSTM(nn.Module):
    def __init__(self, input_size, config, return_sequences_last=True):
        super().__init__()

        self.return_sequences_last = return_sequences_last

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_UNITS_1,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(config.DROPOUT)

        self.lstm2 = nn.LSTM(
            input_size=config.HIDDEN_UNITS_1,
            hidden_size=config.HIDDEN_UNITS_2,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        if not self.return_sequences_last:
            x = x[:, -1, :]

        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()

        self.base_lstm = BaseLSTM(
            input_size=input_size,
            config=config,
            return_sequences_last=False
        )

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_UNITS_2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.base_lstm(x)
        return self.head(x)


def build_lstm(input_shape, config):
    input_size = input_shape[1]
    return LSTMModel(input_size, config)
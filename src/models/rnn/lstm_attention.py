import torch
import torch.nn as nn

from src.models.rnn.lstm import BaseLSTM


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, config):
        super().__init__()

        self.base_lstm = BaseLSTM(
            input_size=input_size,
            config=config,
            return_sequences_last=True
        )
        self.attention_score = nn.Linear(
            config.HIDDEN_UNITS_2,
            1
        )
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_UNITS_2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        sequence = self.base_lstm(x)
        weights = torch.softmax(
            self.attention_score(sequence),
            dim=1
        )
        context = torch.sum(weights * sequence, dim=1)
        return self.head(context)


def build_lstm_attention(input_shape, config):
    input_size = input_shape[1]
    return LSTMAttentionModel(input_size, config)

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

        self.attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_UNITS_2,
            num_heads=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(config.DROPOUT)

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_UNITS_2, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.base_lstm(x)

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        x = attention_output.mean(dim=1)

        x = self.dropout(x)

        return self.head(x)


def build_lstm_attention(input_shape, config):
    input_size = input_shape[1]
    return LSTMAttentionModel(input_size, config)
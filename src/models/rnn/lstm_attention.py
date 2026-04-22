from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)

from src.models.rnn.lstm import base_lstm


def build_lstm_attention(input_shape, config):
    inputs = Input(shape=input_shape)

    # reuse base LSTM
    x = base_lstm(inputs, config, return_sequences_last=True)

    # attention layer
    attention = Attention()([x, x])

    # reduce time dimension
    x = GlobalAveragePooling1D()(attention)

    # dense head
    x = Dense(32, activation="relu")(x)
    x = Dropout(config.DROPOUT)(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    return model
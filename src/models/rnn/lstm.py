from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model



def base_lstm(input_tensor, config, return_sequences_last=True):
    """
    Shared LSTM base (reusable for attention / non-attention models)
    """

    x = LSTM(config.HIDDEN_UNITS_1, return_sequences=True)(input_tensor)
    x = Dropout(config.DROPOUT)(x)

    x = LSTM(config.HIDDEN_UNITS_2, return_sequences=return_sequences_last)(x)
    x = Dropout(config.DROPOUT)(x)

    return x


def build_lstm(input_shape, config):
    inputs = Input(shape=input_shape)

    x = base_lstm(inputs, config, return_sequences_last=False)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    return model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input

def build_gru(input_shape, config):
    model = Sequential([
        Input(shape=input_shape),

        GRU(config.HIDDEN_UNITS_1, return_sequences=True),
        Dropout(config.DROPOUT),

        GRU(config.HIDDEN_UNITS_2),
        Dropout(config.DROPOUT),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    return model
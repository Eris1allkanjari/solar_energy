class BaseConfig:
    SEQ_LEN = 48

    HIDDEN_UNITS_1 = 128
    HIDDEN_UNITS_2 = 64

    DROPOUT = 0.2

    LEARNING_RATE = 0.001

    EPOCHS = 30
    BATCH_SIZE = 64

    PATIENCE = 5


class SmallModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 64
    HIDDEN_UNITS_2 = 32


class LargeModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 256
    HIDDEN_UNITS_2 = 128


class HighDropoutConfig(BaseConfig):
    DROPOUT = 0.4
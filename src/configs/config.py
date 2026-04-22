class BaseConfig:
    SEQ_LEN = 48

    # model
    HIDDEN_UNITS_1 = 128
    HIDDEN_UNITS_2 = 64
    DENSE_UNITS = 32
    DROPOUT = 0.2

    # training
    LEARNING_RATE = 0.001
    EPOCHS = 30
    BATCH_SIZE = 64

class SmallModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 64
    HIDDEN_UNITS_2 = 32


class LargeModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 256
    HIDDEN_UNITS_2 = 128


class HighDropoutConfig(BaseConfig):
    DROPOUT = 0.4


class LowLearningRateConfig(BaseConfig):
    LEARNING_RATE = 0.0005
class BaseConfig:
    SEQ_LEN = 48

    HIDDEN_UNITS_1 = 128
    HIDDEN_UNITS_2 = 64

    DROPOUT = 0.2

    LEARNING_RATE = 0.001

    EPOCHS = 30
    BATCH_SIZE = 64

    PATIENCE = 5
    LOSS = "huber"
    HUBER_DELTA = 0.1
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0


class SmallModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 64
    HIDDEN_UNITS_2 = 32


class LargeModelConfig(BaseConfig):
    HIDDEN_UNITS_1 = 256
    HIDDEN_UNITS_2 = 128


class HighDropoutConfig(BaseConfig):
    DROPOUT = 0.4

class ExperimentConfig:
    def __init__(self, params, seq_len):
        self.SEQ_LEN = seq_len

        self.HIDDEN_UNITS_1 = params["hidden_units_1"]
        self.HIDDEN_UNITS_2 = params["hidden_units_2"]
        self.DROPOUT = params["dropout"]
        self.LEARNING_RATE = params["learning_rate"]
        self.BATCH_SIZE = params["batch_size"]
        self.EPOCHS = params["epochs"]
        self.PATIENCE = params["patience"]
        self.LOSS = params.get("loss", "huber")
        self.HUBER_DELTA = params.get("huber_delta", 0.1)
        self.WEIGHT_DECAY = params.get("weight_decay", 1e-5)
        self.GRADIENT_CLIP = params.get("gradient_clip", 1.0)

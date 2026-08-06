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
        self.LOSS = params.get("loss", "mae")
        # Only read when LOSS is "huber", which the loss comparison experiment
        # sets explicitly; the tuning grid does not use it.
        self.HUBER_DELTA = params.get("huber_delta", 0.1)
        self.WEIGHT_DECAY = params.get("weight_decay", 1e-5)
        self.GRADIENT_CLIP = params.get("gradient_clip", 1.0)

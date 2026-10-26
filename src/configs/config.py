# Per-epoch learning-rate multiplier. The final retraining on train+validation
# runs for a fixed number of epochs with nothing to stop it at a good point, and
# at a constant rate its output scale was still drifting when it ended, by a
# different amount for each seed. Decaying the rate lets training settle. The
# schedule depends only on the epoch number, so a retraining for N epochs
# follows exactly the path tuning took up to epoch N.
DEFAULT_LR_DECAY = 0.85


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
        self.LR_DECAY = params.get("lr_decay", DEFAULT_LR_DECAY)

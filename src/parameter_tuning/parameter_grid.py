from src.configs.config import DEFAULT_LR_DECAY
from src.configs.dataset_selection import hours_to_steps

# Input history in hours, converted to rows of the active dataset, so a
# 10-minute model sees the same span of time as an hourly one.
SEQ_LENGTH_HOURS = [24, 48, 72, 96, 168]
SEQ_LENGTHS = [hours_to_steps(hours) for hours in SEQ_LENGTH_HOURS]

HYPERPARAMETER_GRIDS = {
    "lstm": {
        "hidden_units_1": [128, 256],
        "hidden_units_2": [64, 128],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001],
        "batch_size": [64],
        "epochs": [30],
        "patience": [5],
        "loss": ["mae"],
        "weight_decay": [1e-5],
        "gradient_clip": [1.0],
        "lr_decay": [DEFAULT_LR_DECAY]
    },

    "gru": {
        "hidden_units_1": [128, 256],
        "hidden_units_2": [64, 128],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001],
        "batch_size": [64],
        "epochs": [30],
        "patience": [5],
        "loss": ["mae"],
        "weight_decay": [1e-5],
        "gradient_clip": [1.0],
        "lr_decay": [DEFAULT_LR_DECAY]
    },

    "attention": {
        "hidden_units_1": [128],
        "hidden_units_2": [64],
        "dropout": [0.2, 0.3],
        "learning_rate": [0.001],
        "batch_size": [64],
        "epochs": [30],
        "patience": [5],
        "loss": ["mae"],
        "weight_decay": [1e-5],
        "gradient_clip": [1.0],
        "lr_decay": [DEFAULT_LR_DECAY]
    }
}

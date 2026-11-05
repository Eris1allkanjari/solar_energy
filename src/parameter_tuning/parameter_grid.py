from src.configs.config import DEFAULT_LR_DECAY
from src.configs.dataset_selection import hours_to_steps

# Input history in hours, converted to rows of the active dataset, so a
# 10-minute model sees the same span of time as an hourly one. 12 and 336 hours
# extend the original 24-168 range, whose ends were each chosen by a model.
SEQ_LENGTH_HOURS = [12, 24, 48, 72, 96, 168, 336]
SEQ_LENGTHS = [hours_to_steps(hours) for hours in SEQ_LENGTH_HOURS]

# Values beyond the grid below, where its best settings sat on an edge. They are
# not crossed with each other, which would multiply the grid about twentyfold.
# For each window length, the best configuration of the grid is taken and each
# value here is tried in it, one parameter at a time.
EXTENSION_VALUES = {
    "hidden_units_1": [512],
    "hidden_units_2": [32, 256],
    "dropout": [0.1],
    "learning_rate": [0.0005, 0.002]
}

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

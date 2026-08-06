SEQ_LENGTHS = [24, 48, 72, 96, 168]

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
        "gradient_clip": [1.0]
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
        "gradient_clip": [1.0]
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
        "gradient_clip": [1.0]
    }
}

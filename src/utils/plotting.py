import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, title="Forecast"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("PV Output (kWh)")
    plt.tight_layout()
    plt.show()


def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
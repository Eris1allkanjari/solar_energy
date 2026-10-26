import torch
from torch.utils.data import DataLoader


def predict_model(model, windows, batch_size=64):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    model = model.to(device)

    loader = DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(device, non_blocking=pin_memory)

            y_pred = model(X_batch)

            predictions.append(
                y_pred.cpu()
            )

    predictions = torch.cat(
        predictions,
        dim=0
    )

    return predictions.numpy()

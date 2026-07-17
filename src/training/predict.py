import torch
from torch.utils.data import DataLoader, TensorDataset


def predict_model(model, X_test, batch_size=64):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    model = model.to(device)

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    dataset = TensorDataset(X_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for (X_batch,) in loader:

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

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn


def train_model(model, X_train, y_train, X_val, y_val, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"epoch {epoch + 1}/{config.EPOCHS} "
            f"train loss: {train_loss:.6f} "
            f"val loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            patience_counter += 1

            if patience_counter >= config.PATIENCE:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
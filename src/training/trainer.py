import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_loss(config):
    if config.LOSS == "huber":
        return nn.HuberLoss(
            delta=config.HUBER_DELTA
        )

    if config.LOSS == "mae":
        return nn.L1Loss()

    if config.LOSS == "mse":
        return nn.MSELoss()

    raise ValueError(
        f"unsupported loss: {config.LOSS}"
    )


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    seed=42,
    epochs=None
):
    set_random_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    val_loader = None

    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

    criterion = build_loss(
        config
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    training_epochs = epochs or config.EPOCHS

    for epoch in range(training_epochs):
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

            if config.GRADIENT_CLIP is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.GRADIENT_CLIP
                )

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if val_loader is None:
            print(
                f"epoch {epoch + 1}/{training_epochs} "
                f"train loss: {train_loss:.6f}"
            )
            continue

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
            f"epoch {epoch + 1}/{training_epochs} "
            f"train loss: {train_loss:.6f} "
            f"val loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
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

    if val_loader is not None:
        model.best_epoch = best_epoch or training_epochs
    else:
        model.best_epoch = training_epochs

    return model

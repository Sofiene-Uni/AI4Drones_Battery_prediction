import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from pathlib import Path

from src.utils.config import get_value
from src.utils.dataset import load_split
from src.utils.reports import log_training
from src.utils.model_io import save_model
from src.models.build_model import build_model_from_config


def run():
    # Load paths and training config safely
    prepared_dir = Path(get_value("paths.prepared_dir"))

    training_cfg = get_value("model.training", default={})
    batch_size = training_cfg.get("batch_size", 32)
    lr         = training_cfg.get("lr", 0.001)
    epochs     = training_cfg.get("epochs", 10)
    device     = training_cfg.get("device", "cpu")

    # Load train split
    X_train, y_train, _ = load_split(prepared_dir / "train")
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=False
    )
        
    # Build model
    model, device = build_model_from_config()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            loop.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(train_loader.dataset)
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.6f}")

        # Log epoch loss
        log_training(epoch+1, avg_loss)
        
    # Save model    
    save_model(model)

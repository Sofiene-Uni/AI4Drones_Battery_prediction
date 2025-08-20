
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
from src.models.build_model import build_model_from_config
from src.utils.config import  load_config
import numpy as np 

def load_split(split_dir: Path):
    """Load all features and labels from a given split (train/test/validate)."""
    features_dir = split_dir / "features"
    labels_dir = split_dir / "labels"
    
    # Collect all file IDs (same in both features and labels)
    file_ids = sorted([f.stem for f in features_dir.glob("*.csv")])
    
    X_list, y_list = [], []
    for fid in file_ids:
        X = pd.read_csv(features_dir / f"{fid}.csv").values
        y = pd.read_csv(labels_dir / f"{fid}.csv").values
        
        X_list.append(X)
        y_list.append(y)
    
    # Concatenate all flights into one big array
    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    
    return torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.float32)


def run():
    config = load_config()
    cfg = load_config()
    prepared_dir = Path(cfg["paths"]["prepared_dir"])
    
    # Load train split
    X_train, y_train = load_split(prepared_dir / "train")
    


    # DataLoader
    batch_size = config.get("training", {}).get("batch_size", 32)
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model
    model, device = build_model_from_config(config, X_train)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("training", {}).get("lr", 1e-3))

    # Training loop
    epochs = config.get("training", {}).get("epochs", 10)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader: 
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save trained model
    model_path = Path("saved_models") / f"{config['model']['model_name']}_model.pth"
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

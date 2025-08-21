from datetime import datetime
from pathlib import Path
import torch
from src.utils.config import  load_config


def save_model(model):
    """Save model with timestamp in filename, using config for path + name."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["model_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{model_name}_model_{timestamp}.pth"

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    return model_path


def load_model_state(model, device="cpu"):
    """Load the most recently saved model (based on config) into the given model."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    model_name = cfg["model"]["model_name"]

    candidates = sorted(models_dir.glob(f"{model_name}_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No saved model found for {model_name} in {models_dir}")

    model_path = candidates[-1]  # newest (sorted by timestamp in filename)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    

    print(f"✅ Loaded model from {model_path}")
    return model
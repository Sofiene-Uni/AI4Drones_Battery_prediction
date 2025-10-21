from datetime import datetime
from pathlib import Path
import torch
import joblib
from src.utils.config import load_config


def save_model_onnx(model, input_size=14):
    """
    Export model to ONNX with timestamp in filename, using config for path + name.
    """
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["model_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{model_name}_model_{timestamp}.onnx"

    # Dummy input for tracing
    dummy_input = torch.randn(1, input_size, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )

    print(f"✅ Model exported to ONNX at {model_path}")
    return model_path



def save_model(model):
    """Save model state_dict with timestamp in filename, using config for path + name."""
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
    """Load the most recently saved model state_dict into the given model."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    model_name = cfg["model"]["model_name"]

    candidates = sorted(models_dir.glob(f"{model_name}_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No saved model found for {model_name} in {models_dir}")

    model_path = candidates[-1]  # newest file
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print(f"✅ Loaded model from {model_path}")
    return model


def save_scaler(scaler):
    """Save the fitted scaler with timestamp in filename, using config for path + name."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["model_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler_path = models_dir / f"{model_name}_scaler_{timestamp}.pkl"

    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")
    return scaler_path


def load_scaler():
    """Load the most recently saved scaler for the current model_name."""
    cfg = load_config()
    models_dir = Path(cfg["paths"]["models_dir"])
    model_name = cfg["model"]["model_name"]

    candidates = sorted(models_dir.glob(f"{model_name}_scaler_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No saved scaler found for {model_name} in {models_dir}")

    scaler_path = candidates[-1]  # newest scaler
    scaler = joblib.load(scaler_path)

    print(f"✅ Loaded scaler from {scaler_path}")
    return scaler

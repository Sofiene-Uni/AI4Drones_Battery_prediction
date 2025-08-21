# src/train_test_models/utils.py
import torch
from ..models.fnn import FNN
from ..models.fnn_slide import FNN_SLIDE
from ..models.lstm import LSTM
from ..models.rnn import RNN
from ..models.transformer import TRANSFORMER
from torchinfo import summary
from src.utils.config import get_value


def build_model_from_config():
    """Build a model from config.yaml using get_value() and return model and device."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model name from config
    model_name = get_value("model.model_name")
    if not model_name:
        raise ValueError("Model name not found in config.")

    # Get architecture settings
    arch_cfg = get_value(f"model.architecture.{model_name}", default={})
    input_size = arch_cfg.get("input_size", 0)
    hidden_size = arch_cfg.get("hidden_size", 0)
    output_size = arch_cfg.get("output_size", 0)

    # Build the appropriate model
    if model_name == "mlp":
        model = FNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "mlp_sliding":
        model = FNN_SLIDE(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "lstm":
        model = LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "rnn":
        model = RNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "transformer":
        model = TRANSFORMER(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)

    print(summary(model, input_size=(1, input_size), col_names=("input_size", "output_size", "num_params")))

    return model, device
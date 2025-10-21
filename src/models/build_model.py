# src/train_test_models/utils.py
import torch
from ..models.fnn import FNN
from ..models.fnn_slide import FNN_SLIDE
from ..models.lstm import LSTM
from ..models.rnn import RNN
from ..models.transformer import TRANSFORMER
from torchinfo import summary
from src.utils.config import get_value


def build_model_from_config(X_train=None, y_train=None, input_size=None, output_size=None):
    """
    Build a model from config.yaml using get_value().
    Input/output sizes are inferred from training data if not provided.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model name from config
    model_name = get_value("model.model_name")
    if not model_name:
        raise ValueError("Model name not found in config.")

    # Infer input/output sizes if not provided
    if input_size is None:
        if X_train is None:
            raise ValueError("X_train must be provided if input_size is not specified.")
        input_size = X_train.shape[1] if X_train.ndim > 1 else 1

    if output_size is None:
        if y_train is None:
            raise ValueError("y_train must be provided if output_size is not specified.")
        output_size = y_train.shape[1] if y_train.ndim > 1 else 1

    # Get architecture settings (e.g., hidden size)
    arch_cfg = get_value(f"model.architecture.{model_name}", default={})
    hidden_size = arch_cfg.get("hidden_size", 64)

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

    try:
        print(summary(model, input_size=(1, input_size), col_names=("input_size", "output_size", "num_params")))
    except Exception:
        pass  # summary might fail for some models

    return model, device

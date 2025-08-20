# src/train_test_models/utils.py
import torch
from ..models.fnn import FNN
from ..models.fnn_slide import FNN_SLIDE
from ..models.lstm import LSTM
from ..models.rnn import RNN
from ..models.transformer import TRANSFORMER
from torchinfo import summary


def build_model_from_config(config, X_train, output_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["model"]["model_name"]
    arch_cfg = config["model"]["architecture"][model_name]
    hidden_size = arch_cfg.get("hidden_size", 0)


    # Determine input size depending on model type
    if model_name in ["mlp", "mlp_sliding"]:
        # FNN: each feature is a separate node
        input_size_model = X_train.shape[1]
          
    elif model_name in ["rnn", "lstm", "transformer"]:
        # Sequence models: input per time step
        ...
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Build model
    if model_name == "mlp":
        model = FNN(input_size=input_size_model, output_size=output_size, hidden_size=hidden_size)
        
 
        
    elif model_name == "mlp_sliding":
        model = FNN_SLIDE(input_size=input_size_model, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "lstm":
        model = LSTM(input_size=input_size_model, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "rnn":
        model = RNN(input_size=input_size_model, output_size=output_size, hidden_size=hidden_size)
    elif model_name == "transformer":
        model = TRANSFORMER(input_size=input_size_model, output_size=output_size, hidden_size=hidden_size)

    model = model.to(device)
    print (summary(model, input_size=(1, input_size_model), col_names=("input_size","output_size","num_params")))



    return model, device

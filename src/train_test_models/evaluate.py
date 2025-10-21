import torch
from pathlib import Path
from src.utils.dataset import  load_split
from src.utils.reports import postprocess_evaluation
from src.models.build_model import build_model_from_config
from src.utils.config import  get_value
from src.utils.model_io import load_model_state




def evaluate(model, device, X, y, file_ids=None, batch_size=32, mode="sample"):
    """Evaluate a trained model and log/visualize results given dataset tensors."""

    # --- create DataLoader ---
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # --- evaluation loop ---
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss, all_preds, all_targets = 0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            total_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    avg_loss = total_loss / len(loader.dataset)
    
    
    all_preds = torch.vstack(all_preds).numpy()
    all_targets = torch.vstack(all_targets).numpy()
    
    

    postprocess_evaluation(
        all_preds=all_preds,
        all_targets=all_targets,
        file_ids=file_ids,
        avg_loss=avg_loss,
        mode=mode
    )


def run():
    """Load the latest saved model and evaluate on the test split."""

    # --- load config ---
    test_dir = Path(get_value("paths.prepared_dir"), "test")
    batch_size = get_value("model.training.batch_size", default=32)
    mode = get_value("evaluation.mode", default="flight")
    

    # --- load test data ---
    X_test, y_test, file_ids = load_split(test_dir)

    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    


    # --- build model using test data to infer sizes ---
    model, device = build_model_from_config(X_test, y_test)

    # --- load most recent checkpoint ---
    model = load_model_state(model, device=device)
    model.to(device)

    # --- evaluate ---
    evaluate(model, device, X_test, y_test, file_ids=file_ids, batch_size=batch_size, mode=mode)

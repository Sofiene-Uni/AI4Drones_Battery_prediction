import pandas as pd
from pathlib import Path
from src.utils.config import  get_value
from src.utils.dataset import  fit_scaler,assign_splits
from src.utils.logbook import get_logbook


def prepare_files(logbook, scaler, processed_dir, prepared_dir):
    """
    Prepare all valid files based on logbook splits:
    - Split features/labels
    - Apply scaling using fitted scaler
    - Save into correct split directories
    """
    LABEL_COL = get_value("data.label_col",[])
    DROP_COLS = get_value("data.drop_cols", [])
    CATEGORICAL_COLS = get_value("data.categorical_cols", [])

    prepared_counts = {"train": 0, "test": 0, "validate": 0}

    # Filter only processed AND valid files
    valid_logbook = logbook[(logbook["processed"] == True) & (logbook["valid"] == True)]
    for row in valid_logbook.itertuples():
        # Load file
        df = pd.read_csv(processed_dir / f"{row.id}.csv")

        # Separate label
        y = df[LABEL_COL]
        X = df.drop(columns=[LABEL_COL] + DROP_COLS)

        # Scale numeric columns
        scale_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
        if scaler is not None and scale_cols:
            X[scale_cols] = scaler.transform(X[scale_cols])

        # Determine split
        split = None
        if getattr(row, "use_train", False):
            split = "train"
        elif getattr(row, "use_test", False):
            split = "test"
        elif getattr(row, "use_validate", False):
            split = "validate"
        if not split:
            continue  # Skip if not assigned

        # Ensure directories exist
        (prepared_dir / split / "features").mkdir(parents=True, exist_ok=True)
        (prepared_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Save features and labels
        X.to_csv(prepared_dir / split / "features" / f"{row.id}.csv", index=False)
        y.to_csv(prepared_dir / split / "labels" / f"{row.id}.csv", index=False)

        prepared_counts[split] += 1

    print(
        f"✅ Preparation finished. Files saved → "
        f"Train: {prepared_counts['train']}, "
        f"Test: {prepared_counts['test']}, "
        f"Validate: {prepared_counts['validate']}"
    )
    return prepared_counts


def run():
    """Run the full data preparation pipeline with only valid files."""
    processed_dir = Path(get_value("paths.processed_dir"))
    prepared_dir = Path(get_value("paths.prepared_dir"))
    logbook = get_logbook()
    
    # Step 1: Assign splits (only affects flags, doesn't filter)
    logbook = assign_splits(
        logbook, 
            train_frac = get_value("data.splits.train_frac", 0.7),
            test_frac = get_value("data.splits.test_frac", 0.2),
            validate_frac = get_value("data.splits.validate_frac", 0.1),
            seed = get_value("data.splits.seed", 42),
                )
    
    # Step 2: Fit scaler only on training data (valid + processed)
    scaler = fit_scaler(
        logbook[(logbook["processed"] == True) & (logbook["valid"] == True)], 
        processed_dir, 
        prepared_dir,
        drop_cols=get_value("data.drop_cols", []), 
        categorical_cols=get_value("data.categorical_cols", []),
    )
    
    # Step 3: Prepare all valid files
    prepare_files(logbook, scaler, processed_dir, prepared_dir)

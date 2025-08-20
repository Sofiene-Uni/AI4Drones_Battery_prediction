import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from src.utils.config import  load_config
from src.utils.logbook import get_logbook,save_logbook
import numpy as np 


LABEL_COL ="battery_status_current_a"

DROP_COLS = [
    "timestamp", "sensor_gps_time_utc_usec", "date", "hour", "minute",
    "second","battery_status_current_a", "battery_status_voltage_v"
]

CATEGORICAL_COLS = ["mode", "flight_phase", "vehicle_type"]




def assign_splits(logbook: pd.DataFrame, train_frac=0.7, test_frac=0.2, validate_frac=0.1, seed: int = 42) -> pd.DataFrame:
    """
    Randomly assign logbook entries to train, test, or validate splits and update the logbook.

    Parameters:
        logbook (pd.DataFrame): Current logbook dataframe.
        train_frac (float): Fraction of entries for training.
        test_frac (float): Fraction of entries for testing.
        validate_frac (float): Fraction of entries for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Updated logbook with 'use_train', 'use_test', 'use_validate' columns.
    """
    np.random.seed(seed)



    # Shuffle indices
    shuffled_ids = logbook["id"].sample(frac=1, random_state=seed).tolist()
    n = len(shuffled_ids)

    n_train = int(n * train_frac)
    n_test = int(n * test_frac)

    train_ids = shuffled_ids[:n_train]
    test_ids = shuffled_ids[n_train:n_train + n_test]
    validate_ids = shuffled_ids[n_train + n_test:]

    # Reset all first
    logbook[["use_train", "use_test", "use_validate"]] = False

    # Update splits
    logbook.loc[logbook["id"].isin(train_ids), "use_train"] = True
    logbook.loc[logbook["id"].isin(test_ids), "use_test"] = True
    logbook.loc[logbook["id"].isin(validate_ids), "use_validate"] = True

    save_logbook(logbook)
    #(f"🎯 Assigned splits → Train: {len(train_ids)}, Test: {len(test_ids)}, Validate: {len(validate_ids)}")
    return logbook


def fit_scaler(logbook, processed_dir, prepared_dir):
    """
    Fit a scaler only on numeric features from TRAIN split files,
    excluding unwanted and categorical columns.
    """
    prepared_dir.mkdir(parents=True, exist_ok=True)

    all_features = []

    # Only iterate over train files
    for row in logbook[(logbook["processed"]) & (logbook["use_train"])].itertuples():
        df = pd.read_csv(processed_dir / f"{row.id}.csv")

        # Drop unwanted + categorical columns
        X = df.drop(columns=[col for col in DROP_COLS + CATEGORICAL_COLS if col in df.columns])
        all_features.append(X)

    if all_features:
        all_features_concat = pd.concat(all_features, axis=0)

        scaler = StandardScaler().fit(all_features_concat)
        joblib.dump(scaler, prepared_dir / "scaler.pkl")

        print(f"✅ Scaler fitted on {len(all_features)} TRAIN files and saved.")
        return scaler
    else:
        print("⚠️ No TRAIN files found to fit the scaler.")
        return None

def prepare_single_file(row, scaler, processed_dir, prepared_dir):
    """Prepare one file: split features/labels, apply scaling using the fitted scaler, save into correct split."""
    df = pd.read_csv(processed_dir / f"{row.id}.csv")
    
    # Separate label
    y = df[LABEL_COL]
    
    # Drop columns that should not be included in features
    X = df.drop(columns=[LABEL_COL] + DROP_COLS)

    # Determine numeric columns to scale (excluding categorical)
    scale_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    
    # Apply scaling using the fitted scaler (fit was done only on TRAIN)
    X_scaled = X.copy()
    if scaler is not None and scale_cols:
        X_scaled[scale_cols] = scaler.transform(X[scale_cols])
    
    # Determine split
    if getattr(row, "use_train", False):
        split = "train"
    elif getattr(row, "use_test", False):
        split = "test"
    elif getattr(row, "use_validate", False):
        split = "validate"
    else:
        return None  # Skip if not assigned
    
    # Ensure directories exist
    (prepared_dir / split / "features").mkdir(parents=True, exist_ok=True)
    (prepared_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Save features and labels
    X_scaled.to_csv(prepared_dir / split / "features" / f"{row.id}.csv", index=False)
    y.to_csv(prepared_dir / split / "labels" / f"{row.id}.csv", index=False)
    
    return split



def prepare_all_files(logbook, scaler, processed_dir, prepared_dir):
    """Prepare all files based on logbook splits (train/test/validate)."""
    prepared_counts = {"train": 0, "test": 0, "validate": 0}
    
    for row in logbook[logbook["processed"] == True].itertuples():
        split = prepare_single_file(row, scaler,processed_dir, prepared_dir)
        if split:
            prepared_counts[split] += 1
    
    print(f"✅ Preparation finished. Files saved → Train: {prepared_counts['train']}, "
          f"Test: {prepared_counts['test']}, Validate: {prepared_counts['validate']}")
    return prepared_counts


def run():
    """Run the full data preparation pipeline."""
    cfg = load_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    prepared_dir = Path(cfg["paths"]["prepared_dir"])
    

    logbook = get_logbook()
    
    # Step 1: Assign splits
    logbook = assign_splits(logbook, train_frac=0.7, test_frac=0.2, validate_frac=0.1, seed=42)
    
    # Step 2: Fit scaler only on training data
    scaler = fit_scaler(logbook, processed_dir, prepared_dir)
    
    
    # Step 3: Prepare all files
    prepare_all_files(logbook, scaler, processed_dir, prepared_dir)
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np 
from src.utils.logbook import save_logbook


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
    #(f" Assigned splits → Train: {len(train_ids)}, Test: {len(test_ids)}, Validate: {len(validate_ids)}")
    return logbook


def load_split(split_dir):
    """Load all features and labels from a given split (train/test/validate). Always returns file_ids expanded per row."""
    features_dir = split_dir / "features"
    labels_dir = split_dir / "labels"
    
    file_ids = sorted([f.stem for f in features_dir.glob("*.csv")])
    
    X_list, y_list = [], []
    lengths = []  # track number of rows per file
    
    for fid in file_ids:
        X = pd.read_csv(features_dir / f"{fid}.csv").values
        y = pd.read_csv(labels_dir / f"{fid}.csv").values
        X_list.append(X)
        y_list.append(y)
        lengths.append(len(X))
    
    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)

    # repeat each file_id according to its row count
    expanded_ids = np.concatenate([[fid] * n for fid, n in zip(file_ids, lengths)])
    return X_all, y_all, expanded_ids



def fit_scaler(logbook, processed_dir, prepared_dir ,drop_cols , categorical_cols):
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
        X = df.drop(columns=[col for col in drop_cols + categorical_cols if col in df.columns])
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
import pandas as pd
from pathlib import Path
from src.utils.config import  get_value
from src.utils.dataset import  fit_scaler,assign_splits
from src.data_preprocessing import feature_engineering
from src.utils.logbook import get_logbook
from src.utils.model_io import save_scaler


def prepare_files(logbook, scaler, processed_dir, prepared_dir):
    """
    Prepare valid files:
    - Train: full logs
    - Test/validate: sample N points per log and save a simulation CSV with raw voltage/current values
    - Also saves unscaled features in a separate Excel file for inspection.
    """
    LABEL_COL, DROP_COLS, CATEGORICAL_COLS = get_value(
        ["data.label_col", "data.drop_cols", "data.categorical_cols"]
    )
    n_samples = get_value("data.reduction.n_samples", 10)
    sample_enabled = bool(get_value("data.reduction.sample", False))
    prepared_counts = {"train": 0, "test": 0, "validate": 0}

    valid_logbook = logbook[(logbook["processed"]) & (logbook["valid"])]

    for row in valid_logbook.itertuples():
        df = pd.read_csv(processed_dir / f"{row.id}.csv")

        # Determine split
        split = (
            "train" if getattr(row, "use_train", False) else
            "test" if getattr(row, "use_test", False) else
            "validate" if getattr(row, "use_validate", False) else None
        )
        if not split:
            continue

        # Apply sampling ONLY for test/validate
        if split in ["test", "validate"] and sample_enabled:
            print(f"🔹 Sampling {n_samples} geolocation points for {split} set (ID: {row.id})")
            df = feature_engineering.sample_lines(df, n_samples=n_samples)

        # Separate features/labels
        y = df[LABEL_COL]
        X = df.drop(columns=[LABEL_COL] + DROP_COLS)

        # --- 💾 Save simulation data (raw voltage/current) ---
        if split in ["test", "validate"]:
            simulation_dir = prepared_dir / split / "simulation"
            simulation_dir.mkdir(parents=True, exist_ok=True)
            simulation_cols = [
                "cumulative_flight_time_s",
                "sensor_baro_temperature",
                "battery_status_voltage_v",
                "battery_status_current_a"
            ]
            simulation_df = df[simulation_cols].copy()
            simulation_df.to_csv(simulation_dir / f"{row.id}.csv", index=False)

        # # --- 💾 Save unscaled features before scaling ---
        # unscaled_dir = prepared_dir / split / "features_not_scaled"
        # unscaled_dir.mkdir(parents=True, exist_ok=True)
        # unscaled_path = unscaled_dir / f"{row.id}_features_not_scaled.xlsx"
        # X.to_excel(unscaled_path, index=False)
       
        # --- Scale numeric columns (AFTER saving raw/unscaled data) ---
        scale_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
        if scaler is not None and scale_cols:
            X[scale_cols] = scaler.transform(X[scale_cols])

        # --- 💾 Save scaled features and labels ---
        feature_dir = prepared_dir / split / "features"
        label_dir = prepared_dir / split / "labels"
        feature_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        X.to_csv(feature_dir / f"{row.id}.csv", index=False)
        y.to_csv(label_dir / f"{row.id}.csv", index=False)

        prepared_counts[split] += 1

    print(f"✅ Prep done → Train: {prepared_counts['train']}, "
          f"Test: {prepared_counts['test']}, "
          f"Validate: {prepared_counts['validate']}")
    return prepared_counts





def run():
    """Run the full data preparation pipeline with only valid files."""
    processed_dir,prepared_dir= get_value(["paths.processed_dir","paths.prepared_dir"])
    
    
    
    prepared_dir=Path(prepared_dir)
    processed_dir=Path(processed_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
   
    logbook = get_logbook()
    
    # Step 1: Assign splits
    logbook = assign_splits(
        logbook,
        train_frac=get_value("data.splits.train_frac", 0.7),
        test_frac=get_value("data.splits.test_frac", 0.2),
        validate_frac=get_value("data.splits.validate_frac", 0.1),
        seed=get_value("data.splits.seed", 42),
    )
    
    # Step 2: Fit scaler only on training data (valid + processed)
    scaler = fit_scaler(
        logbook[(logbook["processed"]) & (logbook["valid"])],
        processed_dir,
        prepared_dir,
        drop_cols=get_value("data.drop_cols", []),
        categorical_cols=get_value("data.categorical_cols", []),
    )
    
    save_scaler(scaler)
    
    # Step 3: Prepare all valid files
    prepare_files(logbook, scaler, processed_dir, prepared_dir)

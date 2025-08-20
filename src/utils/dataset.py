import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path

class FlightDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: Path to prepared data folder (e.g., data/prepared)
        split: "train" | "test" | "validate"
        """
        self.features_dir = Path(root_dir) / split / "features"
        self.labels_dir = Path(root_dir) / split / "labels"
        self.transform = transform

        # Match feature files with corresponding label files
        self.feature_files = sorted(self.features_dir.glob("*_X.csv"))
        self.label_files = sorted(self.labels_dir.glob("*_y.csv"))

        assert len(self.feature_files) == len(self.label_files), \
            "Mismatch between features and labels count!"

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        X = pd.read_csv(self.feature_files[idx]).values.astype("float32")
        y = pd.read_csv(self.label_files[idx]).values.astype("float32").ravel()

        # Convert to PyTorch tensors
        X = torch.tensor(X)
        y = torch.tensor(y)

        if self.transform:
            X, y = self.transform(X, y)

        return X, y

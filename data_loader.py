"""
Real data loader for training.
Handles CSV, NPY, NPZ, image folders, text files.
Produces PyTorch DataLoader objects ready for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import json
import logging

log = logging.getLogger("DataLoader")


# ─────────────────────────────────────────────────────────────────────────────
# CSV DATASET  (handles cybersecurity_intrusion_data.csv and generic CSVs)
# ─────────────────────────────────────────────────────────────────────────────

class CSVDataset(Dataset):
    """
    Loads a CSV file, one-hot encodes categoricals, normalises numerics.
    Auto-detects the label column (last column, or 'attack_detected' if present).
    """

    def __init__(self, path: str, label_col: Optional[str] = None,
                 drop_cols: Optional[List[str]] = None):
        self.path = path
        df = pd.read_csv(path)

        # Drop obvious ID columns
        id_cols = [c for c in df.columns
                   if c.lower() in ("session_id", "id", "index", "row_id")]
        df = df.drop(columns=id_cols, errors="ignore")
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # Fill nulls
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Detect label column
        if label_col and label_col in df.columns:
            self.label_col = label_col
        elif "attack_detected" in df.columns:
            self.label_col = "attack_detected"
        elif "list" in df.columns:
            self.label_col = "list"
        else:
            self.label_col = df.columns[-1]

        labels = df[self.label_col].values
        df = df.drop(columns=[self.label_col])

        # One-hot encode categoricals
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, dtype=float)

        # Normalise numerics
        num_cols = df.select_dtypes(include="number").columns
        self.means = df[num_cols].mean()
        self.stds  = df[num_cols].std().replace(0, 1)
        df[num_cols] = (df[num_cols] - self.means) / self.stds

        self.features = torch.tensor(df.values, dtype=torch.float32)
        self.labels   = torch.tensor(labels,    dtype=torch.float32)
        self.feature_dim = self.features.shape[1]

        # Binary or multi-class?
        unique = np.unique(labels)
        self.is_binary = len(unique) <= 2
        self.num_classes = len(unique)

        log.info(f"CSVDataset: {len(self)} rows, {self.feature_dim} features, "
                 f"label='{self.label_col}', classes={self.num_classes}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(0)

    def info(self) -> Dict:
        return {
            "rows":        len(self),
            "feature_dim": self.feature_dim,
            "label_col":   self.label_col,
            "num_classes": self.num_classes,
            "is_binary":   self.is_binary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# NPY / NPZ DATASET
# ─────────────────────────────────────────────────────────────────────────────

class NumpyDataset(Dataset):
    def __init__(self, path: str):
        p = Path(path)
        if p.suffix == ".npz":
            data = np.load(path)
            keys = list(data.keys())
            X = data[keys[0]].astype(np.float32)
            y = data[keys[1]].astype(np.float32) if len(keys) > 1 else np.zeros(len(X))
        else:
            arr = np.load(path).astype(np.float32)
            X, y = arr[:, :-1], arr[:, -1]

        self.features    = torch.from_numpy(X)
        self.labels      = torch.from_numpy(y).unsqueeze(1)
        self.feature_dim = X.shape[1]
        self.num_classes = len(np.unique(y))
        self.is_binary   = self.num_classes <= 2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def info(self) -> Dict:
        return {"rows": len(self), "feature_dim": self.feature_dim,
                "num_classes": self.num_classes}


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FILE LOADER  (merges multiple files into one dataset)
# ─────────────────────────────────────────────────────────────────────────────

class MultiFileDataset(Dataset):
    """Concatenates multiple CSVDataset / NumpyDataset objects."""

    def __init__(self, datasets: list):
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumlen  = np.cumsum([0] + self._lengths)
        self.feature_dim = datasets[0].feature_dim
        self.num_classes = datasets[0].num_classes
        self.is_binary   = datasets[0].is_binary

    def __len__(self):
        return int(self._cumlen[-1])

    def __getitem__(self, idx):
        for i, (lo, hi) in enumerate(
                zip(self._cumlen[:-1], self._cumlen[1:])):
            if lo <= idx < hi:
                local_idx = idx - lo
                x, y = self.datasets[i][local_idx]
                # Pad/trim features to match first dataset's dim
                if x.shape[0] < self.feature_dim:
                    x = torch.cat([x, torch.zeros(self.feature_dim - x.shape[0])])
                elif x.shape[0] > self.feature_dim:
                    x = x[:self.feature_dim]
                return x, y
        raise IndexError(idx)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(file_paths: List[str],
                  batch_size: int = 32,
                  val_split: float = 0.15,
                  num_workers: int = 0
                  ) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Given a list of file paths, build train + val DataLoaders.
    Returns (train_loader, val_loader, info_dict).
    """
    datasets = []
    skipped  = []

    for path in file_paths:
        ext = Path(path).suffix.lower()
        try:
            if ext == ".csv":
                datasets.append(CSVDataset(path))
            elif ext in (".npy", ".npz"):
                datasets.append(NumpyDataset(path))
            else:
                skipped.append(path)
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")
            skipped.append(path)

    if not datasets:
        raise ValueError(
            f"No loadable files found. Skipped: {skipped}\n"
            "Supported: .csv, .npy, .npz"
        )

    if len(datasets) == 1:
        full_ds = datasets[0]
    else:
        full_ds = MultiFileDataset(datasets)

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    info = {
        "total_rows":   len(full_ds),
        "train_rows":   n_train,
        "val_rows":     n_val,
        "feature_dim":  full_ds.feature_dim,
        "num_classes":  full_ds.num_classes,
        "is_binary":    full_ds.is_binary,
        "files_loaded": len(datasets),
        "files_skipped": skipped,
        "batch_size":   batch_size,
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
    }
    return train_loader, val_loader, info


def build_loaders(file_paths, batch_size, val_split, num_workers):
    loaders = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            dataset = CSVDataset(f)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
            loaders.append(loader)
    return loaders
"""
Image dataset for training HMTImageClassifier.

Folder structure expected (standard ImageNet-style):
  root/
    class_a/  img1.jpg  img2.png ...
    class_b/  img3.jpg  ...
    class_c/  ...

Or flat folder (all images, no labels — unsupervised / feature extraction):
  root/  img1.jpg  img2.jpg ...

Also supports a CSV with columns: filepath, label
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import logging

log = logging.getLogger("ImageDataset")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(img_size: int = 64, augment: bool = True):
    """Standard image transforms. Augment=True for training."""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 8, img_size + 8)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """
    Loads images from a folder structure.

    Supports:
    - Subfolder-per-class:  root/cat/img.jpg, root/dog/img.jpg
    - Flat folder:          root/img.jpg  (all label=0)
    - CSV manifest:         filepath,label columns
    - List of paths:        explicit file list passed in

    Auto-resizes all images to img_size × img_size.
    """

    def __init__(self, paths_and_labels: List[Tuple[str, int]],
                 class_names: List[str],
                 transform=None,
                 img_size: int = 64):
        self.samples     = paths_and_labels   # [(path, label), ...]
        self.class_names = class_names
        self.transform   = transform or get_transforms(img_size, augment=True)
        self.img_size    = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return black image on load failure
            img = Image.new("RGB", (self.img_size, self.img_size))
        return self.transform(img), label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def info(self) -> Dict:
        from collections import Counter
        label_counts = Counter(lbl for _, lbl in self.samples)
        return {
            "total_images": len(self.samples),
            "num_classes":  self.num_classes,
            "class_names":  self.class_names,
            "class_counts": {self.class_names[k]: v
                             for k, v in sorted(label_counts.items())},
            "img_size":     self.img_size,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_images(paths: List[str]) -> Tuple[List[Tuple[str, int]], List[str]]:
    """
    Given a list of file/folder paths, discover all images and assign labels.

    Rules:
    - If path is a folder with subfolders → each subfolder = one class
    - If path is a folder with only images → single class (folder name)
    - If path is an image file → single class (parent folder name)
    - If path is a CSV → read filepath,label columns

    Returns (samples, class_names) where samples = [(path, label_idx), ...]
    """
    samples: List[Tuple[str, int]] = []
    class_to_idx: Dict[str, int]   = {}

    def _get_class(name: str) -> int:
        if name not in class_to_idx:
            class_to_idx[name] = len(class_to_idx)
        return class_to_idx[name]

    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            log.warning(f"Path not found: {p}")
            continue

        # CSV manifest
        if p.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(p)
            if "filepath" in df.columns and "label" in df.columns:
                for _, row in df.iterrows():
                    fp  = Path(str(row["filepath"]))
                    lbl = str(row["label"])
                    if fp.exists() and fp.suffix.lower() in IMG_EXTS:
                        samples.append((str(fp), _get_class(lbl)))
            continue

        # Single image file
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            samples.append((str(p), _get_class(p.parent.name)))
            continue

        # Folder
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if subdirs:
                # Subfolder-per-class
                for subdir in sorted(subdirs):
                    cls = subdir.name
                    for img_path in sorted(subdir.rglob("*")):
                        if img_path.suffix.lower() in IMG_EXTS:
                            samples.append((str(img_path), _get_class(cls)))
            else:
                # Flat folder — use folder name as class
                cls = p.name
                for img_path in sorted(p.glob("*")):
                    if img_path.suffix.lower() in IMG_EXTS:
                        samples.append((str(img_path), _get_class(cls)))

    class_names = [k for k, _ in sorted(class_to_idx.items(),
                                         key=lambda x: x[1])]
    log.info(f"Discovered {len(samples)} images, {len(class_names)} classes: "
             f"{class_names}")
    return samples, class_names


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_image_loaders(
    file_paths:  List[str],
    img_size:    int   = 64,
    batch_size:  int   = 32,
    val_split:   float = 0.15,
    num_workers: int   = 0,
) -> Tuple[DataLoader, DataLoader, List[str], Dict]:
    """
    Build train + val DataLoaders for image classification.
    Returns (train_loader, val_loader, class_names, info).
    """
    samples, class_names = discover_images(file_paths)

    if not samples:
        raise ValueError(
            "No images found. Supported: .jpg .jpeg .png .bmp .tiff\n"
            "Folder structure: root/class_name/image.jpg\n"
            "Or flat folder: root/image.jpg")

    if len(class_names) < 2:
        log.warning(f"Only {len(class_names)} class(es) found. "
                    "Add more class folders for multi-class training.")

    # Split into train/val
    n_val   = max(1, int(len(samples) * val_split))
    n_train = len(samples) - n_val

    # Shuffle before split
    import random
    random.shuffle(samples)
    train_samples = samples[:n_train]
    val_samples   = samples[n_train:]

    train_ds = ImageFolderDataset(
        train_samples, class_names,
        transform=get_transforms(img_size, augment=True),
        img_size=img_size)
    val_ds   = ImageFolderDataset(
        val_samples, class_names,
        transform=get_transforms(img_size, augment=False),
        img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    info = {
        "total_images":  len(samples),
        "train_images":  n_train,
        "val_images":    n_val,
        "num_classes":   len(class_names),
        "class_names":   class_names,
        "img_size":      img_size,
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
    }
    return train_loader, val_loader, class_names, info


def get_transforms(img_size, augment):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if augment:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize
        ])
    return transform

def get_transforms(img_size, augment):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # other transforms here

"""
HuggingFace Datasets Support for Language Model Training

Allows training on datasets from the HuggingFace Hub like:
- "ianncity/General-Distillation-Prompts-1M"
- "openwebtext", "wikitext", "ptb", etc.

Supports:
- Text datasets (language modeling)
- Classification datasets (with automatic label detection)
"""

import logging
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset, Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None

from data.text_dataset import CharTokenizer, TextLMDataset

log = logging.getLogger("DatasetLoader")


class HuggingFaceTextDataset(Dataset):
    """Wrapper for HuggingFace datasets for language model training."""
    
    def __init__(self, texts: List[str], tokenizer: CharTokenizer, 
                 seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        corpus = "\n".join(texts)
        tokens = tokenizer.encode(corpus)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        self.indices = list(range(max(0, len(tokens) - seq_len)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        return x, y


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> Tuple[HFDataset, Dict]:
    """
    Load a dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Dataset name (e.g., "ianncity/General-Distillation-Prompts-1M")
        split: Dataset split ("train", "test", "validation", or "all")
        text_column: Name of the text column (auto-detected if not found)
        cache_dir: Directory to cache the dataset
        trust_remote_code: Whether to trust remote code (for custom datasets)
        
    Returns:
        Tuple of (HFDataset, info_dict)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        )
    
    log.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
    except ValueError as e:
        if "split" in str(e):
            dataset = load_dataset(
                dataset_name,
                split="all",
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        else:
            raise
    
    info = {
        "dataset_name": dataset_name,
        "num_rows": len(dataset),
        "features": list(dataset.features.keys()) if hasattr(dataset, 'features') else [],
    }
    
    log.info(f"Loaded {info['num_rows']} rows from {dataset_name}")
    
    return dataset, info


def build_hf_loaders(
    dataset_name: str,
    seq_len: int = 128,
    batch_size: int = 32,
    val_split: float = 0.1,
    text_column: str = "text",
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, CharTokenizer, Dict]:
    """
    Build train and validation loaders from a HuggingFace dataset.
    
    Args:
        dataset_name: Dataset name (e.g., "ianncity/General-Distillation-Prompts-1M")
        seq_len: Sequence length for language modeling
        batch_size: Batch size
        val_split: Fraction of data for validation
        text_column: Name of text column (auto-detected if not provided)
        cache_dir: Cache directory
        
    Returns:
        (train_loader, val_loader, tokenizer, info_dict)
    """
    dataset, info = load_hf_dataset(dataset_name, "all", text_column, cache_dir, **kwargs)
    
    if text_column not in dataset.column_names:
        text_cols = ["text", "content", "input", "output", "sentence"]
        for col in text_cols:
            if col in dataset.column_names:
                text_column = col
                break
        else:
            text_column = dataset.column_names[0]
            log.warning(f"Text column '{text_column}' not found, using first column")
    
    texts = dataset[text_column]
    
    if isinstance(texts, list):
        text_list = [str(t) for t in texts if t is not None]
    else:
        text_list = [str(t) for t in texts.tolist() if t is not None]
    
    log.info(f"Extracted {len(text_list)} text samples")
    
    tokenizer = CharTokenizer()
    tokenizer.build(text_list)
    
    n_val = max(1, int(len(text_list) * val_split))
    n_train = len(text_list) - n_val
    
    train_texts = text_list[:n_train]
    val_texts = text_list[n_train:]
    
    train_ds = HuggingFaceTextDataset(train_texts, tokenizer, seq_len)
    val_ds = HuggingFaceTextDataset(val_texts, tokenizer, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    result_info = {
        "corpus_chars": sum(len(t) for t in text_list),
        "total_tokens": len(tokenizer.char2idx),
        "vocab_size": tokenizer.vocab_size,
        "seq_len": seq_len,
        "total_windows": len(train_ds) + len(val_ds),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "files_loaded": 1,
        "dataset_name": dataset_name,
        "text_column": text_column,
    }
    
    log.info(f"Built loaders: vocab={tokenizer.vocab_size}, "
             f"train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    return train_loader, val_loader, tokenizer, result_info


def load_classification_dataset(
    dataset_name: str,
    text_column: str = "text",
    label_column: str = "label",
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[HFDataset, Dict]:
    """
    Load a classification dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Dataset name
        text_column: Name of text column
        label_column: Name of label column
        cache_dir: Cache directory
        
    Returns:
        Tuple of (HFDataset, info_dict)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not installed")
    
    dataset, info = load_hf_dataset(dataset_name, "all", text_column, cache_dir, **kwargs)
    
    if label_column not in dataset.column_names:
        label_candidates = ["label", "labels", "class", "target"]
        for col in label_candidates:
            if col in dataset.column_names:
                label_column = col
                break
    
    if label_column in dataset.column_names:
        labels = dataset[label_column]
        if hasattr(labels, 'features'):
            info['num_classes'] = len(labels.features[label_column].names)
            info['class_names'] = labels.features[label_column].names
        else:
            unique_labels = set(labels)
            info['num_classes'] = len(unique_labels)
    
    info['text_column'] = text_column
    info['label_column'] = label_column
    
    return dataset, info


def build_classification_loaders(
    dataset_name: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    text_column: str = "text",
    label_column: str = "label",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Build train and validation loaders from a HuggingFace classification dataset.
    
    Returns:
        (train_loader, val_loader, data_info)
    """
    dataset, info = load_classification_dataset(
        dataset_name, text_column, label_column, cache_dir, **kwargs
    )
    
    from torch.utils.data import random_split
    
    all_data = []
    for i in range(len(dataset)):
        row = dataset[i]
        text = str(row.get(text_column, ""))
        label = int(row.get(label_column, 0))
        all_data.append((text, label))
    
    import random
    random.shuffle(all_data)
    
    n_val = int(len(all_data) * val_split)
    train_data = all_data[n_val:]
    val_data = all_data[:n_val]
    
    class TextClassificationDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            text, label = self.data[idx]
            tokens = tokenizer.encode(text[:max_length])
            tokens = tokens + [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens[:self.max_length], dtype=torch.long), label
    
    tokenizer = CharTokenizer()
    tokenizer.build([text for text, _ in all_data])
    
    train_ds = TextClassificationDataset(train_data, tokenizer, max_length)
    val_ds = TextClassificationDataset(val_data, tokenizer, max_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    result_info = {
        "total_rows": len(all_data),
        "train_rows": len(train_data),
        "val_rows": len(val_data),
        "feature_dim": max_length,
        "num_classes": info.get('num_classes', 2),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "dataset_name": dataset_name,
        "text_column": text_column,
        "label_column": label_column,
    }
    
    return train_loader, val_loader, result_info


SUPPORTED_DATASETS = {
    "text_generation": [
        "ianncity/General-Distillation-Prompts-1M",
        "openwebtext",
        "wikitext",
        "ptb",
        "c4",
        "EleutherAI/gpt-neo-125M",
    ],
    "classification": [
        "sst2",
        "imdb",
        "yelp_polarity",
        "ag_news",
        "dbpedia_14",
        "multi_nli",
    ]
}


def list_supported_datasets(task: str = None) -> Dict[str, List[str]]:
    """List supported datasets by task type."""
    if task:
        return SUPPORTED_DATASETS.get(task, [])
    return SUPPORTED_DATASETS

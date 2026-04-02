"""
Text dataset for language model training.
Supports .txt, .jsonl, .json, .csv (text column) files.
Produces overlapping context windows: input[0:seq_len] → target[1:seq_len+1]
This is the standard next-token prediction setup for language models.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import re
import logging

log = logging.getLogger("TextDataset")


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER  (character-level — no external deps, works on any text)
# ─────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Character-level tokenizer.
    Vocab = every unique character in the training corpus.
    Simple, universal — works on any language/domain.
    """

    PAD   = 0
    UNK   = 1
    BOS   = 2   # beginning of sequence
    EOS   = 3   # end of sequence

    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.vocab_size = 0

    def build(self, texts: List[str]) -> None:
        """Build vocab from a list of text strings."""
        chars = set()
        for t in texts:
            chars.update(t)
        vocab = self.SPECIAL + sorted(chars)
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(vocab)
        log.info(f"CharTokenizer: vocab_size={self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(c, self.UNK) for c in text]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars = []
        for i in ids:
            c = self.idx2char.get(i, "")
            if skip_special and i < len(self.SPECIAL):
                continue
            chars.append(c)
        return "".join(chars)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx,
                       "idx2char": {str(k): v
                                    for k, v in self.idx2char.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        t = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        t.char2idx = d["char2idx"]
        t.idx2char = {int(k): v for k, v in d["idx2char"].items()}
        t.vocab_size = len(t.char2idx)
        return t


# ─────────────────────────────────────────────────────────────────────────────
# TEXT FILE READER
# ─────────────────────────────────────────────────────────────────────────────

def read_text_files(paths: List[str]) -> str:
    """
    Read all supported text files and return one big string.
    Supports: .txt, .jsonl (text/content field), .json, .csv (text column)
    """
    corpus = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        ext = p.suffix.lower()
        try:
            if ext == ".txt":
                corpus.append(p.read_text(encoding="utf-8", errors="ignore"))

            elif ext == ".jsonl":
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Try common text field names
                        for key in ("text", "content", "body", "message",
                                    "sentence", "input", "output"):
                            if key in obj:
                                corpus.append(str(obj[key]))
                                break
                        else:
                            # Fallback: join all string values
                            corpus.append(" ".join(
                                str(v) for v in obj.values()
                                if isinstance(v, str)))
                    except json.JSONDecodeError:
                        corpus.append(line)

            elif ext == ".json":
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            corpus.append(item)
                        elif isinstance(item, dict):
                            for key in ("text", "content", "body"):
                                if key in item:
                                    corpus.append(str(item[key]))
                                    break
                elif isinstance(data, str):
                    corpus.append(data)

            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(path)
                # Find text columns
                text_cols = [c for c in df.columns
                             if c.lower() in ("text", "content", "body",
                                              "message", "sentence", "review",
                                              "description", "comment")]
                if not text_cols:
                    # Use all string columns
                    text_cols = df.select_dtypes(
                        include="object").columns.tolist()
                for col in text_cols:
                    corpus.extend(
                        df[col].dropna().astype(str).tolist())

        except Exception as e:
            log.warning(f"Could not read {path}: {e}")

    full = "\n".join(corpus)
    log.info(f"Corpus: {len(full):,} characters from {len(paths)} file(s)")
    return full


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE MODEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TextLMDataset(Dataset):
    """
    Next-token prediction dataset.
    Splits corpus into overlapping windows of length seq_len.
    input  = tokens[i   : i+seq_len]
    target = tokens[i+1 : i+seq_len+1]
    """

    def __init__(self, tokens: List[int], seq_len: int = 128, 
                 reasoning_only: bool = False, tokenizer: Optional['CharTokenizer'] = None):
        self.tokens  = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        if reasoning_only and tokenizer:
            # Filter windows to only include those containing reasoning keywords
            # Keywords: "Thought:", "Reason:", "because", "therefore", "Step"
            valid_indices = []
            for i in range(0, len(tokens) - seq_len):
                # Check a window for reasoning markers
                # We decode a small portion or check for specific char patterns
                window_text = tokenizer.decode(tokens[i : i + seq_len]).lower()
                if any(k in window_text for k in ["thought:", "reason:", "because", "step", "since", "therefore"]):
                    valid_indices.append(i)
            self.indices = valid_indices
        else:
            self.indices = list(range(max(0, len(tokens) - seq_len)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        x = self.tokens[start     : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y

    @property
    def vocab_size(self) -> int:
        return int(self.tokens.max().item()) + 1


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_text_loaders(
    file_paths: List[str],
    seq_len:    int   = 128,
    batch_size: int   = 32,
    val_split:  float = 0.1,
    tokenizer:  Optional[CharTokenizer] = None,
    reasoning_only: bool = False,
) -> Tuple[DataLoader, DataLoader, CharTokenizer, Dict]:
    """
    Build train + val DataLoaders for language model training.
    Returns (train_loader, val_loader, tokenizer, info).
    """
    corpus = read_text_files(file_paths)
    if len(corpus) < seq_len + 2:
        raise ValueError(
            f"Corpus too short ({len(corpus)} chars). "
            f"Need at least {seq_len + 2} characters.")

    if tokenizer is None:
        tokenizer = CharTokenizer()
        tokenizer.build([corpus])

    tokens = tokenizer.encode(corpus)
    log.info(f"Tokenized: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")

    full_ds = TextLMDataset(tokens, seq_len=seq_len, 
                            reasoning_only=reasoning_only, tokenizer=tokenizer)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False)

    info = {
        "corpus_chars":   len(corpus),
        "total_tokens":   len(tokens),
        "vocab_size":     tokenizer.vocab_size,
        "seq_len":        seq_len,
        "total_windows":  len(full_ds),
        "train_windows":  n_train,
        "val_windows":    n_val,
        "train_batches":  len(train_loader),
        "val_batches":    len(val_loader),
        "files_loaded":   len(file_paths),
    }
    return train_loader, val_loader, tokenizer, info

"""
Text dataset for language model training.
Supports .txt, .jsonl, .json, .csv files → overlapping causal LM windows
or structured reasoning templates with loss masking.
Uses AdvancedTokenizer (BPE) with CharTokenizer fallback.

Torch is imported lazily inside functions so that the module can be imported
without torch for tokenizer-only usage.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import logging

log = logging.getLogger("TextDataset")

def _torch():
    import torch
    return torch

def _torch_utils():
    from torch.utils.data import Dataset, DataLoader, random_split
    return Dataset, DataLoader, random_split


class CharTokenizer:
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.vocab_size = 0

    def build(self, texts: List[str]) -> None:
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
                       "idx2char": {str(k): v for k, v in self.idx2char.items()}}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        t = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        t.char2idx = d["char2idx"]
        t.idx2char = {int(k): v for k, v in d["idx2char"].items()}
        t.vocab_size = len(t.char2idx)
        return t


def read_text_files(paths: List[str]) -> str:
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
                        for key in ("text", "content", "body", "message", "sentence", "input", "output"):
                            if key in obj:
                                corpus.append(str(obj[key]))
                                break
                        else:
                            corpus.append(" ".join(str(v) for v in obj.values() if isinstance(v, str)))
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
                text_cols = [c for c in df.columns
                             if c.lower() in ("text", "content", "body", "message", "sentence", "review", "description", "comment")]
                if not text_cols:
                    text_cols = df.select_dtypes(include="object").columns.tolist()
                for col in text_cols:
                    corpus.extend(df[col].dropna().astype(str).tolist())
            elif p.suffix.lower() in ('.py', '.c', '.h', '.js', '.ts', '.rs', '.go', '.java',
                                       '.cpp', '.hpp', '.rb', '.php', '.swift', '.sh', '.lua',
                                       '.cc', '.hh', '.toml', '.yaml', '.yml', '.xml', '.md', '.rst'):
                corpus.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            log.warning(f"Could not read {path}: {e}")

    full = "\n".join(corpus)
    log.info(f"Corpus: {len(full):,} characters from {len(paths)} file(s)")
    return full


class TextLMDataset:
    def __init__(self, tokens, seq_len: int = 128, prompt_ratio: float = 0.0):
        torch = _torch()
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.prompt_ratio = prompt_ratio
        self.indices = list(range(max(0, len(tokens) - seq_len)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        torch = _torch()
        start = self.indices[idx]
        x = self.tokens[start: start + self.seq_len]
        y = self.tokens[start + 1: start + self.seq_len + 1].clone()
        if self.prompt_ratio > 0.0:
            mask_len = int(y.size(0) * self.prompt_ratio)
            y[:mask_len] = -100
        return x, y


class ReasoningTextLMDataset:
    """Dataset that uses template pipeline to create loss-masked reasoning windows."""
    def __init__(self, windows: List[Tuple[List[int], List[int]]]):
        torch = _torch()
        self.windows = [
            (torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))
            for x, y in windows
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def build_text_loaders(
    file_paths: List[str],
    seq_len: int = 128,
    batch_size: int = 32,
    val_split: float = 0.1,
    tokenizer=None,
    prompt_ratio: float = 0.3,
    template_name: str = "step_by_step",
    template_fields: Optional[Dict[str, str]] = None,
):
    Dataset, DataLoader, random_split = _torch_utils()
    from data.advanced_tokenizer import AdvancedTokenizer
    from data.templates import get_template, init_templates
    from data.template_pipeline import parse_examples, render_examples_to_windows

    use_template = template_name != "raw"
    template_dir = "templates"

    if use_template:
        init_templates(template_dir)
        template = get_template(template_name)
        if template is None:
            log.warning(f"Template '{template_name}' not found, falling back to 'raw'")
            use_template = False
        elif template_fields:
            template.field_map.update(template_fields)

    if not use_template:
        # ── Legacy raw mode with prompt_ratio masking ──
        corpus = read_text_files(file_paths)
        if len(corpus) < seq_len + 2:
            raise ValueError(f"Corpus too short ({len(corpus)} chars). Need at least {seq_len + 2}.")

        if tokenizer is None:
            tokenizer = AdvancedTokenizer(vocab_size=8192)
            tokenizer.build([corpus])

        tokens = tokenizer.encode(corpus)
        log.info(f"Raw mode: {len(tokens):,} tokens, vocab={tokenizer.vocab_size_real}")

        full_ds = TextLMDataset(tokens, seq_len=seq_len, prompt_ratio=prompt_ratio)
        n_val = max(1, int(len(full_ds) * val_split))
        n_train = len(full_ds) - n_val

        torch = _torch()
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        info = {
            "template": "raw",
            "corpus_chars": len(corpus),
            "total_tokens": len(tokens),
            "vocab_size": tokenizer.vocab_size_real,
            "seq_len": seq_len,
            "total_windows": len(full_ds),
            "train_windows": n_train,
            "val_windows": n_val,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "files_loaded": len(file_paths),
        }
        return train_loader, val_loader, tokenizer, info

    # ── Template-based reasoning mode ──
    log.info(f"Template mode: '{template_name}' ({template.description})")

    # Parse all files through template pipeline
    examples = parse_examples(file_paths, template)

    if not examples:
        raise ValueError(
            f"No examples could be parsed from {len(file_paths)} file(s) "
            f"with template '{template_name}'. Check file format or try template='raw'."
        )

    # Build tokenizer from all example text
    if tokenizer is None:
        all_text = "\n".join(
            " ".join(ex.segments.values())
            for ex in examples
        )
        tokenizer = AdvancedTokenizer(vocab_size=8192)
        tokenizer.build([all_text])

    # Render examples to windows
    windows = render_examples_to_windows(examples, template, tokenizer, seq_len=seq_len)

    if len(windows) < 2:
        raise ValueError(f"Too few training windows ({len(windows)}). Need more data or shorter seq_len.")

    full_ds = ReasoningTextLMDataset(windows)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    torch = _torch()
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    masked = sum(1 for _, y in windows for l in y if l < 0)
    total = sum(len(y) for _, y in windows)

    info = {
        "template": template_name,
        "examples_parsed": len(examples),
        "total_windows": len(windows),
        "vocab_size": tokenizer.vocab_size_real,
        "seq_len": seq_len,
        "loss_masked_pct": round(masked / max(total, 1) * 100, 1),
        "train_windows": n_train,
        "val_windows": n_val,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "files_loaded": len(file_paths),
    }
    log.info(f"Template '{template_name}': {len(examples)} examples, {len(windows)} windows, "
             f"{info['loss_masked_pct']}% tokens masked")
    return train_loader, val_loader, tokenizer, info

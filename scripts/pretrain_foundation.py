"""
Foundation Model Trainer — Full pipeline: download → train → save.

Downloads real books (public domain), trains a decoder-only Transformer
with prompt-ratio masking, shows live progress with ETA, saves periodic
checkpoints, and supports resume.

Usage:
    python scripts/pretrain_foundation.py                          # defaults
    python scripts/pretrain_foundation.py --data-chars 500000      # more data
    python scripts/pretrain_foundation.py --epochs 20 --dim 256    # bigger model
    python scripts/pretrain_foundation.py --resume                 # resume from checkpoint
"""

import argparse, json, sys, os, logging, re, time, math, urllib.request, shutil
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DML_TRAINING"] = "1"

import torch
import torch.nn.functional as F

from core.implementations import HMTLanguageModel
from core.text_model import save_lm, load_lm
from data.advanced_tokenizer import train_tokenizer, AdvancedTokenizer
try:
    from torch.amp import autocast, GradScaler
    _new_amp = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _new_amp = False

logging.disable(logging.CRITICAL)

FOUNDATION_DIR = Path(__file__).resolve().parent.parent / "foundation"

GUTENBERG_BOOKS = [
    # ── Literature (general language modeling) ──
    ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "pride_and_prejudice.txt"),
    ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "frankenstein.txt"),
    ("https://www.gutenberg.org/cache/epub/11/pg11.txt", "alice_in_wonderland.txt"),
    ("https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "sherlock_holmes.txt"),
    ("https://www.gutenberg.org/cache/epub/74/pg74.txt", "tom_sawyer.txt"),
    ("https://www.gutenberg.org/cache/epub/1260/pg1260.txt", "jane_eyre.txt"),
    ("https://www.gutenberg.org/cache/epub/1400/pg1400.txt", "great_expectations.txt"),
    ("https://www.gutenberg.org/cache/epub/768/pg768.txt", "wuthering_heights.txt"),
    ("https://www.gutenberg.org/cache/epub/43/pg43.txt", "dracula.txt"),
    ("https://www.gutenberg.org/cache/epub/730/pg730.txt", "oliver_twist.txt"),
    # ── Mathematics ──
    ("https://www.gutenberg.org/cache/epub/97/pg97.txt", "flatland.txt"),                                 # Flatland — math fiction
    ("https://www.gutenberg.org/cache/epub/29810/pg29810.txt", "the_number_concept.txt"),
    ("https://www.gutenberg.org/cache/epub/29764/pg29764.txt", "a_history_of_mathematics.txt"),
    # ── Science & Reason ──
    ("https://www.gutenberg.org/cache/epub/22764/pg22764.txt", "origin_of_species.txt"),                  # Darwin — fixed ID
    ("https://www.gutenberg.org/cache/epub/1228/pg1228.txt", "relativity_the_special_and_general_theory.txt"),
    ("https://www.gutenberg.org/cache/epub/5797/pg5797.txt", "scientific_papers.txt"),                   # Maxwell — replaces archimedes
    ("https://www.gutenberg.org/cache/epub/38480/pg38480.txt", "the_universe.txt"),                      # Popular science — replaces intro_math
    ("https://www.gutenberg.org/cache/epub/18757/pg18757.txt", "the_science_of_mathematics.txt"),
    ("https://www.gutenberg.org/cache/epub/33636/pg33636.txt", "philosophical_essays.txt"),
    ("https://www.gutenberg.org/cache/epub/15784/pg15784.txt", "first_principles.txt"),
    ("https://www.gutenberg.org/cache/epub/58213/pg58213.txt", "computing_machinery.txt"),               # Computing machinery
    ("https://www.gutenberg.org/cache/epub/60074/pg60074.txt", "programming_principles.txt"),           # Programming fundamentals
    ("https://www.gutenberg.org/cache/epub/63444/pg63444.txt", "algorithms_logic.txt"),                 # Algorithms & logic
]


@dataclass
class ProgressStats:
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val: float = float("inf")
    epoch_time: float = 0.0
    total_time: float = 0.0
    tokens_seen: int = 0
    tokens_per_sec: float = 0.0


# ── Data ─────────────────────────────────────────────────────────────────────

def download_all(target_dir: Path) -> List[str]:
    """Download and clean all Gutenberg books. Returns paths to cleaned files."""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    total = len(GUTENBERG_BOOKS)
    for i, (url, name) in enumerate(GUTENBERG_BOOKS):
        dest = target_dir / name
        if dest.exists():
            print(f"  [{i+1}/{total}] {name} — cached")
        else:
            print(f"  [{i+1}/{total}] {name} — downloading...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, dest)
                print("done")
            except Exception as e:
                print(f"FAILED ({e})")
                continue
        clean = _clean_book(dest)
        if clean:
            paths.append(str(clean))
            size = clean.stat().st_size
            print(f"         {size//1024} KB")
    if not paths:
        raise RuntimeError("No books could be downloaded.")
    print(f"\n  Total: {len(paths)} books, ~{sum(Path(p).stat().st_size for p in paths)//1024} KB")
    return paths


def _clean_book(src: Path) -> Optional[Path]:
    """Strip Gutenberg headers/footers from a book."""
    text = src.read_text(encoding="utf-8", errors="ignore")
    for pat in [
        r"\*\*\*\s*START OF (THE|THIS)\s+PROJECT\s+GUTENBERG",
        r"\*\*\*\s*END OF (THE|THIS)\s+PROJECT\s+GUTENBERG",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            text = text[m.end():] if "START" in pat else text[:m.start()]
    cleaned = "\n".join(
        l for l in text.split("\n")
        if not l.strip().startswith("***") and "gutenberg" not in l.lower()
    ).strip()
    if len(cleaned) < 1000:
        return None
    clean_path = src.parent / f"{src.stem}.clean.txt"
    clean_path.write_text(cleaned, encoding="utf-8")
    return clean_path


def load_corpus(file_paths: List[str], max_chars: int) -> str:
    """Load and concatenate text files up to max_chars."""
    parts = []
    total = 0
    for p in file_paths:
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        if total + len(text) > max_chars:
            allowed = max_chars - total
            if allowed > 0:
                parts.append(text[:allowed])
            break
        parts.append(text)
        total += len(text)
    return "\n".join(parts)


# ── Training ─────────────────────────────────────────────────────────────────

def make_windows(tokens: List[int], seq_len: int, prompt_ratio: float = 0.0):
    stride = seq_len // 2
    windows, targets = [], []
    for i in range(0, len(tokens) - seq_len - 1, stride):
        x = tokens[i:i + seq_len]
        y = tokens[i + 1:i + seq_len + 1]
        for j in range(int(seq_len * prompt_ratio)):
            y[j] = -100
        windows.append(x)
        targets.append(y)
    return windows, targets


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def print_progress(stats: ProgressStats):
    """Print a compact progress line with all stats."""
    epoch = f"Epoch {stats.epoch:2d}/{stats.total_epochs}"
    train = f"train={stats.train_loss:.4f}"
    val = f"val={stats.val_loss:.4f}"
    best = f"best={stats.best_val:.4f}" if stats.best_val < float("inf") else ""
    speed = f"{stats.tokens_per_sec:.0f} tok/s" if stats.tokens_per_sec else ""
    elapsed = format_time(stats.total_time)
    remaining = ""
    if stats.epoch_time and stats.epoch > 0 and stats.total_epochs:
        eta = stats.epoch_time * (stats.total_epochs - stats.epoch)
        remaining = f"ETA {format_time(eta)}"
    parts = [e for e in [epoch, train, val, best, speed, elapsed, remaining] if e]
    print(f"  {' | '.join(parts)}")


def train_foundation(
    data_paths: List[str],
    max_chars: int = 300000,
    epochs: int = 15,
    dim: int = 256,
    layers: int = 4,
    heads: int = 4,
    seq_len: int = 128,
    batch_size: int = 32,
    vocab_size: int = 4096,
    prompt_ratio: float = 0.0,
    resume: bool = False,
    save_every: int = 5,
    lr: float = 0.001,
    device: str = "cpu",
    data_dir: str = "",
    use_amp: bool = True,
):
    # ── Load or resume ────────────────────────────────────────────────────
    model_path = FOUNDATION_DIR / "base_model.pt"
    checkpoint_path = FOUNDATION_DIR / "checkpoint.pt"
    tok_path = FOUNDATION_DIR / "base_tokenizer.json"

    start_epoch = 1
    model = None
    tokenizer = None
    opt = None
    sched = None

    if resume and checkpoint_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model = HMTLanguageModel(
            vocab_size=ckpt.get("vocab_size", vocab_size),
            dim=ckpt.get("dim", dim),
            num_layers=ckpt.get("layers", layers),
            num_heads=ckpt.get("heads", heads),
            n_kv_heads=max(1, ckpt.get("heads", heads) // 4),
            max_seq=seq_len * 4,
            dropout=0.05,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        opt = torch.optim.AdamW(model.parameters(), lr=ckpt.get("lr", 0.001))
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        tokenizer = None  # will rebuild from corpus
        trained_chars = ckpt.get("trained_chars", 0)
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")
    else:
        best_val = float("inf")
        trained_chars = 0

    device = "cuda" if torch.cuda.is_available() else device
    print(f"  Device: {device.upper()}")
    if device.startswith("cuda"):
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GiB")

    # ── Load data (always — needed for both fresh and resume) ─────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FOUNDATION MODEL TRAINER")
    print(f"{sep}")
    print(f"\n  Model: dim={dim} layers={layers} heads={heads}")
    print(f"  Data:  {max_chars:,} chars max from {len(data_paths)} book(s)")
    print(f"  Train: {epochs} epochs, seq_len={seq_len}, batch={batch_size}")
    if resume:
        mode = f"  Mode:  resume from epoch {start_epoch}" if start_epoch > 1 else "  Mode:  fresh start (checkpoint not found)"
        print(mode)
    if data_dir:
        p = Path(data_dir)
        code_exts = {'.py', '.c', '.h', '.js', '.ts', '.rs', '.go', '.java',
                     '.cpp', '.hpp', '.rb', '.php', '.swift', '.sh', '.lua',
                     '.cc', '.hh', '.txt', '.csv', '.json', '.jsonl',
                     '.md', '.rst', '.yaml', '.yml', '.xml', '.toml'}
        data_paths = sorted([
            str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in code_exts
        ])
        print(f"  Data: {len(data_paths)} file(s) from {data_dir}")

    # ── Streaming Dataset (no giant window lists) ─────────────────────────
    class StreamingLMDataset(torch.utils.data.Dataset):
        """Creates windows on-the-fly from a token tensor — O(1) memory per window."""
        __slots__ = ("tokens", "seq_len", "stride")
        def __init__(self, tokens_tensor, seq_len, stride):
            self.tokens = tokens_tensor
            self.seq_len = seq_len
            self.stride = stride
        def __len__(self):
            return (len(self.tokens) - self.seq_len - 1) // self.stride + 1
        def __getitem__(self, idx):
            start = idx * self.stride
            x = self.tokens[start:start + self.seq_len]
            y = self.tokens[start + 1:start + self.seq_len + 1]
            return x, y

    # ── Pre-processing cache ──────────────────────────────────────────────
    cache_dir = FOUNDATION_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_path = cache_dir / "tokenizer.json"
    tokens_path = cache_dir / "tokens.pt"
    meta_path = cache_dir / "meta.json"
    corpus_path = cache_dir / "corpus.txt"
    has_cache = tokens_path.exists() and meta_path.exists()
    resume_pre = resume and has_cache

    if resume_pre:
        print(f"  Loading cached tokenizer + token tensor...")
        tokenizer = AdvancedTokenizer.load(str(tok_path))
        actual_vocab = tokenizer.vocab_size_real
        tokens_tensor = torch.load(str(tokens_path), map_location="cpu", weights_only=True)
        meta = json.loads(meta_path.read_text())
        seq_len = meta.get("seq_len", seq_len)
        batch_size = meta.get("batch_size", batch_size)
        n_windows = meta["n_windows"]
        print(f"    {len(tokens_tensor):,} tokens, {n_windows:,} windows")
    else:
        # Try loading cached corpus first (saves reading 2577 Drive files)
        if corpus_path.exists():
            print(f"  Loading cached corpus...")
            corpus = corpus_path.read_text(encoding="utf-8")
            print(f"    {len(corpus):,} characters loaded (cached)")
        else:
            print(f"\n  Loading corpus from {len(data_paths)} files...")
            corpus = load_corpus(data_paths, max_chars)
            corpus_path.write_text(corpus, encoding="utf-8")
            print(f"    {len(corpus):,} characters loaded")
        print(f"  Building tokenizer (vocab={vocab_size})...")
        tok_sample = corpus[:min(len(corpus), 2_000_000)]
        tokenizer = train_tokenizer([tok_sample], vocab_size=vocab_size)
        tokenizer.save(str(tok_path))
        actual_vocab = tokenizer.vocab_size_real
        print(f"  Encoding corpus ({len(corpus):,} chars in 5MB chunks)...", flush=True)
        tokens_list = []
        chunk_size = 5_000_000
        for i in range(0, len(corpus), chunk_size):
            chunk = corpus[i:i+chunk_size]
            tokens_list.extend(tokenizer.encode(chunk))
            print(f"    chunk {i//chunk_size+1}/{(len(corpus)-1)//chunk_size+1}: {len(tokens_list):,} tokens so far", flush=True)
        print(f"  Total: {len(tokens_list):,} tokens, vocab={actual_vocab}")
        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
        del tokens_list, corpus
        torch.save(tokens_tensor, str(tokens_path))
        n_windows = (len(tokens_tensor) - seq_len - 1) // (seq_len // 2) + 1
        meta_path.write_text(json.dumps({
            "n_windows": n_windows, "seq_len": seq_len,
            "batch_size": batch_size, "vocab_size": actual_vocab,
        }))
        print(f"    {n_windows:,} training windows")

    stride = seq_len // 2
    full_dataset = StreamingLMDataset(tokens_tensor, seq_len, stride)
    total_windows = len(full_dataset)
    n_val = max(1, int(total_windows * 0.15))
    n_train = total_windows - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2)
    total_batches = len(train_loader)
    print(f"    {n_train} train / {n_val} val ({total_batches} batches/epoch)\n")
    del tokens_tensor, full_dataset

    # ── Build model ───────────────────────────────────────────────────────
    if model is None:
        actual_vocab = tokenizer.vocab_size_real if tokenizer else vocab_size
        model = HMTLanguageModel(
            vocab_size=max(actual_vocab, 256), dim=dim, num_layers=layers,
            num_heads=heads, n_kv_heads=max(1, heads // 4),
            max_seq=seq_len * 4, dropout=0.05,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── AMP setup ─────────────────────────────────────────────────────────
    use_amp = use_amp and device.startswith("cuda")
    if _new_amp:
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("  Using AMP (mixed precision) — ~2x speedup on T4")

    # ── Training loop ─────────────────────────────────────────────────────
    total_t0 = time.time()
    epoch_t0 = time.time()
    stats = ProgressStats(total_epochs=epochs, best_val=best_val)
    total_tokens = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            ac = autocast("cuda", enabled=use_amp) if _new_amp else autocast(enabled=use_amp)
            with ac:
                logits = model(bx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
            total_tokens += bx.numel()

        sched.step()

        # Validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                ac = autocast("cuda", enabled=use_amp) if _new_amp else autocast(enabled=use_amp)
                with ac:
                    logits = model(bx)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1), ignore_index=-100)
                vloss += loss.item()

        avg_train = total_loss / max(n_batches, 1)
        avg_val = vloss / max(len(val_loader), 1)
        epoch_time = time.time() - epoch_t0
        total_time = time.time() - total_t0

        if avg_val < best_val:
            best_val = avg_val

        stats.epoch = epoch
        stats.train_loss = avg_train
        stats.val_loss = avg_val
        stats.best_val = best_val
        stats.epoch_time = epoch_time
        stats.total_time = total_time
        stats.tokens_per_sec = total_tokens / max(total_time, 0.01) if total_time > 0 else 0.0
        print_progress(stats)

        epoch_t0 = time.time()

        # Save checkpoint every epoch (so --resume never loses >1 epoch on Colab)
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "dim": dim,
            "layers": layers,
            "heads": heads,
            "vocab_size": actual_vocab,
            "lr": 0.001,
            "total_time": total_time,
        }
        torch.save(ckpt_data, str(checkpoint_path))
        # Full model save every save_every epochs
        if epoch % save_every == 0 or epoch == epochs:
            FOUNDATION_DIR.mkdir(parents=True, exist_ok=True)
            cfg = dict(vocab_size=actual_vocab, hidden_dim=dim, num_layers=layers,
                       num_heads=heads, n_kv_heads=max(1, heads // 4),
                       max_seq=seq_len * 4, dropout=0.05, seq_len=seq_len,
                       epochs=epochs, epoch=epoch, prompt_ratio=prompt_ratio)
            save_lm(model, tokenizer, config=cfg, epoch=epoch, loss=avg_val,
                    path=str(model_path))
            tokenizer.save(str(tok_path))

        # early stopping removed — run all epochs

    # ── Final save ────────────────────────────────────────────────────────
    FOUNDATION_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(vocab_size=actual_vocab, hidden_dim=dim, num_layers=layers,
               num_heads=heads, n_kv_heads=max(1, heads // 4),
               max_seq=seq_len * 4, dropout=0.05, seq_len=seq_len,
               epochs=epochs, epoch=epoch, prompt_ratio=prompt_ratio)
    save_lm(model, tokenizer, config=cfg, epoch=epoch, loss=avg_val,
            path=str(model_path))
    tokenizer.save(str(tok_path))

    # Remove checkpoint + cache (training complete — no resume needed)
    for p in [checkpoint_path, tokens_path, meta_path, corpus_path, tok_path.parent]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(str(p))
            else:
                p.unlink()

    total_time = time.time() - total_t0
    manifest = dict(
        name="Foundation Model", type="foundation",
        dim=dim, layers=layers, heads=heads,
        seq_len=seq_len, vocab_size=actual_vocab,
        epochs=epoch, loss=round(float(avg_val), 4),
        best_val=round(float(best_val), 4),
        total_time_seconds=round(total_time, 1),
        tokens_per_sec=round(stats.tokens_per_sec, 1),
        prompt_ratio=prompt_ratio,
        weights_file=str(model_path),
        tokenizer_file=str(tok_path),
        corpus_chars=max_chars,
        files_trained_on=data_paths,
    )
    (FOUNDATION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE — {format_time(total_time)} total")
    print(f"  Saved: {model_path}")
    print(f"  Tok/s: {stats.tokens_per_sec:.0f}")
    print(f"  Final val_loss: {avg_val:.4f}  (best: {best_val:.4f})")
    print(f"{'=' * 60}\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train foundation model on real text")
    parser.add_argument("--data-dir", default="", help="Directory of .txt files (default: download Gutenberg)")
    parser.add_argument("--data-chars", type=int, default=300000, help="Max corpus characters (default: 300k)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--dim", type=int, default=384, help="Model dimension (default: 384)")
    parser.add_argument("--layers", type=int, default=6, help="Transformer layers (default: 6)")
    parser.add_argument("--heads", type=int, default=6, help="Attention heads (default: 6)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Vocab size (default: 4096)")
    parser.add_argument("--prompt-ratio", type=float, default=0.0, help="Fraction of sequence to mask (default: 0.0)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda, default: auto-detect cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    if args.data_dir:
        p = Path(args.data_dir)
        if not p.exists() or not p.is_dir():
            print(f"Error: directory not found: {args.data_dir}")
            sys.exit(1)
        code_exts = {'.py', '.c', '.h', '.js', '.ts', '.rs', '.go', '.java',
                     '.cpp', '.hpp', '.rb', '.php', '.swift', '.sh', '.lua',
                     '.cc', '.hh', '.txt', '.csv', '.json', '.jsonl',
                     '.md', '.rst', '.yaml', '.yml', '.xml', '.toml'}
        files = sorted([
            str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in code_exts
        ])
        if not files:
            print(f"No data files found in {args.data_dir}")
            sys.exit(1)
        print(f"Loading {len(files)} file(s) from {args.data_dir}")
    else:
        cache = FOUNDATION_DIR / "downloads"
        print("\nDownloading public-domain books...")
        files = download_all(cache)

    train_foundation(
        data_paths=files,
        max_chars=args.data_chars,
        epochs=args.epochs,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        prompt_ratio=args.prompt_ratio,
        resume=args.resume,
        save_every=args.save_every,
        device=args.device,
        use_amp=not args.no_amp,
    )

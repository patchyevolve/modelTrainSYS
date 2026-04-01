"""
text_model.py — thin compatibility shim.

The real language model is now HMTLanguageModel in implementations.py.
This file re-exports it under the old names so training_ui.py and
model_chat.py keep working without changes.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations import HMTLanguageModel as MambaLM   # same API


# ── Training helpers ──────────────────────────────────────────────────────────

def lm_train_step(model: MambaLM, optimizer: torch.optim.Optimizer,
                  xb: torch.Tensor, yb: torch.Tensor,
                  clip: float = 1.0) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(xb)
    loss   = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        yb.view(-1), ignore_index=0)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def lm_val_loss(model: MambaLM,
                loader: torch.utils.data.DataLoader,
                device: torch.device = None,
                max_batches: int = 100) -> float:
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    total = 0.0
    count = 0
    for xb, yb in loader:
        if isinstance(xb, (list, tuple)):
            xb = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in xb)
            yb = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in yb)
        else:
            xb = xb.to(device)
            yb = yb.to(device)
        logits = model(xb)
        total += F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1), ignore_index=0).item()
        count += 1
        if count >= max_batches:
            break
    return total / max(count, 1)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_lm(model: MambaLM, tokenizer, config: Dict, path: str) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config":     config,
        "char2idx":         tokenizer.char2idx,
        "idx2char":         {str(k): v for k, v in tokenizer.idx2char.items()},
        "vocab_size":       tokenizer.vocab_size,
    }, path)


def load_lm(path: str, device: str = "cpu") -> Tuple:
    from text_dataset import CharTokenizer
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    # Map old MambaLM keys → HMTLanguageModel keys
    hmt_cfg = {
        "vocab_size":  cfg.get("vocab_size",  256),
        "dim":         cfg.get("hidden_dim",  cfg.get("dim", 256)),
        "num_layers":  cfg.get("num_layers",  4),
        "num_heads":   cfg.get("num_heads",   8),
        "num_scales":  cfg.get("num_scales",  3),
        "max_seq":     cfg.get("seq_len",     cfg.get("max_seq", 512)),
        "dropout":     cfg.get("dropout",     0.1),
    }
    # embed_dim alias
    if "embed_dim" in cfg and "dim" not in cfg:
        hmt_cfg["dim"] = cfg["embed_dim"]

    model = MambaLM(**hmt_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    tok = CharTokenizer()
    tok.char2idx   = ckpt["char2idx"]
    tok.idx2char   = {int(k): v for k, v in ckpt["idx2char"].items()}
    tok.vocab_size = ckpt["vocab_size"]

    return model, tok

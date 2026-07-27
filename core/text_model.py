import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger("TextModel")


def _torch():
    import torch
    return torch


def _nn():
    import torch.nn as nn
    return nn


def lm_val_loss(model, loader, device=None, max_batches: int = 50,
                ignore_index: int = -100) -> float:
    torch = _torch()
    nn = _nn()
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    total = 0.0
    count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                                   ignore_index=ignore_index)
        total += loss.item()
        count += 1
        if count >= max_batches:
            break
    return total / max(count, 1)


def save_lm(model, tokenizer, config: Dict,
            optimizer=None,
            scheduler=None,
            epoch: int = 0, loss: float = 0.0, path: str = "model.pt",
            ema_state_dict: Optional[Dict] = None,
            files_trained_on: Optional[list] = None,
            resume_count: int = 0) -> None:
    torch = _torch()
    if hasattr(model, "module"):  # unwrap DataParallel/DDP
        model = model.module
    safe_cfg = dict(config)
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        safe_cfg["num_layers"] = len(model.backbone.blocks)
    if hasattr(model, "dim"):
        safe_cfg["hidden_dim"] = model.dim
    state = {
        "model_state_dict": model.state_dict(),
        "model_config": safe_cfg,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "loss": loss,
        "_files_trained_on": files_trained_on or [],
        "_resume_count": resume_count,
        "_version": 2,
    }
    if ema_state_dict:
        state["ema_state_dict"] = ema_state_dict
    base = Path(path)
    tok_base = base.parent / base.stem
    tok_path = str(tok_base) + ".tokenizer"
    tokenizer.save(tok_path)
    state["tokenizer_path"] = tok_path
    torch.save(state, path)

    # Also save a .json manifest for the UI model list
    meta_path = Path(path).with_suffix(".json")
    import json
    from datetime import datetime
    if not meta_path.exists():
        meta = {
            "name": Path(path).stem,
            "model_type": "text_generation",
            "epochs": config.get("epochs", config.get("epoch", 0)),
            "loss": str(loss) if loss else "—",
            "created": datetime.now().isoformat(),
            "status": "ready",
            "config": config,
            "weights_file": str(path),
            "_resume_count": resume_count,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def load_lm(path: str, device: str = "cpu"):
    torch = _torch()
    from core.implementations import HMTLanguageModel
    from data.advanced_tokenizer import AdvancedTokenizer

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config") or ckpt.get("config") or {}
    sd = ckpt.get("model_state_dict") or ckpt

    block_keys = [k for k in sd if k.startswith("backbone.blocks.") and ".norm1.weight" in k]
    actual_layers = max(int(k.split(".")[2]) for k in block_keys) + 1 if block_keys else cfg.get("num_layers", 4)

    hmt_cfg = {
        "vocab_size": cfg.get("vocab_size", 8192),
        "dim": cfg.get("hidden_dim", cfg.get("dim", 256)),
        "num_layers": actual_layers,
        "num_heads": cfg.get("num_heads", 8),
        "n_kv_heads": cfg.get("num_kv_heads", cfg.get("n_kv_heads")),
        "max_seq": cfg.get("max_seq", cfg.get("seq_len", 512)),
        "dropout": cfg.get("dropout", 0.1),
        "use_gradient_checkpointing": cfg.get("use_gradient_checkpointing", False),
    }

    model = HMTLanguageModel(**hmt_cfg).to(device)
    sd = ckpt.get("model_state_dict") or ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning(f"Missing keys: {missing}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")
    model.eval()

    tok_path = ckpt.get("tokenizer_path", "")
    if tok_path and Path(tok_path).exists():
        tokenizer = AdvancedTokenizer.load(tok_path)
    else:
        base = Path(path)
        found = False
        for candidate in [
            base.parent / f"{base.stem}.tokenizer",
            base.with_suffix(".tokenizer"),
            base.with_suffix(".tok"),
        ]:
            if candidate.exists():
                tokenizer = AdvancedTokenizer.load(str(candidate))
                found = True
                break
        if not found:
            tokenizer = AdvancedTokenizer(vocab_size=hmt_cfg["vocab_size"])
            log.warning(f"Tokenizer '{tok_path}' not found; no local variant; using blank")

    return model, tokenizer

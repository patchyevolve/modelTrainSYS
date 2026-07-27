import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.transformer import TransformerBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass
class GenConfig:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    top_p_min: float = 0.0
    min_p: float = 0.0
    typical_p: float = 0.0
    repetition_penalty: float = 2.0
    repetition_penalty_range: int = 100
    max_new_tokens: int = 300
    eos_id: Optional[int] = None
    stop_sequences: List[int] = field(default_factory=list)
    ban_tokens: List[int] = field(default_factory=list)
    logit_bias: Dict[int, float] = field(default_factory=dict)


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, dim: int = 512, num_layers: int = 6, num_heads: int = 8,
                 n_kv_heads: Optional[int] = None, max_seq: int = 4096,
                 dropout: float = 0.1, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.dim = dim
        self.max_seq = max_seq
        self.use_gc = use_gradient_checkpointing
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, num_heads=num_heads, n_kv_heads=n_kv_heads,
                ff_mult=4, dropout=dropout, max_len=max_seq)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)

    def reset_cache(self):
        for blk in self.blocks:
            blk.reset_cache()

    def forward(self, x: torch.Tensor, use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_gc and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_cache, cache_offset, use_reentrant=False)
            else:
                x = blk(x, use_cache=use_cache, cache_offset=cache_offset)
        return self.norm_out(x)


def _apply_repetition_penalty(logits: torch.Tensor, ids: List[int],
                               penalty: float, window: int) -> None:
    if penalty <= 0.0 or not ids:
        return
    for pid in set(ids[-window:]):
        if 0 <= pid < logits.size(-1):
            logits[pid] -= penalty


def _apply_top_k(logits: torch.Tensor, k: int) -> None:
    if k <= 0:
        return
    kth = torch.topk(logits, min(k, logits.size(-1))).values[-1]
    logits.masked_fill_(logits < kth, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> None:
    if not (0.0 < p < 1.0):
        return
    sl, si = torch.sort(logits, descending=True)
    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
    sl[cp - F.softmax(sl, dim=-1) > p] = float("-inf")
    logits.copy_(torch.zeros_like(logits).scatter_(0, si, sl))


def _apply_min_p(logits: torch.Tensor, min_p: float) -> None:
    if min_p <= 0.0:
        return
    top_prob = F.softmax(logits, dim=-1).max()
    threshold = top_prob * min_p
    logits[F.softmax(logits, dim=-1) < threshold] = float("-inf")


def _apply_typical_p(logits: torch.Tensor, typical_p: float) -> None:
    if typical_p <= 0.0:
        return
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    neg_entropy = entropy - (-torch.log(probs + 1e-10))
    sorted_neg_entropy, indices = torch.sort(neg_entropy)
    cumsum = torch.cumsum(F.softmax(logits, dim=-1)[indices], dim=-1)
    mask = cumsum > typical_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    logits[indices[mask]] = float("-inf")


def _sample_from_logits(logits: torch.Tensor) -> int:
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


class HMTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512,
                 num_layers: int = 6, num_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 max_seq: int = 2048, dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.dim = dim

        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.backbone = DecoderOnlyTransformer(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            n_kv_heads=n_kv_heads, max_seq=max_seq, dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing)
        self.head = LMHead(dim, vocab_size)
        self.head.proj.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head.proj:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, token_ids: torch.Tensor,
                use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        x = self.drop(self.embed(token_ids))
        x = self.backbone(x, use_cache=use_cache, cache_offset=cache_offset)
        return self.head(x)

    def reset_cache(self):
        self.backbone.reset_cache()

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], cfg: Optional[GenConfig] = None,
                 device: str = "cpu") -> List[int]:
        if cfg is None:
            cfg = GenConfig()

        self.eval()
        self.reset_cache()

        ids = list(prompt_ids) if prompt_ids else [0]

        ctx_len = min(len(ids), self.max_seq)
        ctx = ids[-ctx_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = self.forward(x, use_cache=True, cache_offset=0)[0, -1]
        offset = len(ctx)

        new_ids: List[int] = []

        for _ in range(cfg.max_new_tokens):
            _apply_repetition_penalty(logits, ids, cfg.repetition_penalty, cfg.repetition_penalty_range)

            for tid, bias in cfg.logit_bias.items():
                if 0 <= tid < logits.size(-1):
                    logits[tid] += bias

            for tid in cfg.ban_tokens:
                if 0 <= tid < logits.size(-1):
                    logits[tid] = float("-inf")

            if cfg.eos_id is not None and 0 <= cfg.eos_id < logits.size(-1):
                pass

            logits = logits / max(cfg.temperature, 1e-8)

            _apply_top_k(logits, cfg.top_k)
            _apply_top_p(logits, cfg.top_p)
            _apply_min_p(logits, cfg.min_p)
            _apply_typical_p(logits, cfg.typical_p)

            next_id = _sample_from_logits(logits)

            ids.append(next_id)
            new_ids.append(next_id)

            if cfg.eos_id is not None and next_id == cfg.eos_id:
                break

            if cfg.stop_sequences and len(new_ids) >= len(cfg.stop_sequences):
                if new_ids[-len(cfg.stop_sequences):] == cfg.stop_sequences:
                    new_ids = new_ids[:-len(cfg.stop_sequences)]
                    break

            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = self.forward(x, use_cache=True, cache_offset=offset)[0, -1]
            offset += 1

        return new_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

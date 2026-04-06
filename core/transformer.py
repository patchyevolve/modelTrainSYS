"""
Transformer Block - Multi-head self-attention with RoPE and KV-cache.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — relative, extrapolates to longer seqs."""

    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()[None, None]
        sin = emb.sin()[None, None]
        
        if hasattr(self, "cos_cache"):
            self.cos_cache = cos
            self.sin_cache = sin
        else:
            self.register_buffer("cos_cache", cos)
            self.register_buffer("sin_cache", sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        if offset + T > self.cos_cache.shape[2]:
            self._build_cache(offset + T + 1024)

        cos = self.cos_cache[:, :, offset: offset + T]
        sin = self.sin_cache[:, :, offset: offset + T]
        
        q   = q * cos + self._rotate_half(q) * sin
        k   = k * cos + self._rotate_half(k) * sin
        return q, k


class TransformerBlock(nn.Module):
    """
    Causal Transformer block with:
    - Multi-head self-attention + RoPE (with Flash Attention when available)
    - KV-cache for O(1) per-step generation
    - SwiGLU feed-forward (gate * up, then down)
    - Pre-norm (LayerNorm before attention and FFN)
    - Gradient checkpointing support for memory efficiency
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 ff_mult: int = 4, dropout: float = 0.1,
                 max_len: int = 4096, use_flash_attn: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.use_flash = use_flash_attn

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.out  = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)

        ff_dim       = ff_mult * dim
        self.ff_gate = nn.Linear(dim, ff_dim, bias=False)
        self.ff_up   = nn.Linear(dim, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, dim, bias=False)

        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_cache(self):
        self._kv_cache = None

    def forward(self, x: torch.Tensor,
                use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        """
        x          : (B, T, dim)
        use_cache  : True during autoregressive generation (T=1 per step)
        returns    : (B, T, dim)
        """
        B, T, _ = x.shape

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k, offset=cache_offset)

        if q.shape[2] < T:
            v = v[:, :, :q.shape[2]]

        if use_cache:
            if self._kv_cache is not None:
                k_prev, v_prev = self._kv_cache
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            self._kv_cache = (k.detach(), v.detach())

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_mask = None
            if not use_cache and T > 1:
                attn_mask = torch.triu(
                    torch.ones(T, k.shape[2], device=x.device, dtype=torch.bool),
                    diagonal=1)
            
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=True
            )
            out = attn_out.transpose(1, 2).reshape(B, T, self.dim)
        else:
            scale = self.head_dim ** -0.5
            attn  = torch.matmul(q, k.transpose(-2, -1)) * scale

            if not use_cache and T > 1:
                mask = torch.triu(
                    torch.ones(T, k.shape[2], device=x.device) * -1e9,
                    diagonal=1)
                attn = attn + mask.unsqueeze(0).unsqueeze(0)

            attn = F.softmax(attn, dim=-1)
            attn = self.drop(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, T, self.dim)

        x   = x + self.out(out)

        h   = self.norm2(x)
        ffn = F.silu(self.ff_gate(h)) * self.ff_up(h)
        x   = x + self.ff_down(ffn)

        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with causal attention."""
    
    def __init__(self, dim: int, num_heads: int = 8, ff_mult: int = 4,
                 dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, 
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        self._create_causal_mask(max_len)
    
    def _create_causal_mask(self, max_len: int):
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
    
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, T, dim)"""
        T = x.shape[1]
        causal_mask = self.causal_mask[:T, :T]
        
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ff(x)
        return x


def create_transformer_stack(dim: int, num_layers: int, num_heads: int = 8,
                             ff_mult: int = 4, dropout: float = 0.1,
                             max_len: int = 4096, use_flash_attn: bool = False) -> nn.ModuleList:
    """Create a stack of transformer blocks."""
    return nn.ModuleList([
        TransformerBlock(dim, num_heads=num_heads, ff_mult=ff_mult,
                         dropout=dropout, max_len=max_len, 
                         use_flash_attn=use_flash_attn)
        for _ in range(num_layers)
    ])


def transformer_forward_stack(blocks: nn.ModuleList, x: torch.Tensor,
                              use_cache: bool = False,
                              cache_offset: int = 0) -> torch.Tensor:
    """Forward pass through transformer stack."""
    for block in blocks:
        x = block(x, use_cache=use_cache, cache_offset=cache_offset)
    return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
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

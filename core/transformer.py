import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _yarn_freq_scale(
    dim: int, max_len: int, base: float, scale: float,
    original_max_len: int, yarn_factor: float
) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    low, high = dim // 4, dim // 2 - 1
    t = torch.arange(dim // 2).float()
    ramp = (t - low) / max(high - low, 1)
    ramp = ramp.clamp(0, 1)
    ramp = yarn_factor * ramp
    scale_factor = (scale - 1.0) * ramp + 1.0
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim) * scale_factor)
    return torch.where(
        t < low, 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim * (scale ** (dim / (dim - 2))))),
        freq_extra
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_len: int = 8192,
        base: float = 10000.0, scale: float = 1.0,
        original_max_len: int = 4096, yarn_factor: float = 1.0
    ):
        super().__init__()
        self.scale = scale
        self.original_max_len = original_max_len

        if scale > 1.0:
            if yarn_factor > 1.0:
                inv_freq = _yarn_freq_scale(dim, max_len, base, scale, original_max_len, yarn_factor)
            else:
                base_scaled = base * (scale ** (dim / (dim - 2)))
                inv_freq = 1.0 / (base_scaled ** (torch.arange(0, dim, 2).float() / dim))
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        if self.scale > 1.0:
            t = t / self.scale
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

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
        needed = offset + T
        if needed > self.cos_cache.shape[2]:
            self._build_cache(needed + 2048)

        cos = self.cos_cache[:, :, offset: offset + T]
        sin = self.sin_cache[:, :, offset: offset + T]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 ff_mult: int = 4, dropout: float = 0.1,
                 max_len: int = 4096, use_flash_attn: bool = False,
                 rope_base: float = 10000.0, rope_scale: float = 1.0,
                 rope_original_max_len: int = 4096, yarn_factor: float = 1.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.head_dim = dim // num_heads
        self.kv_head_dim = self.head_dim
        self.use_flash = use_flash_attn

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.kv_head_dim, bias=False)
        self.out = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(
            self.head_dim, max_len=max_len, base=rope_base,
            scale=rope_scale, original_max_len=rope_original_max_len,
            yarn_factor=yarn_factor,
        )

        ff_dim = ff_mult * dim
        self.ff_gate = nn.Linear(dim, ff_dim, bias=False)
        self.ff_up = nn.Linear(dim, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, dim, bias=False)

        self._kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._gqa_repeat = num_heads // self.n_kv_heads
        self._causal_mask: Optional[torch.Tensor] = None

    def reset_cache(self):
        self._kv_cache = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        if self._gqa_repeat <= 1:
            return x
        return x[:, :, None].expand(B, H, self._gqa_repeat, T, D).reshape(B, H * self._gqa_repeat, T, D)

    def forward(self, x: torch.Tensor,
                use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        B, T, _ = x.shape

        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, T, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, T, self.n_kv_heads, self.kv_head_dim).transpose(1, 2)

        q, k = self.rope(q, k, offset=cache_offset)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if use_cache:
            if self._kv_cache is not None:
                k_prev, v_prev = self._kv_cache
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            self._kv_cache = (k.detach(), v.detach())

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=(not use_cache and T > 1)
            )
            out = attn_out.transpose(1, 2).reshape(B, T, self.dim)
        else:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if not use_cache and T > 1:
                if self._causal_mask is None or self._causal_mask.shape[-2] < T or self._causal_mask.device != x.device:
                    self._causal_mask = torch.triu(
                        torch.full((T, k.shape[2]), -1e9, device=x.device),
                        diagonal=1)
                attn = attn + self._causal_mask[:T, :k.shape[2]].unsqueeze(0).unsqueeze(0)
            attn = F.softmax(attn, dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, T, self.dim)

        x = x + self.out(out)

        h = self.norm2(x)
        ffn = F.silu(self.ff_gate(h)) * self.ff_up(h)
        x = x + self.ff_down(ffn)

        return x

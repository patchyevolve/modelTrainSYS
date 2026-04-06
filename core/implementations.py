"""
Concrete implementations — real Hierarchical Mamba + Transformer backbone.

Uses modular components from core.mamba and core.transformer.

Modality-specific heads:
  - LMHead          : next-token prediction (text, code generation)
  - ClassifierHead  : binary / multi-class (tabular, detection, cybersecurity)
  - RegressionHead  : continuous output (statistics, forecasting)
  - ImagePatchHead  : image patch reconstruction / classification

Data feeders:
  - ImageFeeder, TextFeeder, StatisticalFeeder (unchanged API)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image

try:
    from core.architecture import (
        DataFeeder, Encoder, Decoder, Reflector, Trainer,
        DataType, ModuleConfig, ComponentType
    )
except ImportError:
    from .architecture import (
        DataFeeder, Encoder, Decoder, Reflector, Trainer,
        DataType, ModuleConfig, ComponentType
    )

try:
    from core.mamba import (
        HierarchicalMambaBlock,
        create_hierarchical_mamba_stack,
        hierarchical_mamba_forward,
        SSMCore,
    )
except ImportError:
    try:
        from .mamba import (
            HierarchicalMambaBlock,
            create_hierarchical_mamba_stack,
            hierarchical_mamba_forward,
            SSMCore,
        )
    except (ImportError, NameError):
        HierarchicalMambaBlock = None
        create_hierarchical_mamba_stack = None
        hierarchical_mamba_forward = None
        SSMCore = None

MAMBA_AVAILABLE = HierarchicalMambaBlock is not None

try:
    from core.transformer import TransformerBlock, RotaryEmbedding
except ImportError:
    from .transformer import TransformerBlock, RotaryEmbedding


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


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK SIMPLE MAMBA BLOCK (when hierarchical not available)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleMambaBlock(nn.Module):
    """
    Simple gated state-space block — fallback when HierarchicalMambaBlock unavailable.
    """

    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 dt_rank: int = None, conv_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim     = dim
        self.d_inner = expand * dim
        self.dropout = dropout

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=conv_size, padding=conv_size - 1,
            groups=self.d_inner, bias=True)
        self.W_f = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.W_i = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.W_o = nn.Linear(self.d_inner, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.constant_(self.W_f.bias, 1.0)
        nn.init.zeros_(self.W_i.bias)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x

        xz       = self.in_proj(x)
        x_in, z  = xz.chunk(2, dim=-1)
        xc = self.conv1d(x_in.transpose(1, 2))[:, :, :T]
        xc = F.silu(xc).transpose(1, 2)
        h = xc
        out = self.W_o(h * F.silu(z))
        return self.norm(self.drop(out) + residual)


# Backward compatibility alias
MambaBlock = SimpleMambaBlock


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL MAMBA-TRANSFORMER  (universal backbone)
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalMambaTransformer(nn.Module):
    """
    Universal backbone for ALL modalities.

    Architecture
    ────────────
    Input (any modality, already embedded to dim)
        │
        ├── Scale 1 (fine):   full sequence, dim
        ├── Scale 2 (medium): full sequence, dim//2 → projected back
        └── Scale 3 (coarse): full sequence, dim//4 → projected back
        │
        └── Learned weighted fusion → dim
        │
        └── Depth stack: [MambaBlock, TransformerBlock] × num_layers
        │
        └── Final LayerNorm → (B, T, dim)

    Attach any head:
        LMHead          → text / code generation
        ClassifierHead  → detection / classification
        RegressionHead  → statistics / forecasting
        ImagePatchHead  → image understanding

    Parameters
    ──────────
    dim        : model dimension (hidden size)
    num_layers : number of Mamba+Transformer pairs
    num_heads  : attention heads in each TransformerBlock
    num_scales : parallel scales (default 3)
    max_seq    : maximum sequence length (RoPE cache)
    dropout    : dropout rate
    use_gradient_checkpointing: recompute activations to save memory
    """

    def __init__(
        self,
        dim:        int   = 512,
        num_layers: int   = 6,
        num_heads:  int   = 8,
        num_scales: int   = 3,
        max_seq:    int   = 4096,
        dropout:    float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim        = dim
        self.num_layers = num_layers
        self.max_seq    = max_seq
        self.use_gc     = use_gradient_checkpointing

        # ── Use Hierarchical Mamba blocks from core.mamba ─────────────────
        if MAMBA_AVAILABLE and HierarchicalMambaBlock is not None:
            self.mamba_blocks = nn.ModuleList([
                HierarchicalMambaBlock(
                    dim=dim,
                    d_state=16,
                    expand=2,
                    num_scales=num_scales,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:
            # Fallback to simple mamba block
            self.mamba_blocks = nn.ModuleList([
                SimpleMambaBlock(dim, d_state=16, expand=2, dropout=dropout)
                for _ in range(num_layers)
            ])

        self.attn_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=num_heads,
                             ff_mult=4, dropout=dropout,
                             max_len=max_seq)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.drop     = nn.Dropout(dropout)

    def reset_cache(self):
        """Clear KV caches — call before each new generation sequence."""
        for blk in self.attn_blocks:
            blk.reset_cache()

    def _layer_forward(self, x, mamba, attn, use_cache, cache_offset):
        """Single layer forward for gradient checkpointing."""
        x = mamba(x)
        x = attn(x, use_cache=use_cache, cache_offset=cache_offset)
        return x

    def forward(
        self,
        x:          torch.Tensor,
        use_cache:  bool = False,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        """
        x : (B, T, dim)  — already embedded input
        returns: (B, T, dim)
        """
        # ── Multi-scale fusion ────────────────────────────────────────────
        weights = F.softmax(self.scale_weights, dim=0)
        fused   = torch.zeros_like(x)
        for i, (proj_in, proj_out) in enumerate(
                zip(self.scale_projs_in, self.scale_projs_out)):
            fused = fused + weights[i] * proj_out(proj_in(x))
        x = self.drop(fused)

        # ── Alternating Mamba + Transformer layers ────────────────────────
        if self.use_gc and self.training:
            # Gradient checkpointing: recompute activations to save memory
            for mamba, attn in zip(self.mamba_blocks, self.attn_blocks):
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, x, mamba, attn, use_cache, cache_offset,
                    use_reentrant=False
                )
        else:
            for mamba, attn in zip(self.mamba_blocks, self.attn_blocks):
                x = mamba(x)
                x = attn(x, use_cache=use_cache, cache_offset=cache_offset)

        return self.norm_out(x)


# ── Backward-compatible alias for existing code ───────────────────────────────
class HierarchicalMambaEncoder(Encoder):
    """
    Wraps HierarchicalMambaTransformer in the Encoder interface.
    Used by MLSystemOrchestrator pipeline.
    """

    def initialize(self) -> None:
        p = self.config.params
        self.input_dim  = p.get("input_dim",  256)
        self.hidden_dim = p.get("hidden_dim", 512)
        self.num_layers = p.get("num_layers", 4)
        self.num_heads  = p.get("num_heads",  8)
        self.num_scales = p.get("num_scales", 3)
        self.max_seq    = p.get("max_seq",    4096)

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.backbone   = HierarchicalMambaTransformer(
            dim        = self.hidden_dim,
            num_layers = self.num_layers,
            num_heads  = self.num_heads,
            num_scales = self.num_scales,
            max_seq    = self.max_seq,
        )
        super().initialize()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# ═══════════════════════════════════════════════════════════════════════════════
# MODALITY-SPECIFIC HEADS
# ═══════════════════════════════════════════════════════════════════════════════

class LMHead(nn.Module):
    """Next-token prediction head for text / code generation."""
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, T, vocab_size)


class ClassifierHead(nn.Module):
    """Binary or multi-class classification head."""
    def __init__(self, dim: int, num_classes: int = 1, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1]
        elif self.pool == "max":
            x = x.max(dim=1).values
        return self.proj(self.norm(x))   # (B, num_classes)


class RegressionHead(nn.Module):
    """Continuous output head for statistics / forecasting."""
    def __init__(self, dim: int, output_dim: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x.mean(dim=1)))   # (B, output_dim)


class ImagePatchHead(nn.Module):
    """
    Patch-based image head.
    Splits image into patches, projects to dim, then uses HMT backbone.
    Returns per-patch features or a classification logit.
    """
    def __init__(self, dim: int, patch_size: int = 16,
                 img_channels: int = 3, num_classes: int = 0):
        super().__init__()
        self.patch_size = patch_size
        patch_dim       = patch_size * patch_size * img_channels
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, dim))
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(dim, num_classes)

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, num_patches, patch_dim)"""
        B, C, H, W = img.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, \
            f"Image {H}x{W} not divisible by patch_size {p}"
        img = img.reshape(B, C, H // p, p, W // p, p)
        img = img.permute(0, 2, 4, 1, 3, 5)           # (B, H/p, W/p, C, p, p)
        img = img.reshape(B, (H // p) * (W // p), C * p * p)
        return img

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W) → (B, num_patches+1, dim)"""
        patches = self.patch_proj(self.patchify(img))  # (B, N, dim)
        cls     = self.cls_token.expand(img.shape[0], -1, -1)
        x       = torch.cat([cls, patches], dim=1)     # (B, N+1, dim)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODELS  (backbone + head, ready to train)
# ═══════════════════════════════════════════════════════════════════════════════

class HMTLanguageModel(nn.Module):
    """
    Full language model: embedding → HMT backbone → LM head.
    Handles text, code, any character/token sequence.
    Supports KV-cache for fast generation.
    """

    def __init__(self, vocab_size: int, dim: int = 512,
                 num_layers: int = 6, num_heads: int = 8,
                 num_scales: int = 3, max_seq: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq    = max_seq
        self.dim        = dim

        self.embed    = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.drop     = nn.Dropout(dropout)
        self.backbone = HierarchicalMambaTransformer(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            num_scales=num_scales, max_seq=max_seq, dropout=dropout)
        self.head     = LMHead(dim, vocab_size)

        # Weight tying
        self.head.proj.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, token_ids: torch.Tensor,
                use_cache: bool = False,
                cache_offset: int = 0) -> torch.Tensor:
        x      = self.drop(self.embed(token_ids))
        x      = self.backbone(x, use_cache=use_cache,
                               cache_offset=cache_offset)
        return self.head(x)                            # (B, T, vocab)

    def reset_cache(self):
        self.backbone.reset_cache()

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new: int = 200,
                 temperature: float = 0.8, top_k: int = 40,
                 top_p: float = 0.9, repetition_penalty: float = 1.1,
                 eos_id: Optional[int] = None,
                 device: str = "cpu") -> List[int]:
        self.eval()
        self.reset_cache()
        ids = list(prompt_ids)
        if not ids:
            ids = [0]  # dummy start token if empty

        # Prefill: process the full prompt at once
        # Use a context window that fits in max_seq
        ctx_len = min(len(ids), self.max_seq)
        ctx     = ids[-ctx_len:]
        x       = torch.tensor([ctx], dtype=torch.long, device=device)
        
        # Prefill forward pass populates KV cache for the prompt
        # We only need the logits for the last token to pick the FIRST new token
        logits = self.forward(x, use_cache=True, cache_offset=0)[0, -1]
        offset = len(ctx)

        for _ in range(max_new):
            # 1. Repetition penalty
            if repetition_penalty != 1.0:
                for pid in set(ids[-64:]):
                    logits[pid] /= repetition_penalty

            # 2. Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # 3. Top-K filtering
            if top_k > 0:
                kth    = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            # 4. Top-P (nucleus) sampling
            if 0.0 < top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp     = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(0, si, sl)

            # 5. Sample next token
            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

            # 6. Forward pass for the next step (using the token we just sampled)
            x      = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = self.forward(x, use_cache=True,
                                  cache_offset=offset)[0, -1]
            offset += 1

        return ids[len(prompt_ids):]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HMTClassifier(nn.Module):
    """
    Tabular / sequence classifier.
    Works for: cybersecurity, statistics, any CSV data.
    """

    def __init__(self, input_dim: int, num_classes: int = 1,
                 dim: int = 256, num_layers: int = 4,
                 num_heads: int = 8, num_scales: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.backbone   = HierarchicalMambaTransformer(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            num_scales=num_scales, max_seq=512, dropout=dropout)
        self.head       = ClassifierHead(dim, num_classes, pool="mean")
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, feat_dim) or (B, T, feat_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)                         # (B, 1, feat_dim)
        x = self.input_proj(x)
        x = self.backbone(x)
        return self.head(x)                            # (B, num_classes)


class HMTImageClassifier(nn.Module):
    """
    Image classifier using patch embeddings + HMT backbone.
    Works for: image classification, detection features.
    """

    def __init__(self, num_classes: int, dim: int = 512,
                 patch_size: int = 16, img_channels: int = 3,
                 num_layers: int = 6, num_heads: int = 8,
                 num_scales: int = 3, dropout: float = 0.1):
        super().__init__()
        self.patcher  = ImagePatchHead(dim, patch_size, img_channels)
        self.backbone = HierarchicalMambaTransformer(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            num_scales=num_scales, max_seq=4096, dropout=dropout)
        self.head     = ClassifierHead(dim, num_classes, pool="last")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, C, H, W)"""
        x = self.patcher(img)                          # (B, N+1, dim)
        x = self.backbone(x)
        return self.head(x)                            # (B, num_classes)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER DECODER  (backward-compatible, used by MLSystemOrchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerDecoder(Decoder):
    """
    Wraps TransformerBlock stack in the Decoder interface.
    Used by MLSystemOrchestrator pipeline.
    """

    def initialize(self) -> None:
        p = self.config.params
        self.latent_dim  = p.get("latent_dim",  512)
        self.output_dim  = p.get("output_dim",  256)
        self.num_heads   = p.get("num_heads",   8)
        self.num_layers  = p.get("num_layers",  3)
        self.max_seq     = p.get("max_seq",     4096)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.latent_dim, num_heads=self.num_heads,
                             max_len=self.max_seq)
            for _ in range(self.num_layers)
        ])
        self.norm     = nn.LayerNorm(self.latent_dim)
        self.out_proj = nn.Linear(self.latent_dim, self.output_dim)
        super().initialize()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


# ═══════════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING  (sinusoidal, backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

# PositionalEncoding is now defined at the top of the file after imports

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FEEDERS  (unchanged API)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageFeeder(DataFeeder):
    def initialize(self) -> None:
        self.supported_formats = {".jpg",".jpeg",".png",".bmp",".gif",".tiff"}
        super().initialize()

    def validate_data(self, data: Any) -> bool:
        if isinstance(data, (str, Path)):
            return str(data).lower().endswith(tuple(self.supported_formats))
        if isinstance(data, np.ndarray):
            return data.ndim in [2, 3]
        if isinstance(data, torch.Tensor):
            return data.ndim in [2, 3, 4]
        return False

    def preprocess(self, data: Any) -> torch.Tensor:
        if isinstance(data, (str, Path)):
            data = np.array(Image.open(data).convert("RGB"))
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.max() > 1:
            data = data / 255.0
        if data.ndim == 3:
            data = data.unsqueeze(0)
        return data

    def load_batch(self, batch_size: int, image_dir: Optional[str] = None,
                   **kw) -> Tuple[torch.Tensor, Dict]:
        if not image_dir:
            raise ValueError("image_dir required")
        paths = [p for p in Path(image_dir).glob("*")
                 if str(p).lower().endswith(tuple(self.supported_formats))]
        imgs  = [self.preprocess(p) for p in paths[:batch_size]
                 if self.validate_data(p)]
        batch = torch.cat(imgs) if imgs else torch.tensor([])
        return batch, {"batch_size": len(imgs), "image_count": len(paths)}

    def forward(self, data: Any) -> torch.Tensor:
        if not self.validate_data(data):
            raise ValueError(f"Invalid image: {type(data)}")
        return self.preprocess(data)


class TextFeeder(DataFeeder):
    def initialize(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.max_length = self.config.params.get("max_length", 512)
        super().initialize()

    def validate_data(self, data: Any) -> bool:
        return isinstance(data, (str, list))

    def build_vocab(self, texts: List[str]) -> None:
        tokens = set()
        for t in texts:
            tokens.update(t.lower().split())
        self.vocab = {tok: i for i, tok in enumerate(sorted(tokens))}

    def tokenize(self, text: str) -> List[int]:
        return [self.vocab.get(t, 0)
                for t in text.lower().split()][:self.max_length]

    def preprocess(self, data: Any) -> torch.Tensor:
        texts = data if isinstance(data, list) else [data]
        seqs  = []
        for t in texts:
            ids = self.tokenize(t)
            ids += [0] * (self.max_length - len(ids))
            seqs.append(ids[:self.max_length])
        return torch.tensor(seqs, dtype=torch.long)

    def load_batch(self, batch_size: int, text_file: Optional[str] = None,
                   **kw) -> Tuple[torch.Tensor, Dict]:
        if not text_file:
            raise ValueError("text_file required")
        with open(text_file) as f:
            texts = [l.strip() for l in f]
        if not self.vocab:
            self.build_vocab(texts)
        batch = self.preprocess(texts[:batch_size])
        return batch, {"batch_size": len(texts[:batch_size]),
                       "vocab_size": len(self.vocab)}

    def forward(self, data: Any) -> torch.Tensor:
        if not self.validate_data(data):
            raise ValueError(f"Invalid text: {type(data)}")
        return self.preprocess(data)


class StatisticalFeeder(DataFeeder):
    def initialize(self) -> None:
        self.scaler = None
        super().initialize()

    def validate_data(self, data: Any) -> bool:
        return isinstance(data, (list, np.ndarray, torch.Tensor))

    def preprocess(self, data: Any) -> torch.Tensor:
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if self.scaler:
            data = (data - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)
        return data

    def fit_scaler(self, data: np.ndarray) -> None:
        self.scaler = {"mean": data.mean(0), "std": data.std(0)}

    def load_batch(self, batch_size: int, data_array: Optional[np.ndarray] = None,
                   **kw) -> Tuple[torch.Tensor, Dict]:
        if data_array is None:
            raise ValueError("data_array required")
        if not self.scaler:
            self.fit_scaler(data_array)
        idx   = np.random.choice(len(data_array), batch_size, replace=False)
        batch = self.preprocess(data_array[idx])
        return batch, {"batch_size": batch_size,
                       "feature_dim": data_array.shape[-1]}

    def forward(self, data: Any) -> torch.Tensor:
        if not self.validate_data(data):
            raise ValueError(f"Invalid data: {type(data)}")
        return self.preprocess(data)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO & VIDEO FEEDERS  (for complete DataType coverage)
# ═══════════════════════════════════════════════════════════════════════════════

class AudioFeeder(DataFeeder):
    """Audio data feeder - handles .wav, .mp3, .flac files."""
    
    def initialize(self) -> None:
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        self.sample_rate = self.config.params.get("sample_rate", 16000)
        self.max_length = self.config.params.get("max_length", 16000)  # 1 second at 16kHz
        super().initialize()
    
    def validate_data(self, data: Any) -> bool:
        if isinstance(data, (str, Path)):
            return str(data).lower().endswith(tuple(self.supported_formats))
        if isinstance(data, np.ndarray):
            return data.ndim == 1 or (data.ndim == 2 and data.shape[0] <= 2)
        if isinstance(data, torch.Tensor):
            return data.ndim == 1 or (data.ndim == 2 and data.shape[0] <= 2)
        return False
    
    def preprocess(self, data: Any) -> torch.Tensor:
        if isinstance(data, (str, Path)):
            data = self._load_audio(data)
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        # Ensure mono and fixed length
        if data.ndim == 2:
            data = data.mean(dim=0)  # Convert stereo to mono
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data = torch.cat([data, torch.zeros(self.max_length - len(data))])
        return data.unsqueeze(0)  # Add batch dimension
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file - requires librosa or scipy."""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return torch.from_numpy(audio)
        except ImportError:
            try:
                from scipy.io import wavfile
                sr, audio = wavfile.read(path)
                if sr != self.sample_rate:
                    # Simple resample
                    ratio = self.sample_rate / sr
                    audio = audio[::int(ratio)] if ratio > 1 else np.repeat(audio, int(1/ratio))
                return torch.from_numpy(audio.astype(np.float32) / 32768.0)
            except Exception as e:
                raise ValueError(f"Could not load audio {path}: {e}")
    
    def load_batch(self, batch_size: int, audio_dir: Optional[str] = None,
                   **kw) -> Tuple[torch.Tensor, Dict]:
        if not audio_dir:
            raise ValueError("audio_dir required")
        paths = [p for p in Path(audio_dir).glob("*")
                 if str(p).lower().endswith(tuple(self.supported_formats))]
        audios = [self.preprocess(p) for p in paths[:batch_size] if self.validate_data(p)]
        batch = torch.cat(audios) if audios else torch.tensor([])
        return batch, {"batch_size": len(audios), "audio_count": len(paths)}
    
    def forward(self, data: Any) -> torch.Tensor:
        if not self.validate_data(data):
            raise ValueError(f"Invalid audio: {type(data)}")
        return self.preprocess(data)


class VideoFeeder(DataFeeder):
    """Video data feeder - handles .mp4, .avi, .mkv files."""
    
    def initialize(self) -> None:
        self.supported_formats = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        self.frame_rate = self.config.params.get("frame_rate", 30)
        self.max_frames = self.config.params.get("max_frames", 30)
        self.img_size = self.config.params.get("img_size", 112)
        super().initialize()
    
    def validate_data(self, data: Any) -> bool:
        if isinstance(data, (str, Path)):
            return str(data).lower().endswith(tuple(self.supported_formats))
        return False
    
    def preprocess(self, data: Any) -> torch.Tensor:
        if isinstance(data, (str, Path)):
            frames = self._extract_frames(data)
            return frames
        raise ValueError(f"Invalid video: {type(data)}")
    
    def _extract_frames(self, path: str) -> torch.Tensor:
        """Extract frames from video - requires cv2 (opencv)."""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frames = []
            frame_count = 0
            while cap.is_open() and frame_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize and convert BGR to RGB
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1
            cap.release()
            if not frames:
                return torch.zeros(self.max_frames, self.img_size, self.img_size, 3)
            # Pad if needed
            while len(frames) < self.max_frames:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            return torch.from_numpy(np.stack(frames))
        except ImportError:
            raise ImportError("opencv-python (cv2) required for video loading")
    
    def load_batch(self, batch_size: int, video_dir: Optional[str] = None,
                   **kw) -> Tuple[torch.Tensor, Dict]:
        if not video_dir:
            raise ValueError("video_dir required")
        paths = [p for p in Path(video_dir).glob("*")
                 if str(p).lower().endswith(tuple(self.supported_formats))]
        videos = [self.preprocess(p).unsqueeze(0) for p in paths[:batch_size] if self.validate_data(p)]
        batch = torch.cat(videos) if videos else torch.zeros(batch_size, self.max_frames, self.img_size, self.img_size, 3)
        return batch, {"batch_size": len(videos), "video_count": len(paths)}
    
    def forward(self, data: Any) -> torch.Tensor:
        if not self.validate_data(data):
            raise ValueError(f"Invalid video: {type(data)}")
        return self.preprocess(data)

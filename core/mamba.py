"""
Hierarchical Mamba Block — Multi-Scale Selective State Space Model
=================================================================
Key improvements over naive MambaBlock:
  • True selective SSM: Δ, B, C are input-dependent (data-driven gating)
  • Proper parallel scan via log-space cumsum (O(T) work, no Python loop)
  • Hierarchical multi-scale processing: SSM cores at stride 1 / 2 / 4
  • ZOH (zero-order hold) discretisation: Ā = exp(Δ·A), B̄ = Δ·B
  • Cross-scale fusion with learned weights + gated context blending
  • Head-parallel multi-head SSM within each scale
  • Dead-code-free: every parameter is actually used in the forward pass
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA_AVAILABLE = True


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Log-space parallel scan  (Heinsen, 2023)
# ─────────────────────────────────────────────────────────────────────────────

def parallel_scan_log(
    log_coeffs: torch.Tensor,   # (B, T, D)  — log|a_t|
    log_values: torch.Tensor,   # (B, T, D)  — log|b_t|
) -> torch.Tensor:
    """
    Numerically stable parallel scan for the linear recurrence
        h_t = a_t · h_{t-1} + b_t,   h_0 = 0
    using Heinsen's log-space trick + cumulative log-sum-exp.

    Complexity: O(T) sequential work, fully vectorised (no Python loop).
    Stable for arbitrarily long sequences and very small / large values.

    Returns h: (B, T, D)
    """
    # Prefix-sum of log coefficients: log(a_1 · a_2 · … · a_t)
    a_star = torch.cumsum(log_coeffs, dim=1)                       # (B, T, D)

    # log-sum-exp accumulation of log(b_s / (a_1…a_s)) over s ≤ t
    log_x0_plus_b_star = torch.logcumsumexp(
        log_values - a_star, dim=1)                                # (B, T, D)

    return torch.exp(a_star + log_x0_plus_b_star)                 # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Selective SSM scan  (ZOH-discretised, parallel)
# ─────────────────────────────────────────────────────────────────────────────

def selective_scan(
    x:   torch.Tensor,   # (B, T, d_inner)
    dt:  torch.Tensor,   # (B, T, d_inner) — raw (pre-softplus) step sizes
    A:   torch.Tensor,   # (d_inner, d_state) — negative log matrix
    B:   torch.Tensor,   # (B, T, d_state)  — input projection (selective)
    C:   torch.Tensor,   # (B, T, d_state)  — output projection (selective)
    D:   torch.Tensor,   # (d_inner,)        — skip connection weight
) -> torch.Tensor:
    """
    Discretised selective SSM:
        Ā_t = exp(Δ_t ⊙ A)                   [ZOH, element-wise]
        B̄_t = Δ_t ⊙ B_t
        h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t   [parallel scan]
        y_t = C_t · h_t + D ⊙ x_t

    Δ, B, C are all input-dependent → true selective / data-driven gating.
    Runs via log-space parallel scan; no Python loop over T.

    Returns y: (B, T, d_inner)
    """
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]

    dt = F.softplus(dt)                                            # (B, T, d_inner) > 0

    # ZOH discretisation ──────────────────────────────────────────────────
    # A is stored as -log|eigenvalues| so exp(dt·A) is stable decay in (0,1)
    # dA: (B, T, d_inner, d_state)
    dA = torch.exp(
        dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
    )
    # dB·x: (B, T, d_inner, d_state)
    dBx = (
        dt.unsqueeze(-1)          # (B, T, d_inner, 1)
        * B.unsqueeze(2)          # (B, T, 1,       d_state)
        * x.unsqueeze(-1)         # (B, T, d_inner, 1)
    )

    # Parallel scan over the (d_inner × d_state) recurrence ──────────────
    flat_len = d_inner * d_state
    log_dA   = torch.log(dA  .clamp(min=1e-38)).view(B_batch, T, flat_len)
    log_dBx  = torch.log(dBx .clamp(min=1e-38)).view(B_batch, T, flat_len)

    h = parallel_scan_log(log_dA, log_dBx)                        # (B, T, flat)
    h = h.view(B_batch, T, d_inner, d_state)                      # (B, T, d_inner, d_state)

    # Output projection ───────────────────────────────────────────────────
    # y_t = C_t · h_t  +  D ⊙ x_t
    y = torch.einsum("bts,btds->btd", C, h) + D * x              # (B, T, d_inner)
    return y


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Single-scale SSM core  (one head group)
# ─────────────────────────────────────────────────────────────────────────────

class SSMCore(nn.Module):
    """
    One selective-SSM block at a fixed temporal scale.

    Pipeline:
        x → depthwise causal conv1d → SiLU
          → x_proj → (dt_raw, B_ssm, C_ssm)   [all input-dependent]
          → selective_scan(x_conv, dt, A, B, C, D)
          → y  (same shape as x)
    """

    def __init__(
        self,
        d_inner:   int,
        d_state:   int = 16,
        dt_rank:   Optional[int] = None,
        conv_size: int = 4,
    ):
        super().__init__()
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.dt_rank  = dt_rank or max(1, d_inner // 16)

        # Local mixing — causal depthwise conv
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=d_inner,
            bias=True,
        )

        # Selective projections: x → (dt_low_rank, B, C) jointly
        self.x_proj = nn.Linear(
            d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        # Low-rank Δ up-projection with bias (controls initial step size)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # SSM state matrix A (stored as positive log for stability)
        # Initialise as HiPPO-like: eigenvalues 1…d_state
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)   # (d_state,)
        A_init = A_init.unsqueeze(0).expand(d_inner, -1)              # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A_init))                  # positive → -exp = decay

        # Skip connection weight D (every parameter used!)
        self.D = nn.Parameter(torch.ones(d_inner))

        self._init_weights()

    def _init_weights(self):
        # dt_proj bias: initialise so Δ starts in a reasonable range
        nn.init.uniform_(self.dt_proj.bias, math.log(1e-3), math.log(0.1))
        nn.init.zeros_(self.x_proj.weight)   # start selective proj near zero
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_inner) → (B, T, d_inner)"""
        B, T, _ = x.shape

        # 1. Causal depthwise conv (local pattern extraction)
        xc = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xc = F.silu(xc)                                            # (B, T, d_inner)

        # 2. Selective projections (all input-dependent)
        proj   = self.x_proj(xc)                                   # (B, T, dt_rank + 2*d_state)
        dt_raw, B_ssm, C_ssm = proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt_raw)                                  # (B, T, d_inner)

        # 3. SSM matrix A (negative: ensures stable decay)
        A = -torch.exp(self.A_log)                                 # (d_inner, d_state)

        # 4. Selective scan
        return selective_scan(xc, dt, A, B_ssm, C_ssm, self.D)   # (B, T, d_inner)


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Hierarchical Mamba Block
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalMambaBlock(nn.Module):
    """
    Hierarchical Selective State Space Block.

    Processes the input sequence at ``num_scales`` temporal resolutions
    simultaneously and fuses the results via learned cross-scale attention.

    Scale s uses stride 2^s:
        scale 0 — full resolution  (stride 1) — fine-grained / local
        scale 1 — half resolution  (stride 2) — medium-range context
        scale 2 — quarter res      (stride 4) — long-range / global

    Full forward pass:
        x  →  in_proj  →  [x_in | gate]
        x_in  →  ┌ scale-0 SSMCore (stride 1)  →  y0
                  ├ scale-1 SSMCore (stride 2)  →  y1  (up-sampled)
                  └ scale-2 SSMCore (stride 4)  →  y2  (up-sampled)
        [y0, y1, y2]  →  cross-scale fusion  →  fused
        out  =  out_proj( fused × SiLU(gate) )
        return  LayerNorm( Dropout(out) + residual )

    Args:
        dim:        model dimension
        d_state:    SSM state dimension  (default 16)
        expand:     inner expansion factor  (default 2)
        dt_rank:    rank for low-rank Δ projection  (default dim//16)
        conv_size:  depthwise conv kernel size  (default 4)
        num_scales: number of hierarchical scales  (default 3)
        dropout:    output dropout probability  (default 0.0)
    """

    def __init__(
        self,
        dim:        int,
        d_state:    int = 16,
        expand:     int = 2,
        dt_rank:    Optional[int] = None,
        conv_size:  int = 4,
        num_scales: int = 3,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.dim        = dim
        self.d_inner    = expand * dim
        self.num_scales = num_scales

        # ── Gated input projection ─────────────────────────────────────────
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # ── One SSM core per scale ─────────────────────────────────────────
        self.ssm_cores = nn.ModuleList([
            SSMCore(
                d_inner=self.d_inner,
                d_state=d_state,
                dt_rank=dt_rank,
                conv_size=conv_size,
            )
            for _ in range(num_scales)
        ])

        # ── Cross-scale fusion ─────────────────────────────────────────────
        # Learnable per-scale mixing weights (softmax-normalised at runtime)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

        # Gated cross-scale context:
        #   ctx = mean of all scale outputs  →  2-layer gate  →  sigmoid mask
        self.cross_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 2, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_inner // 2, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

        # ── Output ──────────────────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.norm     = nn.LayerNorm(dim)
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for m in self.cross_gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # ── Scale utilities ────────────────────────────────────────────────────

    @staticmethod
    def _downsample(x: torch.Tensor, stride: int) -> torch.Tensor:
        """Average-pool x along T by ``stride``."""
        if stride == 1:
            return x
        B, T, D = x.shape
        T_trim = (T // stride) * stride
        return x[:, :T_trim, :].reshape(B, T_trim // stride, stride, D).mean(2)

    @staticmethod
    def _upsample(x: torch.Tensor, T_target: int) -> torch.Tensor:
        """Nearest-neighbour upsample x from its T back to T_target."""
        B, T_src, D = x.shape
        if T_src == T_target:
            return x
        # repeat_interleave + trim/pad
        factor = math.ceil(T_target / T_src)
        x_up   = x.repeat_interleave(factor, dim=1)                # (B, T_src*factor, D)
        if x_up.shape[1] > T_target:
            return x_up[:, :T_target, :]
        # pad last frame if needed (rare rounding edge)
        pad = x_up[:, -1:, :].expand(B, T_target - x_up.shape[1], D)
        return torch.cat([x_up, pad], dim=1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
        Returns:
            (B, T, dim)
        """
        B, T, _ = x.shape
        residual = x

        # ── 1. Gated projection ────────────────────────────────────────────
        xz          = self.in_proj(x)                              # (B, T, 2·d_inner)
        x_in, gate  = xz.chunk(2, dim=-1)                         # each (B, T, d_inner)

        # ── 2. Multi-scale SSM ─────────────────────────────────────────────
        scale_outs: List[torch.Tensor] = []
        for s, core in enumerate(self.ssm_cores):
            stride = 2 ** s                                        # 1, 2, 4, …
            x_s    = self._downsample(x_in, stride)               # (B, T//stride, d_inner)
            y_s    = core(x_s)                                     # (B, T//stride, d_inner)
            y_up   = self._upsample(y_s, T)                       # (B, T, d_inner)
            scale_outs.append(y_up)

        # ── 3. Cross-scale fusion ──────────────────────────────────────────
        # Softmax-normalised weighted combination
        w      = F.softmax(self.scale_weights, dim=0)              # (num_scales,)
        fused  = sum(w[s] * scale_outs[s] for s in range(self.num_scales))

        # Gated cross-scale context modulation
        ctx    = torch.stack(scale_outs, dim=0).mean(dim=0)       # (B, T, d_inner)
        fused  = fused * self.cross_gate(ctx)                     # element-wise gate

        # ── 4. SiLU output gate (uses `gate` — never wasted) ──────────────
        out = fused * F.silu(gate)                                 # (B, T, d_inner)

        # ── 5. Project back + residual + norm ─────────────────────────────
        out = self.out_proj(out)                                   # (B, T, dim)
        return self.norm(self.drop(out) + residual)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Stack utilities
# ─────────────────────────────────────────────────────────────────────────────

def create_hierarchical_mamba_stack(
    dim:        int,
    num_layers: int,
    dropout:    float = 0.1,
    d_state:    int   = 16,
    expand:     int   = 2,
    num_scales: int   = 3,
) -> nn.ModuleList:
    """Build a stack of ``num_layers`` HierarchicalMambaBlocks."""
    return nn.ModuleList([
        HierarchicalMambaBlock(
            dim=dim,
            d_state=d_state,
            expand=expand,
            num_scales=num_scales,
            dropout=dropout,
        )
        for _ in range(num_layers)
    ])


def hierarchical_mamba_forward(
    blocks: nn.ModuleList,
    x:      torch.Tensor,
) -> torch.Tensor:
    """Sequential forward through a HierarchicalMamba stack."""
    for block in blocks:
        x = block(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, dim = 2, 128, 64
    x = torch.randn(B, T, dim)

    block = HierarchicalMambaBlock(dim=dim, d_state=16, expand=2, num_scales=3)
    y = block(x)

    assert y.shape == (B, T, dim), f"Shape mismatch: {y.shape}"
    assert not y.isnan().any(),    "NaNs in output!"
    print(f"HierarchicalMambaBlock  in={tuple(x.shape)}  out={tuple(y.shape)}  ✓")

    # Stack test
    stack = create_hierarchical_mamba_stack(dim=dim, num_layers=4, dropout=0.1)
    y2    = hierarchical_mamba_forward(stack, x)
    assert y2.shape == (B, T, dim)
    print(f"Stack (4 layers)        in={tuple(x.shape)}  out={tuple(y2.shape)}  ✓")

    # Parameter count
    total = sum(p.numel() for p in block.parameters())
    print(f"Block parameter count: {total:,}")
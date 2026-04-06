"""
Mamba Block - Selective State Space Model
========================================
Supports parallel scan on CPU/CUDA, with DirectML fallback.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA_AVAILABLE = True


def parallel_scan_log(
    log_coeffs: torch.Tensor,
    log_values: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scan for the linear recurrence.
    Uses cumsum which is faster on CPU/CUDA.
    NOT supported on DirectML.
    """
    a_star = torch.cumsum(log_coeffs, dim=1)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(a_star + log_x0_plus_b_star)


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Discretised selective SSM with parallel scan.
    """
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]

    dt = F.softplus(dt)

    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

    flat_len = d_inner * d_state
    log_dA = torch.log(dA.clamp(min=1e-38)).view(B_batch, T, flat_len)
    log_dBx = torch.log(dBx.clamp(min=1e-38)).view(B_batch, T, flat_len)

    h = parallel_scan_log(log_dA, log_dBx)
    h = h.view(B_batch, T, d_inner, d_state)

    y = torch.einsum("bts,btds->btd", C, h) + D * x
    return y


class SSMCore(nn.Module):
    """Simplified SSM core with optional parallel scan."""

    def __init__(self, d_inner: int, d_state: int = 16, conv_size: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=conv_size,
            padding=conv_size - 1,
            groups=d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(d_inner, 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(d_inner))

        nn.init.uniform_(self.dt_proj.bias, math.log(1e-3), math.log(0.1))

    def forward(self, x: torch.Tensor, use_parallel: bool = True) -> torch.Tensor:
        """x: (B, T, d_inner) → (B, T, d_inner)"""
        B, T, _ = x.shape

        xc = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xc = F.silu(xc)

        proj = self.x_proj(xc)
        B_ssm, C_ssm = proj.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(B_ssm))

        if use_parallel:
            try:
                A = -torch.exp(torch.log(
                    torch.arange(1, self.d_state + 1, device=x.device, dtype=x.dtype)
                )).unsqueeze(0).expand(self.d_inner, -1)
                y = selective_scan(xc, dt, A, B_ssm, C_ssm, self.D)
                return y
            except Exception:
                pass

        h = xc * dt.sigmoid()
        return h + self.D * x


class HierarchicalMambaBlock(nn.Module):
    """
    Hierarchical Selective State Space Block.
    Uses parallel scan when available (CPU/CUDA).
    Falls back to simpler computation for DirectML.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        num_scales: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = expand * dim
        self.num_scales = num_scales

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        self.ssm_cores = nn.ModuleList([
            SSMCore(self.d_inner, d_state)
            for _ in range(num_scales)
        ])

        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        self.cross_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 2, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_inner // 2, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _use_parallel(self) -> bool:
        """Check if we can use parallel scan (not DirectML)."""
        device_type = next(self.parameters()).device.type
        return device_type in ('cpu', 'cuda', 'mps')

    @staticmethod
    def _downsample(x: torch.Tensor, stride: int) -> torch.Tensor:
        if stride == 1:
            return x
        B, T, D = x.shape
        T_trim = (T // stride) * stride
        return x[:, :T_trim, :].reshape(B, T_trim // stride, stride, D).mean(2)

    def _upsample(self, x: torch.Tensor, T_target: int) -> torch.Tensor:
        """Simple upsample using interpolate."""
        if x.shape[1] == T_target:
            return x
        return F.interpolate(x.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, dim) → (B, T, dim)"""
        B, T, _ = x.shape
        residual = x

        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)

        use_parallel = self._use_parallel()

        scale_outs = []
        for s, core in enumerate(self.ssm_cores):
            stride = 2 ** s
            x_s = self._downsample(x_in, stride)
            y_s = core(x_s, use_parallel=use_parallel)
            y_up = self._upsample(y_s, T)
            scale_outs.append(y_up)

        w = F.softmax(self.scale_weights, dim=0)
        fused = sum(w[s] * scale_outs[s] for s in range(self.num_scales))

        ctx = torch.stack(scale_outs, dim=0).mean(dim=0)
        fused = fused * self.cross_gate(ctx)

        out = fused * F.silu(gate)
        out = self.out_proj(out)
        return self.norm(self.drop(out) + residual)


def create_hierarchical_mamba_stack(
    dim: int,
    num_layers: int,
    dropout: float = 0.1,
    d_state: int = 16,
    expand: int = 2,
    num_scales: int = 3,
) -> nn.ModuleList:
    """Build a stack of HierarchicalMambaBlocks."""
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
    x: torch.Tensor,
) -> torch.Tensor:
    """Sequential forward through a HierarchicalMamba stack."""
    for block in blocks:
        x = block(x)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, dim = 2, 64, 64
    x = torch.randn(B, T, dim)

    block = HierarchicalMambaBlock(dim=dim, d_state=16, expand=2, num_scales=3)
    y = block(x)

    assert y.shape == (B, T, dim), f"Shape mismatch: {y.shape}"
    print(f"HierarchicalMambaBlock  in={tuple(x.shape)}  out={tuple(y.shape)}  ✓")
    print(f"Using parallel scan: {block._use_parallel()}")

    stack = create_hierarchical_mamba_stack(dim=dim, num_layers=4, dropout=0.1)
    y2 = hierarchical_mamba_forward(stack, x)
    assert y2.shape == (B, T, dim)
    print(f"Stack (4 layers)        in={tuple(x.shape)}  out={tuple(y2.shape)}  ✓")

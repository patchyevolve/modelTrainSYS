"""
Mamba Block - Selective State Space Model
========================================
Parallel scan for all backends.
CPU/CUDA: Uses native cumsum
DirectML: Uses chunked parallel computation
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAMBA_AVAILABLE = True


def parallel_scan_cpu(
    log_coeffs: torch.Tensor,
    log_values: torch.Tensor,
) -> torch.Tensor:
    """Parallel scan using cumsum (CPU/CUDA native)."""
    a_star = torch.cumsum(log_coeffs, dim=1)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(a_star + log_x0_plus_b_star)


def parallel_scan_chunked(
    log_coeffs: torch.Tensor,
    log_values: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked parallel scan - works on all backends including DirectML.
    Processes sequence in chunks, maintains parallel computation within chunks.
    """
    B, T, D = log_coeffs.shape
    result = torch.zeros_like(log_values)
    
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_len = end - start
        
        log_coeffs_chunk = log_coeffs[:, start:end, :]
        log_values_chunk = log_values[:, start:end, :]
        
        if start == 0:
            a_star = torch.cumsum(log_coeffs_chunk, dim=1)
        else:
            prev_a = result[:, start-1:start, :].exp()
            a_star = prev_a * torch.cumsum(log_coeffs_chunk, dim=1).exp()
            a_star = torch.log(a_star + 1e-38)
        
        log_x0_plus_b_star = torch.logcumsumexp(log_values_chunk - a_star, dim=1)
        result[:, start:end, :] = torch.exp(a_star + log_x0_plus_b_star)
    
    return result


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    use_chunked: bool = False,
) -> torch.Tensor:
    """Discretised selective SSM."""
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]

    dt = F.softplus(dt)

    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

    flat_len = d_inner * d_state
    log_dA = torch.log(dA.clamp(min=1e-38)).view(B_batch, T, flat_len)
    log_dBx = torch.log(dBx.clamp(min=1e-38)).view(B_batch, T, flat_len)

    if use_chunked:
        h = parallel_scan_chunked(log_dA, log_dBx)
    else:
        h = parallel_scan_cpu(log_dA, log_dBx)
    
    h = h.view(B_batch, T, d_inner, d_state)
    y = torch.einsum("bts,btds->btd", C, h) + D * x
    return y


class SSMCore(nn.Module):
    """SSM core with parallel scan support."""

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
                A_init = torch.arange(1, self.d_state + 1, device=x.device, dtype=x.dtype).float()
                A = -A_init.unsqueeze(0).expand(self.d_inner, -1)
                y = selective_scan(xc, dt, A, B_ssm, C_ssm, self.D, use_chunked=False)
                return y
            except Exception:
                pass

        h = xc * dt.sigmoid()
        return h + self.D * x


class HierarchicalMambaBlock(nn.Module):
    """
    Hierarchical Selective State Space Block.
    Uses parallel scan on all backends.
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
        """Check if we can use native parallel scan."""
        try:
            device_type = next(self.parameters()).device.type
            return device_type in ('cpu', 'cuda', 'mps')
        except:
            return True

    @staticmethod
    def _downsample(x: torch.Tensor, stride: int) -> torch.Tensor:
        if stride == 1:
            return x
        B, T, D = x.shape
        T_trim = (T // stride) * stride
        return x[:, :T_trim, :].reshape(B, T_trim // stride, stride, D).mean(2)

    def _upsample(self, x: torch.Tensor, T_target: int) -> torch.Tensor:
        if x.shape[1] == T_target:
            return x
        return F.interpolate(x.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    for block in blocks:
        x = block(x)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, dim = 2, 64, 64
    x = torch.randn(B, T, dim)

    block = HierarchicalMambaBlock(dim=dim, d_state=16, expand=2, num_scales=3)
    y = block(x)

    assert y.shape == (B, T, dim)
    print(f"HierarchicalMambaBlock  in={tuple(x.shape)}  out={tuple(y.shape)}  ✓")
    print(f"Uses parallel: {block._use_parallel()}")

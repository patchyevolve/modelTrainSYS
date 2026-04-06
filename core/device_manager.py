"""
Device manager — picks the best available compute device.

Priority: CUDA > DirectML (if model is large enough) > CPU

For AMD iGPU (Radeon 680M):
  - DirectML is only faster when model_params > ~5M AND batch_size >= 64
  - Below that threshold, CPU+MKL is faster due to transfer overhead
"""

import torch
import os
from typing import Tuple


def get_best_device(model_params: int = 0,
                    batch_size: int = 32,
                    force: str = "auto") -> Tuple[object, str]:
    """
    Returns (device, device_name_str).

    force: "auto" | "cpu" | "cuda" | "dml"
    
    Default: GPU (CUDA/DirectML) when available, CPU otherwise.
    """
    if force == "cpu":
        return _cpu_device()

    # CUDA (NVIDIA) - preferred
    if force == "cuda" or (force == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                return torch.device("cuda:0"), f"CUDA — {name}"
            except:
                pass

    # DirectML (AMD/Intel iGPU on Windows)
    if force == "dml" or force == "auto":
        try:
            import torch_directml
            dml = torch_directml.device()
            # Test if DirectML works
            test_tensor = torch.randn(2, 32, device=dml)
            _ = test_tensor * 2
            return dml, "DirectML — GPU (parallel enabled)"
        except ImportError:
            pass
        except Exception as e:
            print(f"DirectML test failed: {e}")

    return _cpu_device()


def _cpu_device(note: str = ""):
    n = os.cpu_count() or 4
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(max(2, n // 2))
    except RuntimeError:
        pass
    msg = f"CPU — {n} cores (MKL)"
    if note:
        msg += f"  [{note}]"
    return torch.device("cpu"), msg


def move_batch(batch, device):
    """Move a (x, y) batch tuple to device."""
    if isinstance(batch, (list, tuple)):
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t
                     for t in batch)
    return batch.to(device) if isinstance(batch, torch.Tensor) else batch


def device_info() -> str:
    """Return a human-readable string of available devices."""
    lines = []
    lines.append(f"CPU: {os.cpu_count()} cores, MKL={torch.backends.mkl.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"CUDA GPU {i}: {p.name} ({p.total_memory//1024//1024} MB)")
    try:
        import torch_directml
        lines.append("DirectML: AMD Radeon 680M (iGPU, shared RAM)")
    except ImportError:
        lines.append("DirectML: not installed")
    return "\n".join(lines)

from __future__ import annotations

import os
from typing import Optional

import torch


def parse_env_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def cuda_required(default: bool = False) -> bool:
    return parse_env_bool(os.environ.get("ISATCR_REQUIRE_CUDA"), default=default)


def select_torch_device(
    context: str,
    *,
    require_cuda: Optional[bool] = None,
) -> torch.device:
    must_use_cuda = cuda_required(default=False) if require_cuda is None else bool(require_cuda)
    cuda_available = torch.cuda.is_available()
    if must_use_cuda and not cuda_available:
        raise RuntimeError(
            f"{context} requires CUDA, but this Python environment cannot see a CUDA-enabled PyTorch build. "
            "Install a CUDA PyTorch wheel in .venv or run with --allow-cpu where supported."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(
            f"[GPU] {context}: torch={torch.__version__}, device=cuda:0 ({device_name})",
            flush=True,
        )
    else:
        print(f"[GPU] {context}: torch={torch.__version__}, device=cpu", flush=True)
    return device


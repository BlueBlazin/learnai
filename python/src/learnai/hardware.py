"""What hardware this machine has, for ``requires`` (docs/ARCHITECTURE.md section 3.3).

A requirement is ``cpu`` (always met), ``mlx-gpu`` (an Apple GPU that MLX can use), or
``cuda`` (an NVIDIA GPU that torch can use). Detection imports the framework, so it
happens only for requirements a problem actually names.

``LEARNAI_HARDWARE`` overrides detection with a comma-separated list of what to
pretend is present, for example ``LEARNAI_HARDWARE=cpu`` (nothing but the CPU) or
``LEARNAI_HARDWARE=cpu,cuda``. It exists for the judge's own tests and so that an
author can see how a problem behaves on a machine they do not have; it cannot make
missing hardware work.
"""

from __future__ import annotations

import os
from functools import cache

REQUIREMENTS = ("cpu", "mlx-gpu", "cuda")

_LABELS = {
    "cpu": "a CPU",
    "mlx-gpu": "an Apple GPU (MLX with Metal)",
    "cuda": "an NVIDIA GPU (CUDA)",
}


def describe(requirement: str) -> str:
    """The requirement in words: "an NVIDIA GPU (CUDA)"."""
    return _LABELS.get(requirement, requirement)


def missing_message(requirement: str) -> str:
    """Why a case was skipped: one line, shown to the learner."""
    return f"needs {describe(requirement)}, which this machine does not have"


@cache
def _detect(requirement: str) -> bool:
    if requirement == "cpu":
        return True
    if requirement == "mlx-gpu":
        try:
            import mlx.core as mx

            return bool(mx.metal.is_available()) and "gpu" in str(mx.default_device()).lower()
        except Exception:  # noqa: BLE001 - not installed, or no Metal device
            return False
    if requirement == "cuda":
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False
    return False


def cuda_devices() -> list[dict[str, object]]:
    """The NVIDIA GPUs torch can use: ``[{"index", "name", "memory_bytes"}, ...]``.

    Empty when torch is missing, was built without CUDA, or finds no device. A device
    whose name or memory cannot be read is still listed, with nulls.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - a broken driver must not break `learnai doctor`
        return []
    devices: list[dict[str, object]] = []
    for index in range(count):
        try:
            name: str | None = str(torch.cuda.get_device_name(index))
            memory: int | None = int(torch.cuda.get_device_properties(index).total_memory)
        except Exception:  # noqa: BLE001 - the device exists even if it will not say more
            name, memory = None, None
        devices.append({"index": index, "name": name, "memory_bytes": memory})
    return devices


def mlx_gpu_device() -> dict[str, object] | None:
    """The Apple GPU MLX uses, ``{"name", "memory_bytes"}``, or None."""
    if not _detect("mlx-gpu"):
        return None
    try:
        import mlx.core as mx

        info = (mx.device_info if hasattr(mx, "device_info") else mx.metal.device_info)()
        name, memory = info.get("device_name"), info.get("memory_size")
        return {
            "name": str(name) if name else None,
            "memory_bytes": int(memory) if memory else None,
        }
    except Exception:  # noqa: BLE001
        return {"name": None, "memory_bytes": None}


def report() -> dict[str, object]:
    """What ``learnai doctor --json`` prints under ``hardware`` (see cli.py).

    Real detection, ignoring ``LEARNAI_HARDWARE``: the override changes which cases
    are skipped, not what the machine has. ``cuda.present`` is exactly what makes
    ``requires="cuda"`` cases run: ``torch.cuda.is_available()``.
    """
    devices = cuda_devices() if _detect("cuda") else []
    first = devices[0] if devices else {}
    try:
        import torch

        cuda_build: str | None = torch.version.cuda
    except Exception:  # noqa: BLE001 - torch missing or broken: nothing to report
        cuda_build = None
    apple = mlx_gpu_device()
    return {
        "cpu": {"present": True, "cores": os.cpu_count()},
        "mlx-gpu": {
            "present": apple is not None,
            "name": (apple or {}).get("name"),
            "memory_bytes": (apple or {}).get("memory_bytes"),
        },
        "cuda": {
            "present": _detect("cuda"),
            "name": first.get("name"),
            "memory_bytes": first.get("memory_bytes"),
            "devices": devices,
            "torch_cuda_version": cuda_build,
        },
    }


def available(requirement: str) -> bool:
    """Whether this machine meets ``requirement``."""
    override = os.environ.get("LEARNAI_HARDWARE")
    if override is not None:
        present = {item.strip() for item in override.split(",") if item.strip()} | {"cpu"}
        return requirement in present
    return _detect(requirement)

"""What hardware this machine has, for ``requires`` (docs/ARCHITECTURE.md section 3.3).

A requirement is ``cpu`` (always met), ``mlx-gpu`` (an Apple GPU that MLX can use), or
``cuda`` (an NVIDIA GPU that torch can use). Detection imports the framework, so it
happens only for requirements a problem actually names.

``WHETSTONE_HARDWARE`` overrides detection with a comma-separated list of what to
pretend is present, for example ``WHETSTONE_HARDWARE=cpu`` (nothing but the CPU) or
``WHETSTONE_HARDWARE=cpu,cuda``. It exists for the judge's own tests and so that an
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


def available(requirement: str) -> bool:
    """Whether this machine meets ``requirement``."""
    override = os.environ.get("WHETSTONE_HARDWARE")
    if override is not None:
        present = {item.strip() for item in override.split(",") if item.strip()} | {"cpu"}
        return requirement in present
    return _detect(requirement)

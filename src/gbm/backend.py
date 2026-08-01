"""Device selection and torch interop.

torch is an optional dependency: the whole numerical core (processes, pricing,
risk) runs on NumPy alone. This module is the only place that touches torch, so
importing :mod:`gbm` without torch installed works.

Fixes carried over from the audit of the old code:

* ``torch.backends.mps`` and ``torch.mps.*`` were called unguarded, raising
  ``AttributeError`` on older builds. Every access here is capability-checked.
* Benchmarking was gated on ``torch.cuda.is_available()``, so it silently never
  ran on Apple MPS. The gate is now "any accelerator".
* ``.numpy()`` was called without ``.cpu()`` in four places, crashing on any
  non-CPU device. :func:`to_numpy` always detaches and moves to host first.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .logging import get_logger

log = get_logger(__name__)

_TORCH_IMPORT_ERROR = (
    "PyTorch is required for this feature. Install it with:\n"
    '    pip install "gbm-quant[ml]"'
)


def torch_available() -> bool:
    """True if torch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def require_torch() -> Any:
    """Import and return torch, or raise a message that says how to install it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_TORCH_IMPORT_ERROR) from exc
    return torch


def _mps_available(torch: Any) -> bool:
    """MPS check that tolerates torch builds without the attribute chain."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_available = getattr(mps, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def get_device(preference: str = "auto") -> Any:
    """Resolve a device string to a ``torch.device``.

    Args:
        preference: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``. ``"auto"``
            prefers CUDA, then MPS, then CPU. An explicit choice that is not
            available raises rather than silently falling back — silent
            downgrades are how the old code hid failures.
    """
    torch = require_torch()
    pref = preference.lower()

    if pref == "auto":
        if _cuda_available(torch):
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not _cuda_available(torch):
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    if pref == "mps":
        if not _mps_available(torch):
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device("mps")

    raise ValueError(f"Unknown device {preference!r}; expected auto, cpu, cuda or mps.")


def has_accelerator() -> bool:
    """True if any non-CPU device is usable. Used to gate benchmarks."""
    if not torch_available():
        return False
    torch = require_torch()
    return _cuda_available(torch) or _mps_available(torch)


def describe_device(device: Any) -> str:
    """Human-readable device description for logs."""
    torch = require_torch()
    kind = device.type if hasattr(device, "type") else str(device)

    if kind == "cuda":
        props = torch.cuda.get_device_properties(0)
        return (
            f"CUDA: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1e9:.1f} GB, CUDA {torch.version.cuda})"
        )
    if kind == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert a torch tensor to a NumPy array, from any device.

    Always detaches and moves to host, which the old ``.numpy()`` calls did not.
    Passes arrays through unchanged.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return tensor.detach().to("cpu").numpy()
    return np.asarray(tensor)


def synchronize(device: Any) -> None:
    """Block until queued work on ``device`` finishes. No-op on CPU.

    Required for honest timings; guarded because ``torch.mps.synchronize`` does
    not exist on every build.
    """
    torch = require_torch()
    kind = device.type if hasattr(device, "type") else str(device)

    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        sync = getattr(getattr(torch, "mps", None), "synchronize", None)
        if callable(sync):
            sync()


def empty_cache(device: Any) -> None:
    """Release cached device memory where the backend supports it."""
    torch = require_torch()
    kind = device.type if hasattr(device, "type") else str(device)

    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        clear = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if callable(clear):
            clear()


def seed_everything(seed: int | None) -> None:
    """Seed torch's global RNG so model init and dropout are reproducible.

    The old code never called ``torch.manual_seed`` at all, so weight init and
    MC-dropout uncertainty were non-reproducible even when NumPy was seeded.

    Note this deliberately does *not* seed global NumPy state: simulators own
    private ``default_rng`` instances instead.
    """
    if seed is None:
        return
    if not torch_available():
        return
    torch = require_torch()
    torch.manual_seed(seed)
    if _cuda_available(torch):
        torch.cuda.manual_seed_all(seed)

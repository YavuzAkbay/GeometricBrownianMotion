"""Shared helpers for the path simulators."""

from __future__ import annotations

import numpy as np

from ..config import SimConfig


def make_rng(cfg: SimConfig) -> np.random.Generator:
    """Build a private RNG for one simulation.

    The old simulators called ``np.random.seed(42)`` *inside* the function
    body. That clobbered global process state and, worse, made consecutive
    calls replay the same stream — so a Heston run and its "independent" GBM
    baseline drew identical normals. A local Generator avoids both.
    """
    return np.random.default_rng(cfg.seed)


def validate_inputs(s0: float, sigma: float | np.ndarray | None = None) -> None:
    """Reject inputs that would produce silently meaningless paths."""
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError(f"s0 must be a positive finite number, got {s0}")
    if sigma is not None:
        arr = np.asarray(sigma, dtype=float)
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError(f"sigma must be non-negative and finite, got {sigma}")


def exponentiate_log_paths(s0: float, log_increments: np.ndarray) -> np.ndarray:
    """Turn per-step log increments into a price path array.

    Prepends log(S0) as a zero column then cumulative-sums, so paths are
    ``S0 * exp(cumsum(...))`` and therefore *strictly positive* by
    construction. The legacy additive-Euler scheme had no such guarantee.

    Args:
        s0: Initial price.
        log_increments: Array of shape ``(n_paths, steps)``.

    Returns:
        Array of shape ``(n_paths, steps + 1)`` starting at ``s0``.
    """
    n_paths = log_increments.shape[0]
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return s0 * np.exp(log_paths)

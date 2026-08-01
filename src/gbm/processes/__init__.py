"""Stochastic process simulators.

One implementation per model. The old codebase kept a CPU copy in ``gbm.py``
and a hand-ported GPU copy in ``enhanced_gbm.py``; the two had already drifted
into different discretisation schemes under the same names.
"""

from __future__ import annotations

from .gbm import GBMParams, estimate_parameters, simulate_gbm
from .heston import feller_condition, simulate_heston
from .jump import draw_jump_counts, expected_jump_multiplier, simulate_merton_jump
from .regime import (
    DEFAULT_TRANSITION_MATRIX,
    simulate_regime_switching,
    stationary_distribution,
)

__all__ = [
    "DEFAULT_TRANSITION_MATRIX",
    "GBMParams",
    "draw_jump_counts",
    "estimate_parameters",
    "expected_jump_multiplier",
    "feller_condition",
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton_jump",
    "simulate_regime_switching",
    "stationary_distribution",
]

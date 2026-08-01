"""Regime-switching GBM.

Each path follows a Markov chain over market states; within a state the price
evolves as GBM with that state's drift and volatility. Regimes are drawn per
path, so different paths sit in different states at the same time.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .base import exponentiate_log_paths, make_rng, validate_inputs

# Illustrative bull / bear / crisis parameters. Exposed as a named constant
# rather than being pasted as a literal at three separate call sites.
DEFAULT_TRANSITION_MATRIX = np.array(
    [
        [0.98, 0.015, 0.005],
        [0.05, 0.90, 0.05],
        [0.02, 0.18, 0.80],
    ]
)


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Return the Markov chain's stationary distribution.

    Used as the default initial-state distribution, so paths do not all start
    in state 0 and spend the first stretch of the horizon relaxing away from it.
    """
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vec = np.real(eigenvectors[:, idx])
    vec = np.abs(vec)
    total = vec.sum()
    if total <= 0:  # pragma: no cover - defensive
        return np.full(len(vec), 1.0 / len(vec))
    return vec / total


def _validate(
    mu_states: np.ndarray, sigma_states: np.ndarray, transition_matrix: np.ndarray
) -> None:
    n_states = len(mu_states)

    if len(sigma_states) != n_states:
        raise ValueError(
            f"mu_states and sigma_states must have equal length, "
            f"got {n_states} and {len(sigma_states)}"
        )
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(
            f"transition_matrix must be {n_states}x{n_states}, got {transition_matrix.shape}"
        )
    if np.any(sigma_states < 0):
        raise ValueError("sigma_states must be non-negative.")
    if np.any(transition_matrix < 0):
        raise ValueError("transition_matrix entries must be non-negative.")

    row_sums = transition_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"transition_matrix rows must sum to 1, got {row_sums}")


def simulate_regime_switching(
    s0: float,
    mu_states: np.ndarray,
    sigma_states: np.ndarray,
    transition_matrix: np.ndarray | None = None,
    cfg: SimConfig | None = None,
    initial_distribution: np.ndarray | None = None,
    return_regimes: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate regime-switching GBM paths.

    Args:
        s0: Initial price.
        mu_states: Annualised drift per state, shape ``(n_states,)``.
        sigma_states: Annualised volatility per state, shape ``(n_states,)``.
        transition_matrix: Row-stochastic ``(n_states, n_states)`` matrix of
            per-step transition probabilities. Defaults to
            :data:`DEFAULT_TRANSITION_MATRIX`.
        cfg: Simulation settings.
        initial_distribution: Starting state probabilities. Defaults to the
            chain's stationary distribution.
        return_regimes: Also return the per-step regime index array.

    Returns:
        Price paths of shape ``(n_paths, steps + 1)``, or a
        ``(prices, regimes)`` tuple.
    """
    cfg = cfg or SimConfig()
    validate_inputs(s0, sigma_states)

    mu_states = np.asarray(mu_states, dtype=float)
    sigma_states = np.asarray(sigma_states, dtype=float)
    if transition_matrix is None:
        transition_matrix = DEFAULT_TRANSITION_MATRIX
    transition_matrix = np.asarray(transition_matrix, dtype=float)

    _validate(mu_states, sigma_states, transition_matrix)

    rng = make_rng(cfg)
    dt = cfg.dt
    sqrt_dt = np.sqrt(dt)
    n, steps = cfg.n_paths, cfg.steps
    n_states = len(mu_states)

    if initial_distribution is None:
        initial_distribution = stationary_distribution(transition_matrix)
    initial_distribution = np.asarray(initial_distribution, dtype=float)
    initial_distribution = initial_distribution / initial_distribution.sum()

    regimes = np.empty((n, steps + 1), dtype=np.int8)
    regimes[:, 0] = rng.choice(n_states, size=n, p=initial_distribution)

    # Inverse-CDF sampling, vectorised across paths: one uniform per path per
    # step, compared against the current state's cumulative row.
    cumulative = np.cumsum(transition_matrix, axis=1)
    uniforms = rng.random((n, steps))

    for t in range(steps):
        thresholds = cumulative[regimes[:, t]]
        regimes[:, t + 1] = (uniforms[:, t, None] > thresholds).sum(axis=1)

    # Regime governing step t is the state at the START of the interval.
    active = regimes[:, :-1]
    mu_path = mu_states[active]
    sigma_path = sigma_states[active]

    z = rng.standard_normal((n, steps))
    log_increments = (mu_path - 0.5 * sigma_path**2) * dt + sigma_path * sqrt_dt * z

    prices = exponentiate_log_paths(s0, log_increments)

    if return_regimes:
        return prices, regimes
    return prices

"""Heston stochastic volatility model.

The variance follows a CIR process correlated with the price::

    dS_t = mu S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

Discretised with Andersen's **full truncation** scheme: the drift and diffusion
use ``max(v, 0)`` while ``v`` itself is allowed to go negative before being
truncated at read time. This is the standard low-bias choice and is applied
consistently, unlike the legacy code which clamped at ``1e-8`` on GPU and
``0.0`` on CPU while its comment claimed "reflection".
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .base import make_rng, validate_inputs


def feller_condition(kappa: float, theta: float, sigma_v: float) -> bool:
    """True if ``2 kappa theta >= sigma_v^2``.

    When it holds, the CIR variance process stays strictly positive in
    continuous time. Violation is legitimate (real calibrations often violate
    it) and full truncation handles it — so this is reported, never silently
    "corrected" by rewriting the user's parameters as the legacy code did.
    """
    return 2.0 * kappa * theta >= sigma_v**2


def simulate_heston(
    s0: float,
    mu: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    cfg: SimConfig | None = None,
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths.

    Args:
        s0: Initial price.
        mu: Annualised drift of the price process.
        v0: Initial **variance** (not volatility). For 20% vol, pass 0.04.
        kappa: Mean-reversion speed of the variance.
        theta: Long-run variance.
        sigma_v: Volatility of variance.
        rho: Correlation between the price and variance shocks, in [-1, 1].
            Negative values give the usual leverage effect.
        cfg: Simulation settings.
        return_variance: Also return the variance paths.

    Returns:
        Price paths of shape ``(n_paths, steps + 1)``, or a
        ``(prices, variance)`` tuple when ``return_variance`` is set.

    Raises:
        ValueError: On non-positive ``s0``, negative ``v0``/``theta``/
            ``kappa``/``sigma_v``, or ``rho`` outside [-1, 1].
    """
    cfg = cfg or SimConfig()
    validate_inputs(s0)

    if v0 < 0:
        raise ValueError(f"v0 is a variance and must be >= 0, got {v0}")
    if theta < 0:
        raise ValueError(f"theta is a variance and must be >= 0, got {theta}")
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if sigma_v < 0:
        raise ValueError(f"sigma_v must be >= 0, got {sigma_v}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    rng = make_rng(cfg)
    dt = cfg.dt
    sqrt_dt = np.sqrt(dt)
    n, steps = cfg.n_paths, cfg.steps

    log_prices = np.empty((n, steps + 1))
    variance = np.empty((n, steps + 1))

    log_prices[:, 0] = np.log(s0)
    # v0 is written once, from the caller's value. The legacy code wrote theta
    # here and then recomputed theta afterwards, so v0 was silently wrong.
    variance[:, 0] = v0

    # Correlated shocks via Cholesky of [[1, rho], [rho, 1]].
    z1 = rng.standard_normal((n, steps))
    z2 = rng.standard_normal((n, steps))
    dw_v = z1 * sqrt_dt
    dw_s = (rho * z1 + np.sqrt(max(1.0 - rho**2, 0.0)) * z2) * sqrt_dt

    v = np.full(n, float(v0))
    log_s = np.full(n, np.log(s0))

    for t in range(steps):
        # Full truncation: every use of v within the step reads max(v, 0).
        v_pos = np.maximum(v, 0.0)
        vol = np.sqrt(v_pos)

        # The price step uses the variance at the START of the interval. The
        # legacy scheme substituted the freshly-computed v_{t+1}, which is not
        # adapted to the filtration and corrupts the rho leverage effect.
        log_s = log_s + (mu - 0.5 * v_pos) * dt + vol * dw_s[:, t]

        v = v + kappa * (theta - v_pos) * dt + sigma_v * vol * dw_v[:, t]

        log_prices[:, t + 1] = log_s
        variance[:, t + 1] = np.maximum(v, 0.0)

    prices = np.exp(log_prices)

    if return_variance:
        return prices, variance
    return prices

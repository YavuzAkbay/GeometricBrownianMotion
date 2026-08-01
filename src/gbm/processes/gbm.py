"""Geometric Brownian Motion and its parameter estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, SimConfig
from .base import exponentiate_log_paths, make_rng, validate_inputs


@dataclass(frozen=True)
class GBMParams:
    """Annualised GBM parameters.

    Attributes:
        mu: Arithmetic drift, the mu in ``E[S_T] = S_0 exp(mu T)``.
        sigma: Volatility.
        log_drift: The raw annualised mean log return, ``m = mu - sigma^2/2``.
            Kept so callers can see both without re-deriving either.
    """

    mu: float
    sigma: float
    log_drift: float


def estimate_parameters(prices: pd.Series | np.ndarray) -> GBMParams:
    """Estimate annualised GBM parameters from a price series.

    Uses **log** returns, then applies the Ito correction to recover the
    arithmetic drift::

        m     = mean(log returns) * 252      # estimates mu - sigma^2/2
        sigma = std(log returns)  * sqrt(252)
        mu    = m + sigma^2 / 2

    The legacy code applied the ``+ sigma^2/2`` correction on top of *simple*
    returns (``pct_change``), which double-counts drift: simple returns already
    average roughly ``mu``, not ``mu - sigma^2/2``.

    Args:
        prices: Positive price series, chronologically ordered.

    Returns:
        Estimated :class:`GBMParams`.

    Raises:
        ValueError: If fewer than two usable prices, or any price <= 0.
    """
    values = np.asarray(
        prices.to_numpy() if isinstance(prices, pd.Series) else prices, dtype=float
    ).ravel()
    values = values[np.isfinite(values)]

    if values.size < 2:
        raise ValueError(f"Need at least 2 prices to estimate parameters, got {values.size}")
    if np.any(values <= 0):
        raise ValueError("Prices must be strictly positive to take log returns.")

    log_returns = np.diff(np.log(values))

    sigma = float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    log_drift = float(log_returns.mean() * TRADING_DAYS)
    mu = log_drift + 0.5 * sigma**2

    return GBMParams(mu=mu, sigma=sigma, log_drift=log_drift)


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Simulate GBM paths with the exact log-normal scheme.

    Each step is::

        log S_{t+1} = log S_t + (mu - sigma^2/2) dt + sigma sqrt(dt) Z

    This is exact for GBM (no discretisation bias at any ``dt``) and keeps
    prices strictly positive. The legacy baseline instead used additive Euler
    without the Ito term, which is both biased and able to go negative.

    Args:
        s0: Initial price, must be positive.
        mu: Annualised arithmetic drift.
        sigma: Annualised volatility, must be non-negative.
        cfg: Simulation settings; defaults to :class:`SimConfig`.

    Returns:
        Array of shape ``(n_paths, steps + 1)``; column 0 is ``s0``.
    """
    cfg = cfg or SimConfig()
    validate_inputs(s0, sigma)

    rng = make_rng(cfg)
    dt = cfg.dt

    z = rng.standard_normal((cfg.n_paths, cfg.steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    return exponentiate_log_paths(s0, log_increments)

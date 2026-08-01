"""Merton jump-diffusion model.

Price dynamics combine a diffusion with a compound Poisson jump term::

    dS_t / S_t = (mu - lambda k) dt + sigma dW_t + (Y - 1) dN_t

where ``N_t`` is Poisson with intensity ``lambda``, jump multipliers are
lognormal ``ln Y ~ N(mu_j, sigma_j^2)``, and::

    k = E[Y] - 1 = exp(mu_j + sigma_j^2 / 2) - 1

The ``- lambda k dt`` **compensator** is what makes ``mu`` the true expected
return. The legacy implementation omitted it entirely, so under the
risk-neutral setting (``mu = r``) the discounted price was not a martingale and
every jump-diffusion option price was biased.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from .base import exponentiate_log_paths, make_rng, validate_inputs


def expected_jump_multiplier(mu_jump: float, sigma_jump: float) -> float:
    """Return ``k = E[Y] - 1`` for lognormal jump multipliers."""
    return float(np.exp(mu_jump + 0.5 * sigma_jump**2) - 1.0)


def draw_jump_counts(
    lambda_jump: float,
    dt: float,
    size: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw Poisson jump counts per interval.

    The legacy code drew a Bernoulli with probability ``1 - exp(-lambda dt)``,
    capping each interval at a single jump. That understates clustering and
    caps the tail whenever ``lambda dt`` is not small.
    """
    if lambda_jump < 0:
        raise ValueError(f"lambda_jump must be >= 0, got {lambda_jump}")
    return rng.poisson(lam=lambda_jump * dt, size=size)


def simulate_merton_jump(
    s0: float,
    mu: float,
    sigma: float,
    lambda_jump: float,
    mu_jump: float,
    sigma_jump: float,
    cfg: SimConfig | None = None,
    return_jump_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate Merton jump-diffusion paths.

    Args:
        s0: Initial price.
        mu: Annualised expected return, *after* jump compensation. Passing the
            risk-free rate here yields a risk-neutral process.
        sigma: Annualised diffusive volatility.
        lambda_jump: Poisson jump intensity, in jumps per year.
        mu_jump: Mean of the log jump size. Negative models crash risk.
        sigma_jump: Standard deviation of the log jump size.
        cfg: Simulation settings.
        return_jump_counts: Also return the per-step jump counts.

    Returns:
        Price paths of shape ``(n_paths, steps + 1)``, or a
        ``(prices, jump_counts)`` tuple.
    """
    cfg = cfg or SimConfig()
    validate_inputs(s0, sigma)

    if sigma_jump < 0:
        raise ValueError(f"sigma_jump must be >= 0, got {sigma_jump}")

    rng = make_rng(cfg)
    dt = cfg.dt
    n, steps = cfg.n_paths, cfg.steps

    k = expected_jump_multiplier(mu_jump, sigma_jump)

    # Diffusion part, compensated so that E[S_T] = S0 * exp(mu * T).
    z = rng.standard_normal((n, steps))
    log_increments = (mu - lambda_jump * k - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    counts = draw_jump_counts(lambda_jump, dt, (n, steps), rng)

    # Sum of `counts` iid N(mu_j, sigma_j^2) draws is N(counts*mu_j,
    # counts*sigma_j^2) -- so the whole compound sum is one vectorised draw,
    # rather than the legacy per-step loop that sampled n_paths lognormals
    # regardless of how few paths actually jumped. Jump sizes are NOT clamped:
    # the legacy [0.1, 10] clip truncated the tail the model exists to capture.
    if lambda_jump > 0:
        jump_noise = rng.standard_normal((n, steps))
        log_increments += counts * mu_jump + np.sqrt(counts) * sigma_jump * jump_noise

    prices = exponentiate_log_paths(s0, log_increments)

    if return_jump_counts:
        return prices, counts
    return prices

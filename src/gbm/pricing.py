"""European option pricing: Black-Scholes, Greeks, Monte Carlo, implied vol.

Conventions, stated once and applied everywhere:

* ``vega`` is per 1.00 change in volatility (divide by 100 for "per vol point").
* ``theta`` is per **calendar day**, and is negative when time decay hurts the
  holder. The legacy code returned a per-year figure and printed it next to
  per-share dollar amounts without saying so.
* ``rho`` is per 1.00 change in the rate. The legacy code never computed it.
* Degenerate inputs (``T = 0`` or ``sigma = 0``) return the exact limiting
  value rather than a guarded approximation, so price and Greeks agree. The
  legacy code applied an epsilon guard to ``d1``/``d2`` but not to the discount
  factor, leaving the two mutually inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "Greeks",
    "MonteCarloResult",
    "black_scholes_price",
    "greeks",
    "implied_volatility",
    "monte_carlo_price",
]

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Greeks:
    """Option sensitivities. See module docstring for units."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo price with its sampling uncertainty."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int

    @property
    def relative_error(self) -> float:
        return self.std_error / self.price if self.price > 0 else float("inf")


def _validate(s, k, t, r, sigma, option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if np.any(np.asarray(s, dtype=float) <= 0):
        raise ValueError("Spot price s must be positive.")
    if np.any(np.asarray(k, dtype=float) <= 0):
        raise ValueError("Strike k must be positive.")
    if np.any(np.asarray(t, dtype=float) < 0):
        raise ValueError("Maturity t must be non-negative.")
    if np.any(np.asarray(sigma, dtype=float) < 0):
        raise ValueError("Volatility sigma must be non-negative.")
    if not np.all(np.isfinite(np.asarray(r, dtype=float))):
        raise ValueError("Rate r must be finite.")


def _d1_d2(s, k, t, r, sigma):
    """Return ``(d1, d2)``. Callers must exclude the degenerate branch first."""
    vol_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _intrinsic(s, k, t, r, option_type: str):
    """Discounted intrinsic value: the exact limit as sigma or t goes to 0."""
    forward_intrinsic = s - k * np.exp(-r * t)
    if option_type == "call":
        return np.maximum(forward_intrinsic, 0.0)
    return np.maximum(-forward_intrinsic, 0.0)


def black_scholes_price(s, k, t, r, sigma, option_type: str = "call"):
    """Price a European option with the Black-Scholes formula.

    Args:
        s: Spot price. Scalar or array.
        k: Strike price.
        t: Time to maturity in years. ``0`` returns intrinsic value.
        r: Continuously compounded risk-free rate.
        sigma: Annualised volatility. ``0`` returns discounted intrinsic value.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        Option price, matching the broadcast shape of the inputs.
    """
    _validate(s, k, t, r, sigma, option_type)

    s, k, t, r, sigma = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (s, k, t, r, sigma))
    )

    degenerate = (t <= 0) | (sigma <= 0)
    # Placeholder values keep the vectorised d1/d2 computation finite; the
    # degenerate entries are overwritten with the exact limit below.
    safe_t = np.where(degenerate, 1.0, t)
    safe_sigma = np.where(degenerate, 1.0, sigma)

    d1, d2 = _d1_d2(s, k, safe_t, r, safe_sigma)
    discount = np.exp(-r * t)

    if option_type == "call":
        price = s * norm.cdf(d1) - k * discount * norm.cdf(d2)
    else:
        price = k * discount * norm.cdf(-d2) - s * norm.cdf(-d1)

    price = np.where(degenerate, _intrinsic(s, k, t, r, option_type), price)

    return float(price) if price.ndim == 0 else price


def greeks(s, k, t, r, sigma, option_type: str = "call") -> Greeks:
    """Compute the five first- and second-order Greeks.

    Args:
        s, k, t, r, sigma: As for :func:`black_scholes_price`. Scalars only.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        :class:`Greeks`. Units are documented on the module.
    """
    _validate(s, k, t, r, sigma, option_type)
    s, k, t, r, sigma = (float(x) for x in (s, k, t, r, sigma))

    if t <= 0 or sigma <= 0:
        # At expiry the option is its intrinsic value: delta is a step, and the
        # remaining Greeks vanish. Returning finite values keeps report code
        # from having to special-case the boundary.
        in_the_money = (s > k) if option_type == "call" else (s < k)
        delta = (1.0 if option_type == "call" else -1.0) if in_the_money else 0.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    discount = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(t)

    gamma = pdf_d1 / (s * sigma * sqrt_t)
    vega = s * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = norm.cdf(d1)
        theta_per_year = -(s * pdf_d1 * sigma) / (2 * sqrt_t) - r * k * discount * norm.cdf(d2)
        rho = k * t * discount * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta_per_year = -(s * pdf_d1 * sigma) / (2 * sqrt_t) + r * k * discount * norm.cdf(-d2)
        rho = -k * t * discount * norm.cdf(-d2)

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta_per_year / DAYS_PER_YEAR),
        rho=float(rho),
    )


def monte_carlo_price(
    paths: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    confidence: float = 0.95,
) -> MonteCarloResult:
    """Price a European option from simulated terminal prices.

    Args:
        paths: Price paths, shape ``(n_paths, n_steps + 1)``. Only the final
            column is used. For the result to be a valid risk-neutral price the
            paths must have been simulated with drift equal to ``rate``.
        strike: Strike price.
        maturity: Time to maturity in years, used for discounting.
        rate: Continuously compounded risk-free rate.
        option_type: ``"call"`` or ``"put"``.
        confidence: Confidence level for the reported interval.

    Returns:
        :class:`MonteCarloResult` carrying the price and its standard error.
        The legacy GPU variant omitted the interval entirely, so its output
        could not be compared against the analytic price on equal terms.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D (n_paths, n_steps+1), got shape {paths.shape}")

    terminal = paths[:, -1]
    payoffs = (
        np.maximum(terminal - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - terminal, 0.0)
    )
    discounted = np.exp(-rate * maturity) * payoffs

    n = discounted.size
    price = float(discounted.mean())
    # ddof=1 everywhere. The legacy CPU and GPU paths disagreed (0 vs 1), so
    # their "Std Error" columns were not the same estimator.
    std_error = float(discounted.std(ddof=1) / np.sqrt(n))

    z = float(norm.ppf(0.5 + confidence / 2.0))
    return MonteCarloResult(
        price=price,
        std_error=std_error,
        ci_low=price - z * std_error,
        ci_high=price + z * std_error,
        n_paths=n,
    )


def implied_volatility(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    option_type: str = "call",
    bounds: tuple[float, float] = (1e-6, 10.0),
) -> float:
    """Invert Black-Scholes for volatility via Brent's method.

    Args:
        price: Observed option price.
        s, k, t, r: Spot, strike, maturity in years, risk-free rate.
        option_type: ``"call"`` or ``"put"``.
        bounds: Search bracket for volatility.

    Returns:
        The implied volatility.

    Raises:
        ValueError: If ``price`` violates the no-arbitrage bounds, or lies
            outside the range reachable within ``bounds``.
    """
    _validate(s, k, t, r, 0.0, option_type)
    if price < 0:
        raise ValueError(f"Option price must be non-negative, got {price}")
    if t <= 0:
        raise ValueError("Cannot infer volatility at zero maturity.")

    lower = _intrinsic(s, k, t, r, option_type)
    upper = s if option_type == "call" else k * np.exp(-r * t)

    if not lower <= price <= upper:
        raise ValueError(
            f"Price {price:.6f} violates no-arbitrage bounds "
            f"[{float(lower):.6f}, {float(upper):.6f}] for this {option_type}."
        )

    def objective(vol: float) -> float:
        return black_scholes_price(s, k, t, r, vol, option_type) - price

    lo, hi = bounds
    if objective(lo) > 0 or objective(hi) < 0:
        raise ValueError(
            f"Implied volatility for price {price:.6f} lies outside the search "
            f"bracket {bounds}."
        )

    return float(brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12))

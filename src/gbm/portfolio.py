"""Multi-asset portfolio simulation with an option overlay.

The legacy implementation deposited option payoffs into
``options_impact[:, expiration_step]`` but read the portfolio value from
``[:, -1]``, so the payoff column was never read. As a result the
"with options" and "portfolio only" risk metrics were always identical and the
entire options-contribution report was structurally zero.

Here a payoff realised at expiry is **carried forward to the horizon** at the
risk-free rate, so it reaches the terminal value it is measured at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RISK_FREE_RATE, SimConfig
from .logging import get_logger
from .processes.base import make_rng, validate_inputs
from .risk import RiskMetrics, terminal_returns

log = get_logger(__name__)

__all__ = [
    "OptionPosition",
    "PortfolioResult",
    "analyse_portfolio",
    "simulate_correlated_gbm",
]


@dataclass(frozen=True)
class OptionPosition:
    """An option overlay on one holding.

    Attributes:
        asset: Name of the underlying, must match a portfolio holding.
        strike: Strike price.
        maturity_years: Time to expiry. Must not exceed the sim horizon.
        option_type: ``"call"`` or ``"put"``.
        quantity: Number of contracts. Negative sells (writes) the option.
        contract_size: Shares per contract.
    """

    asset: str
    strike: float
    maturity_years: float
    option_type: str = "put"
    quantity: float = 1.0
    contract_size: float = 100.0

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.maturity_years <= 0:
            raise ValueError(f"maturity_years must be > 0, got {self.maturity_years}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")


def simulate_correlated_gbm(
    spots: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    correlation: np.ndarray,
    cfg: SimConfig | None = None,
) -> np.ndarray:
    """Simulate correlated GBM paths for several assets.

    Args:
        spots: Initial prices, shape ``(n_assets,)``.
        mus: Annualised drifts, shape ``(n_assets,)``.
        sigmas: Annualised volatilities, shape ``(n_assets,)``.
        correlation: Symmetric positive semi-definite correlation matrix.
        cfg: Simulation settings.

    Returns:
        Array of shape ``(n_paths, n_assets, steps + 1)``.
    """
    cfg = cfg or SimConfig()
    spots = np.asarray(spots, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    correlation = np.asarray(correlation, dtype=float)

    n_assets = spots.size
    for name, arr in (("mus", mus), ("sigmas", sigmas)):
        if arr.size != n_assets:
            raise ValueError(f"{name} must have {n_assets} entries, got {arr.size}")
    if correlation.shape != (n_assets, n_assets):
        raise ValueError(
            f"correlation must be {n_assets}x{n_assets}, got {correlation.shape}"
        )
    if not np.allclose(correlation, correlation.T):
        raise ValueError("correlation matrix must be symmetric")

    for spot, sigma in zip(spots, sigmas, strict=True):
        validate_inputs(float(spot), float(sigma))

    # Cholesky needs positive definiteness; nudge the diagonal if the supplied
    # matrix is merely semi-definite rather than failing outright.
    try:
        chol = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        eigenvalues = np.linalg.eigvalsh(correlation)
        jitter = abs(min(eigenvalues.min(), 0.0)) + 1e-10
        log.debug("Correlation matrix not positive definite; adding %.2e jitter", jitter)
        chol = np.linalg.cholesky(correlation + jitter * np.eye(n_assets))

    rng = make_rng(cfg)
    dt = cfg.dt

    z = rng.standard_normal((cfg.n_paths, cfg.steps, n_assets))
    correlated = z @ chol.T

    drift = (mus - 0.5 * sigmas**2) * dt
    log_increments = drift + sigmas * np.sqrt(dt) * correlated

    log_paths = np.concatenate(
        [np.zeros((cfg.n_paths, 1, n_assets)), np.cumsum(log_increments, axis=1)], axis=1
    )
    paths = spots * np.exp(log_paths)

    # (paths, steps+1, assets) -> (paths, assets, steps+1)
    return np.ascontiguousarray(paths.transpose(0, 2, 1))


@dataclass(frozen=True)
class PortfolioResult:
    """Portfolio outcomes with and without the option overlay."""

    equity_values: np.ndarray
    total_values: np.ndarray
    equity_metrics: RiskMetrics
    total_metrics: RiskMetrics
    option_payoff_pv: np.ndarray
    asset_names: tuple[str, ...]

    @property
    def var_reduction(self) -> float:
        """Improvement in 5% VaR from the overlay.

        Positive means the overlay reduced risk. Computed on the
        losses-negative convention, so a less-negative VaR is an improvement.
        """
        return self.total_metrics.var_5 - self.equity_metrics.var_5

    @property
    def cvar_reduction(self) -> float:
        """Improvement in 5% CVaR. Positive means risk reduced."""
        return self.total_metrics.cvar_5 - self.equity_metrics.cvar_5

    @property
    def mean_option_payoff(self) -> float:
        return float(self.option_payoff_pv.mean())


def _option_payoff(paths: np.ndarray, position: OptionPosition, step: int) -> np.ndarray:
    """Payoff per path at the expiry step, signed by quantity."""
    price_at_expiry = paths[:, step]

    if position.option_type == "call":
        intrinsic = np.maximum(price_at_expiry - position.strike, 0.0)
    else:
        intrinsic = np.maximum(position.strike - price_at_expiry, 0.0)

    return position.quantity * position.contract_size * intrinsic


def analyse_portfolio(
    holdings: dict[str, float],
    spots: dict[str, float],
    mus: dict[str, float],
    sigmas: dict[str, float],
    correlation: np.ndarray,
    options: list[OptionPosition] | None = None,
    cfg: SimConfig | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> PortfolioResult:
    """Simulate a portfolio with an optional option overlay.

    Args:
        holdings: Shares held per asset.
        spots: Current price per asset.
        mus: Annualised drift per asset.
        sigmas: Annualised volatility per asset.
        correlation: Correlation matrix, ordered as ``sorted(holdings)``.
        options: Option positions overlaid on the holdings.
        cfg: Simulation settings.
        risk_free_rate: Rate at which option proceeds accrue from expiry to
            the horizon.

    Returns:
        A :class:`PortfolioResult`. ``total_metrics`` genuinely differs from
        ``equity_metrics`` whenever an overlay is present.

    Raises:
        ValueError: On unknown assets, missing parameters, or an option
            expiring after the simulation horizon.
    """
    cfg = cfg or SimConfig()
    options = options or []

    # Deterministic ordering: the legacy code took the correlation matrix from
    # one asset's dict and assumed every other asset's row order agreed.
    names = tuple(sorted(holdings))
    if not names:
        raise ValueError("holdings must contain at least one asset")

    for mapping, label in ((spots, "spots"), (mus, "mus"), (sigmas, "sigmas")):
        missing = [n for n in names if n not in mapping]
        if missing:
            raise ValueError(f"{label} is missing entries for {missing}")

    for position in options:
        if position.asset not in holdings:
            raise ValueError(
                f"Option references unknown asset {position.asset!r}; "
                f"portfolio holds {list(names)}"
            )
        if position.maturity_years > cfg.horizon_years + 1e-12:
            raise ValueError(
                f"Option on {position.asset!r} expires at {position.maturity_years}y, "
                f"after the {cfg.horizon_years}y simulation horizon."
            )

    paths = simulate_correlated_gbm(
        spots=np.array([spots[n] for n in names]),
        mus=np.array([mus[n] for n in names]),
        sigmas=np.array([sigmas[n] for n in names]),
        correlation=correlation,
        cfg=cfg,
    )

    share_counts = np.array([holdings[n] for n in names])
    equity_values = np.einsum("pat,a->pt", paths, share_counts)

    option_pv = np.zeros(cfg.n_paths)
    for position in options:
        asset_index = names.index(position.asset)
        step = min(round(position.maturity_years / cfg.dt), cfg.steps)

        payoff = _option_payoff(paths[:, asset_index, :], position, step)

        # Carry the payoff from expiry to the horizon at the risk-free rate,
        # so it actually reaches the terminal value the metrics are read from.
        remaining = cfg.horizon_years - step * cfg.dt
        option_pv += payoff * np.exp(risk_free_rate * remaining)

    total_values = equity_values.copy()
    total_values[:, -1] += option_pv

    return PortfolioResult(
        equity_values=equity_values,
        total_values=total_values,
        equity_metrics=RiskMetrics.from_paths(
            equity_values, cfg.horizon_years, risk_free_rate
        ),
        total_metrics=RiskMetrics.from_paths(
            total_values, cfg.horizon_years, risk_free_rate
        ),
        option_payoff_pv=option_pv,
        asset_names=names,
    )


def portfolio_terminal_returns(result: PortfolioResult) -> np.ndarray:
    """Terminal returns of the option-overlaid portfolio."""
    return terminal_returns(result.total_values)

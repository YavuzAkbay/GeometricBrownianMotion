"""Risk metrics.

**Sign convention, applied everywhere without exception: losses are negative.**

A VaR of ``-0.18`` means "the 5% worst outcomes lose at least 18%". A *lower*
number is therefore *worse*. Because that is easy to get backwards when
sorting, comparisons must go through :func:`rank_by_risk` rather than a raw
``min``/``max``/``sorted`` — the legacy code got this wrong in three places and
reported its riskiest model as the best.

Drawdown is always computed **along each path** and then aggregated across
paths. The legacy code had three incompatible definitions, none of which
measured a peak-to-trough decline of a price series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import DEFAULT_RISK_FREE_RATE, RiskConfig

__all__ = [
    "RiskMetrics",
    "VaRResult",
    "information_ratio",
    "max_drawdown",
    "path_max_drawdown",
    "rank_by_risk",
    "sharpe_ratio",
    "terminal_returns",
    "value_at_risk",
]


@dataclass(frozen=True)
class VaRResult:
    """Value at Risk and Conditional VaR at one tail probability.

    Both are returns, negative for losses. ``cvar <= var`` always holds.
    """

    alpha: float
    var: float
    cvar: float


def _as_returns(returns: np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute risk metrics from an empty return series.")
    return arr


def _is_degenerate(std: float, values: np.ndarray) -> bool:
    """True if dispersion is indistinguishable from zero at this scale.

    An exact ``std == 0`` test is not enough: the sample standard deviation of
    a constant array is ~1e-17 rather than 0, which would otherwise divide
    through to a Sharpe ratio of ~1e15.
    """
    scale = max(float(np.abs(values).max()), 1.0)
    return std <= 1e-12 * scale


def terminal_returns(paths: np.ndarray) -> np.ndarray:
    """Return each path's total return from its own starting price."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D (n_paths, n_steps+1), got shape {paths.shape}")
    return paths[:, -1] / paths[:, 0] - 1.0


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> VaRResult:
    """Historical VaR and CVaR at tail probability ``alpha``.

    Args:
        returns: Realised or simulated returns.
        alpha: Tail probability in ``(0, 1)``. ``0.05`` is the 5% worst cases.

    Returns:
        :class:`VaRResult` with both figures on the losses-negative convention.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    arr = _as_returns(returns)
    var = float(np.quantile(arr, alpha))

    tail = arr[arr <= var]
    # With few samples the tail can be empty at small alpha; VaR is then the
    # best available estimate of the conditional loss.
    cvar = float(tail.mean()) if tail.size else var

    return VaRResult(alpha=alpha, var=var, cvar=cvar)


def rank_by_risk(candidates: dict[str, float]) -> list[str]:
    """Order candidates from least to most risky by a signed risk figure.

    Args:
        candidates: Mapping of name to a signed VaR or CVaR (losses negative).

    Returns:
        Names ordered safest first. Since losses are negative, "safest" is the
        *largest* value — the inversion the legacy ranking got wrong.
    """
    return [name for name, _ in sorted(candidates.items(), key=lambda kv: -kv[1])]


def path_max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough decline along each path.

    Args:
        paths: Price paths, shape ``(n_paths, n_steps + 1)``.

    Returns:
        One drawdown per path, in ``[-1, 0]``. ``-0.5`` is a 50% decline from a
        running peak.
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError(f"paths must be 2-D (n_paths, n_steps+1), got shape {paths.shape}")

    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdown = paths / running_peak - 1.0
    return drawdown.min(axis=1)


def max_drawdown(paths: np.ndarray, aggregate: str = "mean") -> float:
    """Aggregate the per-path drawdowns into a single figure.

    Args:
        paths: Price paths.
        aggregate: ``"mean"`` for the typical path, ``"median"``, or ``"qXX"``
            for a quantile (``"q05"`` is the 5% worst-case drawdown).
            ``"worst"`` is available but converges to -100% as paths grow, so
            it discriminates poorly between models — it was the legacy default.

    Returns:
        A single drawdown in ``[-1, 0]``.
    """
    per_path = path_max_drawdown(paths)

    if aggregate == "mean":
        return float(per_path.mean())
    if aggregate == "median":
        return float(np.median(per_path))
    if aggregate == "worst":
        return float(per_path.min())
    if aggregate.startswith("q"):
        pct = float(aggregate[1:])
        return float(np.quantile(per_path, pct / 100.0))

    raise ValueError(
        f"Unknown aggregate {aggregate!r}; expected mean, median, worst or qXX."
    )


def sharpe_ratio(
    returns: np.ndarray,
    horizon_years: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Annualised Sharpe ratio of horizon returns.

    The risk-free rate is compounded to the horizon before being deducted, and
    the ratio is annualised by ``sqrt(1 / horizon_years)``::

        rf_period = (1 + r)^T - 1
        sharpe    = sqrt(1/T) * (mean(R) - rf_period) / std(R)

    The legacy version omitted both steps, calling ``mean/std`` the Sharpe
    ratio — that is the reward-to-variability ratio, and it drove the automated
    "best model" selection.

    Args:
        returns: Total returns over the horizon, one per path.
        horizon_years: Horizon T in years.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        The annualised Sharpe ratio, or ``0.0`` when returns have no dispersion.
    """
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be > 0, got {horizon_years}")

    arr = _as_returns(returns)
    std = float(arr.std(ddof=1))
    if _is_degenerate(std, arr):
        return 0.0

    rf_period = (1.0 + risk_free_rate) ** horizon_years - 1.0
    return float(np.sqrt(1.0 / horizon_years) * (arr.mean() - rf_period) / std)


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    horizon_years: float,
) -> float:
    """Annualised information ratio against a benchmark.

    Uses the standard deviation of the *active* return (strategy minus
    benchmark) as the tracking error. The legacy code set tracking error equal
    to the strategy's own standard deviation, which made this identical to the
    Sharpe ratio while being reported as a separate statistic.

    Args:
        returns: Strategy returns over the horizon.
        benchmark_returns: Benchmark returns over the same horizon.
        horizon_years: Horizon T in years.

    Returns:
        The annualised information ratio, or ``0.0`` if tracking error is zero.
    """
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be > 0, got {horizon_years}")

    strategy = _as_returns(returns)
    benchmark = _as_returns(benchmark_returns)

    # Paired differencing needs equal lengths; otherwise compare distributions.
    if strategy.size == benchmark.size:
        active = strategy - benchmark
        tracking_error = float(active.std(ddof=1))
        mean_active = float(active.mean())
    else:
        mean_active = float(strategy.mean() - benchmark.mean())
        tracking_error = float(np.sqrt(strategy.var(ddof=1) + benchmark.var(ddof=1)))

    if _is_degenerate(tracking_error, np.concatenate([strategy, benchmark])):
        return 0.0

    return float(np.sqrt(1.0 / horizon_years) * mean_active / tracking_error)


@dataclass(frozen=True)
class RiskMetrics:
    """The standard risk summary for a set of simulated paths.

    Field names are the single documented contract. The legacy CPU and GPU
    implementations disagreed (``mean`` vs ``mean_return``), so passing one
    dict to the other's consumer raised ``KeyError``.
    """

    mean_return: float
    volatility: float
    sharpe_ratio: float
    var_1: float
    var_5: float
    cvar_1: float
    cvar_5: float
    max_drawdown: float
    profit_probability: float
    tail_risk: float
    skewness: float
    excess_kurtosis: float
    n_paths: int

    @classmethod
    def from_paths(
        cls,
        paths: np.ndarray,
        horizon_years: float,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        drawdown_aggregate: str = "mean",
    ) -> RiskMetrics:
        """Compute every metric from simulated price paths.

        Args:
            paths: Price paths, shape ``(n_paths, n_steps + 1)``.
            horizon_years: Horizon T in years, used for annualisation.
            risk_free_rate: Annualised risk-free rate.
            drawdown_aggregate: Passed to :func:`max_drawdown`.
        """
        returns = terminal_returns(paths)
        arr = _as_returns(returns)

        var_1 = value_at_risk(arr, alpha=0.01)
        var_5 = value_at_risk(arr, alpha=0.05)

        mean = float(arr.mean())
        std = float(arr.std(ddof=1))

        if _is_degenerate(std, arr):
            skewness = 0.0
            excess_kurtosis = 0.0
        else:
            standardised = (arr - mean) / std
            skewness = float((standardised**3).mean())
            excess_kurtosis = float((standardised**4).mean() - 3.0)

        return cls(
            mean_return=mean,
            volatility=std,
            sharpe_ratio=sharpe_ratio(arr, horizon_years, risk_free_rate),
            var_1=var_1.var,
            var_5=var_5.var,
            cvar_1=var_1.cvar,
            cvar_5=var_5.cvar,
            max_drawdown=max_drawdown(paths, aggregate=drawdown_aggregate),
            profit_probability=float((arr > 0).mean()),
            # A genuine probability: how often the 1% tail threshold is
            # breached. The legacy field of this name was a duplicate of cvar_1.
            tail_risk=float((arr <= var_1.var).mean()),
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            n_paths=int(arr.size),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dict, keys as documented above."""
        return asdict(self)

    def summary_lines(self, config: RiskConfig | None = None) -> list[str]:
        """Human-readable lines for reports. Percentages where natural."""
        del config  # reserved for per-report confidence levels
        return [
            f"Expected return:     {self.mean_return:+.2%}",
            f"Volatility:          {self.volatility:.2%}",
            f"Sharpe ratio:        {self.sharpe_ratio:.3f}  (annualised, excess of rf)",
            f"VaR (5%):            {self.var_5:.2%}",
            f"CVaR (5%):           {self.cvar_5:.2%}",
            f"VaR (1%):            {self.var_1:.2%}",
            f"CVaR (1%):           {self.cvar_1:.2%}",
            f"Max drawdown:        {self.max_drawdown:.2%}  (mean over paths)",
            f"Probability of gain: {self.profit_probability:.1%}",
            f"Skewness:            {self.skewness:+.3f}",
            f"Excess kurtosis:     {self.excess_kurtosis:+.3f}",
        ]

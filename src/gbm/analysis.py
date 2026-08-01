"""High-level analysis orchestration.

Composes data, processes, pricing and risk into the workflows the CLI exposes.
These replace the legacy 337-line ``comprehensive_quantitative_analysis`` and
its 568-line demo sibling: each function here does one thing and returns data,
so it can be tested and reused rather than only printed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_RISK_FREE_RATE, SimConfig
from .data import fetch
from .logging import get_logger
from .processes import (
    DEFAULT_TRANSITION_MATRIX,
    GBMParams,
    estimate_parameters,
    simulate_gbm,
    simulate_heston,
    simulate_merton_jump,
    simulate_regime_switching,
)
from .risk import RiskMetrics, rank_by_risk

log = get_logger(__name__)

__all__ = [
    "MODEL_NAMES",
    "AnalysisResult",
    "ComparisonResult",
    "analyse",
    "compare_models",
    "simulate_model",
]

MODEL_NAMES = ("gbm", "heston", "regime", "jump")


@dataclass(frozen=True)
class AnalysisResult:
    """One model's simulation and its risk summary."""

    ticker: str
    model: str
    spot: float
    params: GBMParams
    paths: np.ndarray
    metrics: RiskMetrics
    config: SimConfig

    @property
    def terminal_prices(self) -> np.ndarray:
        return self.paths[:, -1]

    def summary(self) -> str:
        lines = [
            f"{self.ticker} - {self.model} model",
            f"Spot price:          ${self.spot:,.2f}",
            f"Estimated drift:     {self.params.mu:+.2%} annualised",
            f"Estimated vol:       {self.params.sigma:.2%} annualised",
            f"Horizon:             {self.config.horizon_years:.2f} years "
            f"({self.config.steps} steps, {self.config.n_paths:,} paths)",
            "",
            *self.metrics.summary_lines(),
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ComparisonResult:
    """Several models simulated on the same underlying."""

    ticker: str
    results: dict[str, AnalysisResult]

    def ranked_by_risk(self) -> list[str]:
        """Model names ordered safest first, by 5% CVaR.

        Goes through :func:`gbm.risk.rank_by_risk`, so the losses-negative
        convention is handled correctly. The legacy ranking sorted signed CVaR
        ascending and reported the riskiest model as rank 1.
        """
        return rank_by_risk({n: r.metrics.cvar_5 for n, r in self.results.items()})

    def summary(self) -> str:
        header = (
            f"{'Model':<10}{'Mean':>10}{'Vol':>10}{'VaR 5%':>10}"
            f"{'CVaR 5%':>10}{'Sharpe':>9}{'MaxDD':>10}"
        )
        rows = [f"Model comparison for {self.ticker}", "", header, "-" * len(header)]

        for name in self.ranked_by_risk():
            m = self.results[name].metrics
            rows.append(
                f"{name:<10}{m.mean_return:>9.2%}{m.volatility:>10.2%}"
                f"{m.var_5:>10.2%}{m.cvar_5:>10.2%}{m.sharpe_ratio:>9.3f}"
                f"{m.max_drawdown:>10.2%}"
            )

        rows += ["", f"Least risky by CVaR 5%: {self.ranked_by_risk()[0]}"]
        return "\n".join(rows)


def simulate_model(
    model: str,
    spot: float,
    params: GBMParams,
    cfg: SimConfig,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> np.ndarray:
    """Simulate one model, deriving its parameters from the fitted GBM ones.

    Args:
        model: One of :data:`MODEL_NAMES`.
        spot: Current price.
        params: Fitted GBM parameters.
        cfg: Simulation settings.
        risk_free_rate: Used only to set regime drifts relative to the
            risk-free rate.

    Returns:
        Price paths of shape ``(n_paths, steps + 1)``.
    """
    if model == "gbm":
        return simulate_gbm(spot, params.mu, params.sigma, cfg)

    if model == "heston":
        variance = params.sigma**2
        # kappa chosen so the Feller condition holds for the fitted variance.
        sigma_v = 0.5 * params.sigma
        kappa = max(2.0, sigma_v**2 / (2.0 * variance) + 0.5) if variance > 0 else 2.0
        return simulate_heston(
            spot, mu=params.mu, v0=variance, kappa=kappa, theta=variance,
            sigma_v=sigma_v, rho=-0.7, cfg=cfg,
        )

    if model == "regime":
        # Bull / bear / crisis: drifts spread around the fitted drift, vols
        # scaled off the fitted vol.
        mu_states = np.array([params.mu, params.mu - 0.10, params.mu - 0.25])
        sigma_states = np.array(
            [params.sigma * 0.8, params.sigma * 1.3, params.sigma * 2.0]
        )
        return simulate_regime_switching(
            spot, mu_states, sigma_states, DEFAULT_TRANSITION_MATRIX, cfg
        )

    if model == "jump":
        # Split the fitted variance between diffusion and jumps so total
        # variance stays consistent with the estimate.
        lambda_jump = 1.0
        mu_jump, sigma_jump = -0.03, 0.10
        jump_var = lambda_jump * (mu_jump**2 + sigma_jump**2)
        diffusive_var = max(params.sigma**2 - jump_var, (0.3 * params.sigma) ** 2)
        return simulate_merton_jump(
            spot, mu=params.mu, sigma=np.sqrt(diffusive_var),
            lambda_jump=lambda_jump, mu_jump=mu_jump, sigma_jump=sigma_jump, cfg=cfg,
        )

    raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}")


def analyse(
    ticker: str,
    model: str = "gbm",
    months: int = 6,
    n_paths: int = 10_000,
    seed: int | None = 42,
    period: str = "5y",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    cache_dir=None,
) -> AnalysisResult:
    """Fetch data, fit parameters, simulate, and summarise risk.

    Args:
        ticker: Symbol to analyse.
        model: One of :data:`MODEL_NAMES`.
        months: Forecast horizon in months.
        n_paths: Monte Carlo paths.
        seed: RNG seed. ``None`` for a non-reproducible run.
        period: History window to fit on.
        risk_free_rate: Annualised risk-free rate.
        cache_dir: Data cache location; ``None`` uses the default.

    Returns:
        An :class:`AnalysisResult`.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}")

    kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
    data = fetch(ticker, period=period, **kwargs)

    params = estimate_parameters(data.close)
    spot = data.latest_price

    log.info(
        "%s: spot $%.2f, drift %+.2f%%, vol %.2f%% (from %d observations)",
        data.ticker, spot, params.mu * 100, params.sigma * 100, len(data),
    )

    cfg = SimConfig.from_months(months, n_paths=n_paths, seed=seed)
    paths = simulate_model(model, spot, params, cfg, risk_free_rate)

    return AnalysisResult(
        ticker=data.ticker,
        model=model,
        spot=spot,
        params=params,
        paths=paths,
        metrics=RiskMetrics.from_paths(paths, cfg.horizon_years, risk_free_rate),
        config=cfg,
    )


def compare_models(
    ticker: str,
    models: tuple[str, ...] = MODEL_NAMES,
    months: int = 6,
    n_paths: int = 10_000,
    seed: int | None = 42,
    period: str = "5y",
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    cache_dir=None,
) -> ComparisonResult:
    """Run several models on one ticker and rank them by risk.

    Each model gets a distinct derived seed, so their random streams are
    genuinely independent. The legacy code reseeded to 42 inside every
    simulator, which meant a model and its "independent" baseline drew
    identical normals.
    """
    results = {}
    for offset, model in enumerate(models):
        model_seed = None if seed is None else seed + offset * 1000
        results[model] = analyse(
            ticker, model=model, months=months, n_paths=n_paths,
            seed=model_seed, period=period, risk_free_rate=risk_free_rate,
            cache_dir=cache_dir,
        )

    ticker_name = next(iter(results.values())).ticker
    return ComparisonResult(ticker=ticker_name, results=results)

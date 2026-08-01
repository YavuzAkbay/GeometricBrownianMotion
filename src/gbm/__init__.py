"""Geometric Brownian Motion and advanced stochastic models for equity analysis.

Importing this package has no side effects: it creates no directories, prints
nothing, configures no logging handlers, and does not import torch.

Typical use::

    from gbm import SimConfig, simulate_gbm, estimate_parameters

    params = estimate_parameters(prices)
    paths = simulate_gbm(s0, params.mu, params.sigma, SimConfig(seed=42))
"""

from __future__ import annotations

from .config import (
    DEFAULT_RISK_FREE_RATE,
    TRADING_DAYS,
    OptionSpec,
    OutputConfig,
    RiskConfig,
    SimConfig,
)
from .pricing import (
    Greeks,
    MonteCarloResult,
    black_scholes_price,
    greeks,
    implied_volatility,
    monte_carlo_price,
)
from .processes import (
    GBMParams,
    estimate_parameters,
    simulate_gbm,
    simulate_heston,
    simulate_merton_jump,
    simulate_regime_switching,
)
from .risk import (
    RiskMetrics,
    VaRResult,
    information_ratio,
    max_drawdown,
    rank_by_risk,
    sharpe_ratio,
    terminal_returns,
    value_at_risk,
)

__version__ = "2.0.0"

__all__ = [
    "DEFAULT_RISK_FREE_RATE",
    "TRADING_DAYS",
    "GBMParams",
    "Greeks",
    "MonteCarloResult",
    "OptionSpec",
    "OutputConfig",
    "RiskConfig",
    "RiskMetrics",
    "SimConfig",
    "VaRResult",
    "__version__",
    "black_scholes_price",
    "estimate_parameters",
    "greeks",
    "implied_volatility",
    "information_ratio",
    "max_drawdown",
    "monte_carlo_price",
    "rank_by_risk",
    "sharpe_ratio",
    "simulate_gbm",
    "simulate_heston",
    "simulate_merton_jump",
    "simulate_regime_switching",
    "terminal_returns",
    "value_at_risk",
]

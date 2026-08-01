"""Risk metric tests.

The legacy code had four separate sign/convention defects here: an inverted
CVaR ranking, three mutually incompatible max-drawdown definitions, an
un-annualised Sharpe with no risk-free rate, and an information ratio that was
numerically identical to Sharpe. Each has a named test.
"""

from __future__ import annotations

import numpy as np
import pytest

from gbm.risk import (
    RiskMetrics,
    max_drawdown,
    path_max_drawdown,
    rank_by_risk,
    sharpe_ratio,
    terminal_returns,
    value_at_risk,
)

ALPHA = 0.05


@pytest.fixture
def normal_returns() -> np.ndarray:
    return np.random.default_rng(0).normal(0.08, 0.20, size=200_000)


# ---------------------------------------------------------------- VaR / CVaR


def test_var_matches_normal_quantile(normal_returns):
    """VaR is the alpha-quantile of returns: negative for a loss."""
    from scipy.stats import norm

    var = value_at_risk(normal_returns, alpha=ALPHA)
    assert var.var == pytest.approx(norm.ppf(ALPHA, loc=0.08, scale=0.20), abs=0.005)
    assert var.var < 0


def test_cvar_matches_normal_tail_expectation(normal_returns):
    from scipy.stats import norm

    var = value_at_risk(normal_returns, alpha=ALPHA)
    expected = 0.08 - 0.20 * norm.pdf(norm.ppf(ALPHA)) / ALPHA
    assert var.cvar == pytest.approx(expected, abs=0.005)


def test_cvar_never_exceeds_var(normal_returns):
    """CVaR averages the tail beyond VaR, so it is always the more negative."""
    for alpha in (0.01, 0.05, 0.10, 0.25):
        var = value_at_risk(normal_returns, alpha=alpha)
        assert var.cvar <= var.var


def test_var_is_monotone_in_alpha(normal_returns):
    """A deeper tail is a worse loss."""
    var_1 = value_at_risk(normal_returns, alpha=0.01).var
    var_5 = value_at_risk(normal_returns, alpha=0.05).var
    var_10 = value_at_risk(normal_returns, alpha=0.10).var
    assert var_1 < var_5 < var_10


def test_var_sign_convention_is_losses_negative():
    """An all-gains distribution yields a positive VaR, not a sign flip."""
    assert value_at_risk(np.full(1000, 0.05), alpha=ALPHA).var == pytest.approx(0.05)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_var_rejects_invalid_alpha(normal_returns, alpha):
    with pytest.raises(ValueError):
        value_at_risk(normal_returns, alpha=alpha)


def test_var_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        value_at_risk(np.array([]), alpha=ALPHA)


# ---------------------------------------------------------------- ranking


def test_rank_by_risk_puts_safest_first():
    """REGRESSION: legacy sorted signed CVaR ascending under the banner
    'Lower CVaR = Better', so rank 1 was the *worst* model.

    Legacy sites: gbm.py:2618-2619, gbm.py:2716, enhanced_gbm.py:3659-3661.
    """
    candidates = {"safe": -0.05, "medium": -0.15, "risky": -0.40}
    assert rank_by_risk(candidates) == ["safe", "medium", "risky"]


def test_rank_by_risk_handles_positive_values():
    assert rank_by_risk({"a": 0.02, "b": -0.01, "c": -0.30}) == ["a", "b", "c"]


def test_rank_by_risk_is_stable_for_ties():
    assert rank_by_risk({"a": -0.1, "b": -0.1, "c": -0.2}) == ["a", "b", "c"]


# ---------------------------------------------------------------- drawdown


def test_path_max_drawdown_on_known_series():
    """100 -> 120 -> 60 -> 90 has a peak-to-trough drawdown of exactly -50%."""
    path = np.array([[100.0, 120.0, 60.0, 90.0]])
    assert path_max_drawdown(path)[0] == pytest.approx(-0.5)


def test_path_max_drawdown_is_zero_for_monotone_increase():
    path = np.array([[100.0, 110.0, 120.0, 130.0]])
    assert path_max_drawdown(path)[0] == pytest.approx(0.0)


def test_path_max_drawdown_bounded_in_unit_interval():
    """REGRESSION: legacy ran np.cumprod over *independent* MC terminal returns
    (gbm.py:2533), compounding unrelated scenarios into a meaningless number.
    """
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, mu=0.05, sigma=0.3, cfg=SimConfig(n_paths=5000, seed=8))
    dd = path_max_drawdown(paths)
    assert dd.shape == (5000,)
    assert np.all(dd <= 0.0)
    assert np.all(dd >= -1.0)


def test_max_drawdown_aggregates_by_mean_not_worst_path():
    """REGRESSION: legacy took min() over 10,000 paths, which converges to
    ~-100% for every model and so cannot discriminate between them.
    """
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    calm = simulate_gbm(100.0, 0.05, 0.10, SimConfig(n_paths=20_000, seed=9))
    wild = simulate_gbm(100.0, 0.05, 0.60, SimConfig(n_paths=20_000, seed=9))

    assert max_drawdown(calm) > max_drawdown(wild)
    assert -1.0 < max_drawdown(calm) < 0.0


def test_max_drawdown_accepts_quantile_aggregation():
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, 0.05, 0.3, SimConfig(n_paths=5000, seed=10))
    typical = max_drawdown(paths, aggregate="mean")
    tail = max_drawdown(paths, aggregate="q05")
    assert tail < typical


# ---------------------------------------------------------------- Sharpe


def test_sharpe_is_zero_when_excess_return_is_zero():
    """REGRESSION: legacy omitted the risk-free rate entirely (gbm.py:2528)."""
    returns = np.random.default_rng(1).normal(0.03, 0.15, 100_000)
    assert sharpe_ratio(returns, horizon_years=1.0, risk_free_rate=0.03) == pytest.approx(
        0.0, abs=0.02
    )


def test_sharpe_matches_hand_computation():
    returns = np.random.default_rng(2).normal(0.10, 0.20, 500_000)
    expected = (returns.mean() - 0.02) / returns.std(ddof=1)
    assert sharpe_ratio(returns, horizon_years=1.0, risk_free_rate=0.02) == pytest.approx(
        expected, rel=1e-9
    )


def test_sharpe_annualises_by_inverse_sqrt_horizon():
    """A 6-month Sharpe annualises by sqrt(1/T) = sqrt(2)."""
    returns = np.random.default_rng(3).normal(0.05, 0.10, 200_000)
    half = sharpe_ratio(returns, horizon_years=0.5, risk_free_rate=0.0)
    full = sharpe_ratio(returns, horizon_years=1.0, risk_free_rate=0.0)
    assert half / full == pytest.approx(np.sqrt(2.0), rel=1e-9)


def test_sharpe_compounds_risk_free_to_horizon():
    """A 6-month horizon must deduct the 6-month rate, not the annual one."""
    returns = np.full(1000, 0.10)
    # Zero dispersion -> infinite Sharpe; use a tiny spread to keep it finite.
    returns = returns + np.random.default_rng(4).normal(0, 0.01, 1000)
    result = sharpe_ratio(returns, horizon_years=0.5, risk_free_rate=0.04)
    period_rf = (1 + 0.04) ** 0.5 - 1
    expected = np.sqrt(2.0) * (returns.mean() - period_rf) / returns.std(ddof=1)
    assert result == pytest.approx(expected, rel=1e-9)


def test_sharpe_of_constant_returns_is_zero_not_nan():
    assert sharpe_ratio(np.full(100, 0.05), horizon_years=1.0) == 0.0


# ---------------------------------------------------------------- full metrics


def test_metrics_from_paths_are_self_consistent():
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, 0.08, 0.25, SimConfig(n_paths=50_000, seed=11))
    m = RiskMetrics.from_paths(paths, horizon_years=1.0, risk_free_rate=0.03)

    assert m.var_5 < 0
    assert m.cvar_5 <= m.var_5
    assert m.cvar_1 <= m.cvar_5
    assert -1.0 < m.max_drawdown < 0.0
    assert 0.0 <= m.profit_probability <= 1.0
    assert m.volatility > 0


def test_tail_risk_is_not_a_duplicate_of_cvar():
    """REGRESSION: legacy 'tail_risk' was bit-identical to cvar_1 while being
    documented as a probability (enhanced_gbm.py:627-629, 3393-3395).
    """
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, 0.08, 0.25, SimConfig(n_paths=20_000, seed=12))
    m = RiskMetrics.from_paths(paths, horizon_years=1.0)

    assert m.tail_risk != m.cvar_1
    assert 0.0 <= m.tail_risk <= 1.0  # it is a probability


def test_information_ratio_differs_from_sharpe():
    """REGRESSION: legacy set tracking_error = std_error, making the two
    identical, so two report lines always printed the same number
    (enhanced_gbm.py:2615, printed at :1231-1232).
    """
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm
    from gbm.risk import information_ratio

    strategy = simulate_gbm(100.0, 0.12, 0.25, SimConfig(n_paths=20_000, seed=13))
    benchmark = simulate_gbm(100.0, 0.07, 0.18, SimConfig(n_paths=20_000, seed=14))

    ir = information_ratio(
        terminal_returns(strategy), terminal_returns(benchmark), horizon_years=1.0
    )
    sr = sharpe_ratio(terminal_returns(strategy), horizon_years=1.0)

    assert ir != pytest.approx(sr, rel=0.01)
    assert np.isfinite(ir)


def test_terminal_returns_are_relative_to_start():
    paths = np.array([[100.0, 110.0, 120.0], [100.0, 90.0, 50.0]])
    np.testing.assert_allclose(terminal_returns(paths), [0.2, -0.5])


def test_profit_probability_matches_manual_count():
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, 0.08, 0.25, SimConfig(n_paths=10_000, seed=15))
    m = RiskMetrics.from_paths(paths, horizon_years=1.0)
    assert m.profit_probability == pytest.approx((terminal_returns(paths) > 0).mean())


def test_metrics_dict_roundtrip_has_documented_keys():
    """Guards the CPU/GPU key divergence ('mean' vs 'mean_return') that caused
    a KeyError in legacy consumers.
    """
    from gbm.config import SimConfig
    from gbm.processes import simulate_gbm

    paths = simulate_gbm(100.0, 0.08, 0.25, SimConfig(n_paths=1000, seed=16))
    d = RiskMetrics.from_paths(paths, horizon_years=1.0).to_dict()

    for key in ("mean_return", "volatility", "var_5", "cvar_5", "sharpe_ratio",
                "max_drawdown", "profit_probability", "tail_risk"):
        assert key in d, f"missing documented key {key!r}"

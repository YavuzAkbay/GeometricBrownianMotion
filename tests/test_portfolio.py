"""Portfolio simulation and option-overlay tests."""

from __future__ import annotations

import numpy as np
import pytest

from gbm.config import SimConfig
from gbm.portfolio import OptionPosition, analyse_portfolio, simulate_correlated_gbm

HOLDINGS = {"AAPL": 100.0, "MSFT": 50.0, "GOOGL": 25.0}
SPOTS = {"AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0}
MUS = {"AAPL": 0.10, "MSFT": 0.09, "GOOGL": 0.08}
SIGMAS = {"AAPL": 0.28, "MSFT": 0.25, "GOOGL": 0.30}

CORRELATION = np.array(
    [
        [1.00, 0.62, 0.58],
        [0.62, 1.00, 0.55],
        [0.58, 0.55, 1.00],
    ]
)


@pytest.fixture
def cfg() -> SimConfig:
    return SimConfig(horizon_years=1.0, steps=252, n_paths=20_000, seed=77)


# ---------------------------------------------------------------- correlated paths


def test_correlated_paths_have_expected_shape(cfg):
    paths = simulate_correlated_gbm(
        np.array([100.0, 200.0]), np.array([0.05, 0.06]),
        np.array([0.2, 0.3]), np.eye(2), cfg=cfg,
    )
    assert paths.shape == (cfg.n_paths, 2, cfg.steps + 1)


def test_correlated_paths_reproduce_input_correlation(cfg):
    paths = simulate_correlated_gbm(
        np.array([100.0, 200.0, 150.0]),
        np.array([0.05, 0.06, 0.04]),
        np.array([0.2, 0.3, 0.25]),
        CORRELATION,
        cfg=cfg,
    )
    log_returns = np.log(paths[:, :, -1] / paths[:, :, 0])
    realised = np.corrcoef(log_returns.T)
    np.testing.assert_allclose(realised, CORRELATION, atol=0.03)


def test_each_asset_preserves_its_own_terminal_mean(cfg):
    spots = np.array([100.0, 200.0])
    mus = np.array([0.05, 0.09])
    paths = simulate_correlated_gbm(spots, mus, np.array([0.2, 0.3]), np.eye(2), cfg=cfg)

    for i, (s0, mu) in enumerate(zip(spots, mus, strict=True)):
        terminal = paths[:, i, -1]
        expected = s0 * np.exp(mu * cfg.horizon_years)
        stderr = terminal.std(ddof=1) / np.sqrt(cfg.n_paths)
        assert abs(terminal.mean() - expected) < 4 * stderr


def test_semi_definite_correlation_is_accepted(cfg):
    """A perfectly collinear pair must not blow up Cholesky."""
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    paths = simulate_correlated_gbm(
        np.array([100.0, 100.0]), np.array([0.05, 0.05]),
        np.array([0.2, 0.2]), singular, cfg=SimConfig(n_paths=500, seed=1),
    )
    assert np.all(paths > 0)


def test_asymmetric_correlation_rejected(cfg):
    with pytest.raises(ValueError, match="symmetric"):
        simulate_correlated_gbm(
            np.array([100.0, 100.0]), np.array([0.05, 0.05]),
            np.array([0.2, 0.2]), np.array([[1.0, 0.5], [0.3, 1.0]]), cfg=cfg,
        )


# ---------------------------------------------------------------- overlay effect


def test_protective_put_changes_the_risk_metrics(cfg):
    """REGRESSION: the headline bug.

    Legacy wrote payoffs to options_impact[:, expiration_step] but read the
    portfolio from [:, -1], so with-options and portfolio-only metrics were
    byte-identical and the contribution report was structurally zero.
    """
    put = OptionPosition(
        asset="AAPL", strike=170.0, maturity_years=0.5, option_type="put", quantity=1.0
    )
    result = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, options=[put], cfg=cfg
    )

    assert result.equity_metrics.var_5 != result.total_metrics.var_5
    assert result.mean_option_payoff > 0
    assert not np.array_equal(result.equity_values, result.total_values)


def test_protective_put_reduces_downside_risk(cfg):
    """A long put must improve tail risk, and the sign convention must say so."""
    put = OptionPosition(
        asset="AAPL", strike=200.0, maturity_years=1.0,
        option_type="put", quantity=1.0, contract_size=100.0,
    )
    result = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, options=[put], cfg=cfg
    )

    assert result.var_reduction > 0
    assert result.cvar_reduction > 0
    assert result.total_metrics.cvar_5 > result.equity_metrics.cvar_5


def test_written_call_caps_upside(cfg):
    """A short call must reduce the mean terminal value."""
    call = OptionPosition(
        asset="AAPL", strike=190.0, maturity_years=1.0,
        option_type="call", quantity=-1.0,
    )
    result = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, options=[call], cfg=cfg
    )

    assert result.mean_option_payoff < 0
    assert result.total_metrics.mean_return < result.equity_metrics.mean_return


def test_no_options_leaves_portfolio_untouched(cfg):
    result = analyse_portfolio(HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg)

    np.testing.assert_array_equal(result.equity_values, result.total_values)
    assert result.equity_metrics.var_5 == result.total_metrics.var_5
    assert result.mean_option_payoff == 0.0


def test_multiple_overlays_accumulate(cfg):
    one = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg,
        options=[OptionPosition("AAPL", 170.0, 0.5, "put")],
    )
    two = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg,
        options=[
            OptionPosition("AAPL", 170.0, 0.5, "put"),
            OptionPosition("MSFT", 360.0, 0.5, "put"),
        ],
    )
    assert two.mean_option_payoff > one.mean_option_payoff


def test_payoff_is_carried_forward_at_the_risk_free_rate(cfg):
    """A payoff realised at expiry must accrue to the horizon, not vanish.

    Deep-in-the-money so every path pays the same intrinsic amount.
    """
    put = OptionPosition(
        asset="AAPL", strike=1e6, maturity_years=0.5, option_type="put",
        quantity=1.0, contract_size=1.0,
    )
    rate = 0.05
    result = analyse_portfolio(
        HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, options=[put], cfg=cfg,
        risk_free_rate=rate,
    )

    step = round(0.5 / cfg.dt)
    remaining = cfg.horizon_years - step * cfg.dt
    expected_growth = np.exp(rate * remaining)

    # Re-simulate with the same seed to recover the underlying at expiry, then
    # confirm the payoff was accrued by exactly exp(r * remaining).
    paths = simulate_correlated_gbm(
        np.array([SPOTS[n] for n in result.asset_names]),
        np.array([MUS[n] for n in result.asset_names]),
        np.array([SIGMAS[n] for n in result.asset_names]),
        CORRELATION, cfg=cfg,
    )
    aapl = paths[:, result.asset_names.index("AAPL"), step]
    expected = np.maximum(1e6 - aapl, 0.0) * expected_growth

    np.testing.assert_allclose(result.option_payoff_pv, expected, rtol=1e-9)


# ---------------------------------------------------------------- validation


def test_option_on_unknown_asset_rejected(cfg):
    with pytest.raises(ValueError, match="unknown asset"):
        analyse_portfolio(
            HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg,
            options=[OptionPosition("TSLA", 200.0, 0.5)],
        )


def test_option_expiring_after_horizon_rejected(cfg):
    with pytest.raises(ValueError, match="after the"):
        analyse_portfolio(
            HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg,
            options=[OptionPosition("AAPL", 170.0, 2.0)],
        )


def test_missing_parameters_rejected(cfg):
    with pytest.raises(ValueError, match="sigmas is missing"):
        analyse_portfolio(HOLDINGS, SPOTS, MUS, {"AAPL": 0.2}, CORRELATION, cfg=cfg)


def test_empty_portfolio_rejected(cfg):
    with pytest.raises(ValueError, match="at least one asset"):
        analyse_portfolio({}, {}, {}, {}, np.eye(0), cfg=cfg)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"strike": -1.0}, "strike"),
        ({"maturity_years": 0.0}, "maturity_years"),
        ({"option_type": "swap"}, "option_type"),
    ],
)
def test_invalid_option_position_rejected(kwargs, match):
    base = {"asset": "AAPL", "strike": 100.0, "maturity_years": 0.5}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        OptionPosition(**base)


def test_asset_ordering_is_deterministic(cfg):
    """REGRESSION: legacy took the correlation matrix from one asset's dict and
    assumed every other asset's row order matched list(portfolio_data.keys()).
    """
    shuffled = {"GOOGL": 25.0, "AAPL": 100.0, "MSFT": 50.0}
    a = analyse_portfolio(HOLDINGS, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg)
    b = analyse_portfolio(shuffled, SPOTS, MUS, SIGMAS, CORRELATION, cfg=cfg)

    assert a.asset_names == b.asset_names == ("AAPL", "GOOGL", "MSFT")
    np.testing.assert_array_equal(a.equity_values, b.equity_values)

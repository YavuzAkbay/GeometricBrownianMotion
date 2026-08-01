"""Analytic and Monte Carlo option pricing tests.

Reference values are computed from closed-form Black-Scholes or from
first principles, never from this package's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from gbm.config import SimConfig
from gbm.pricing import (
    black_scholes_price,
    greeks,
    implied_volatility,
    monte_carlo_price,
)
from gbm.processes import simulate_gbm

S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2


# ---------------------------------------------------------------- Black-Scholes


def test_atm_call_matches_reference_value():
    """Hull, Options Futures and Other Derivatives: S=K=100, T=1, r=5%, vol=20%."""
    assert black_scholes_price(S, K, T, R, SIGMA, "call") == pytest.approx(10.4506, abs=1e-4)


def test_atm_put_matches_reference_value():
    assert black_scholes_price(S, K, T, R, SIGMA, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("spot", [50.0, 90.0, 100.0, 110.0, 200.0])
@pytest.mark.parametrize("maturity", [0.08, 0.5, 1.0, 3.0])
def test_put_call_parity(spot, maturity):
    """C - P = S - K exp(-rT), exactly, for every spot and maturity."""
    call = black_scholes_price(spot, K, maturity, R, SIGMA, "call")
    put = black_scholes_price(spot, K, maturity, R, SIGMA, "put")
    assert call - put == pytest.approx(spot - K * np.exp(-R * maturity), abs=1e-10)


def test_zero_maturity_returns_intrinsic_value():
    """REGRESSION: legacy guarded d1/d2 with T_eff but discounted with raw T.

    The two disagreed at T=0, so price and Greeks were mutually inconsistent.
    """
    assert black_scholes_price(120.0, 100.0, 0.0, R, SIGMA, "call") == pytest.approx(20.0)
    assert black_scholes_price(80.0, 100.0, 0.0, R, SIGMA, "call") == pytest.approx(0.0)
    assert black_scholes_price(80.0, 100.0, 0.0, R, SIGMA, "put") == pytest.approx(20.0)


def test_zero_volatility_returns_discounted_intrinsic():
    price = black_scholes_price(110.0, 100.0, 1.0, R, 0.0, "call")
    assert price == pytest.approx(110.0 - 100.0 * np.exp(-R), abs=1e-9)


def test_price_is_monotone_in_volatility():
    prices = [black_scholes_price(S, K, T, R, v, "call") for v in (0.1, 0.2, 0.3, 0.4)]
    assert prices == sorted(prices)


def test_call_price_bounded_by_no_arbitrage():
    call = black_scholes_price(S, K, T, R, SIGMA, "call")
    assert max(S - K * np.exp(-R * T), 0.0) <= call <= S


@pytest.mark.parametrize("bad", [{"s": -1.0}, {"k": 0.0}, {"sigma": -0.2}])
def test_invalid_inputs_rejected(bad):
    kwargs = {"s": S, "k": K, "t": T, "r": R, "sigma": SIGMA, "option_type": "call"}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        black_scholes_price(**kwargs)


def test_unknown_option_type_rejected():
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(S, K, T, R, SIGMA, "straddle")


def test_pricing_is_vectorised_over_spot():
    spots = np.array([90.0, 100.0, 110.0])
    prices = black_scholes_price(spots, K, T, R, SIGMA, "call")
    assert prices.shape == (3,)
    for spot, price in zip(spots, prices, strict=True):
        assert price == pytest.approx(black_scholes_price(float(spot), K, T, R, SIGMA, "call"))


# ---------------------------------------------------------------- Greeks


def _finite_difference(arg: str, bump: float, option_type: str = "call") -> float:
    base = {"s": S, "k": K, "t": T, "r": R, "sigma": SIGMA, "option_type": option_type}
    up, down = dict(base), dict(base)
    up[arg] = base[arg] + bump
    down[arg] = base[arg] - bump
    return (black_scholes_price(**up) - black_scholes_price(**down)) / (2 * bump)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    g = greeks(S, K, T, R, SIGMA, option_type)
    assert g.delta == pytest.approx(_finite_difference("s", 1e-4, option_type), abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(option_type):
    """Vega is reported per 1.00 of vol, so it compares directly to d(price)/d(sigma)."""
    g = greeks(S, K, T, R, SIGMA, option_type)
    assert g.vega == pytest.approx(_finite_difference("sigma", 1e-5, option_type), abs=1e-4)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_finite_difference(option_type):
    """REGRESSION: rho was never computed at all in the legacy code."""
    g = greeks(S, K, T, R, SIGMA, option_type)
    assert g.rho == pytest.approx(_finite_difference("r", 1e-6, option_type), abs=1e-3)


def test_gamma_matches_second_difference():
    bump = 1e-3
    second = (
        black_scholes_price(S + bump, K, T, R, SIGMA, "call")
        - 2 * black_scholes_price(S, K, T, R, SIGMA, "call")
        + black_scholes_price(S - bump, K, T, R, SIGMA, "call")
    ) / bump**2
    assert greeks(S, K, T, R, SIGMA, "call").gamma == pytest.approx(second, abs=1e-4)


def test_theta_is_per_day_and_negative_for_atm_call():
    """REGRESSION: legacy theta was per-year but printed beside per-day dollars.

    d(price)/dT is the sensitivity to *increasing* maturity; theta is the decay
    per calendar day, hence the sign flip and the /365.
    """
    g = greeks(S, K, T, R, SIGMA, "call")
    per_year = -_finite_difference("t", 1e-6, "call")
    assert g.theta == pytest.approx(per_year / 365.0, abs=1e-6)
    assert g.theta < 0


def test_call_and_put_delta_differ_by_one():
    call = greeks(S, K, T, R, SIGMA, "call")
    put = greeks(S, K, T, R, SIGMA, "put")
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-9)


def test_gamma_and_vega_are_type_independent():
    call = greeks(S, K, T, R, SIGMA, "call")
    put = greeks(S, K, T, R, SIGMA, "put")
    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)


def test_call_delta_in_unit_interval():
    for spot in (10.0, 100.0, 500.0):
        assert 0.0 <= greeks(spot, K, T, R, SIGMA, "call").delta <= 1.0


def test_greeks_at_expiry_are_finite():
    g = greeks(S, K, 0.0, R, SIGMA, "call")
    assert all(np.isfinite(v) for v in (g.delta, g.gamma, g.vega, g.theta, g.rho))


# ---------------------------------------------------------------- Monte Carlo


def test_monte_carlo_converges_to_black_scholes():
    cfg = SimConfig(horizon_years=T, steps=252, n_paths=200_000, seed=99)
    paths = simulate_gbm(S, mu=R, sigma=SIGMA, cfg=cfg)

    result = monte_carlo_price(paths, strike=K, maturity=T, rate=R, option_type="call")
    analytic = black_scholes_price(S, K, T, R, SIGMA, "call")

    assert abs(result.price - analytic) < 4 * result.std_error
    assert result.ci_low <= analytic <= result.ci_high


def test_monte_carlo_put_converges_to_black_scholes():
    cfg = SimConfig(horizon_years=T, steps=252, n_paths=200_000, seed=100)
    paths = simulate_gbm(S, mu=R, sigma=SIGMA, cfg=cfg)

    result = monte_carlo_price(paths, strike=K, maturity=T, rate=R, option_type="put")
    analytic = black_scholes_price(S, K, T, R, SIGMA, "put")

    assert abs(result.price - analytic) < 4 * result.std_error


def test_monte_carlo_std_error_shrinks_with_paths():
    def se(n):
        cfg = SimConfig(horizon_years=T, steps=50, n_paths=n, seed=5)
        paths = simulate_gbm(S, mu=R, sigma=SIGMA, cfg=cfg)
        return monte_carlo_price(paths, K, T, R, "call").std_error

    small, large = se(2_000), se(32_000)
    # 16x the paths should cut the standard error by roughly 4x.
    assert large < small
    assert small / large == pytest.approx(4.0, rel=0.4)


def test_monte_carlo_price_is_non_negative():
    cfg = SimConfig(horizon_years=T, steps=50, n_paths=1000, seed=6)
    paths = simulate_gbm(S, mu=R, sigma=SIGMA, cfg=cfg)
    assert monte_carlo_price(paths, strike=1e6, maturity=T, rate=R).price >= 0.0


# ---------------------------------------------------------------- implied vol


@pytest.mark.parametrize("true_vol", [0.10, 0.20, 0.35, 0.80])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_inverts_black_scholes(true_vol, option_type):
    price = black_scholes_price(S, K, T, R, true_vol, option_type)
    assert implied_volatility(price, S, K, T, R, option_type) == pytest.approx(
        true_vol, abs=1e-6
    )


def test_implied_volatility_rejects_arbitrage_violating_price():
    """A call worth more than spot admits no implied vol."""
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(S + 1.0, S, K, T, R, "call")


def test_implied_volatility_recovers_smile_shape():
    """Each strike must invert back to the vol it was priced with."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.array([0.28, 0.24, 0.20, 0.22, 0.26])

    for strike, vol in zip(strikes, vols, strict=True):
        price = black_scholes_price(S, strike, T, R, vol, "call")
        assert implied_volatility(price, S, strike, T, R, "call") == pytest.approx(
            vol, abs=1e-6
        )

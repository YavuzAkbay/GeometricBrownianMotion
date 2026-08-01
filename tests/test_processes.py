"""Property tests for the path simulators.

Several of these fail against the pre-rewrite code; each such test names the
defect it pins so the regression is traceable. See the plan for full detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from gbm.config import SimConfig
from gbm.processes import (
    estimate_parameters,
    simulate_gbm,
    simulate_heston,
    simulate_merton_jump,
    simulate_regime_switching,
)

S0 = 100.0


@pytest.fixture
def cfg() -> SimConfig:
    return SimConfig(horizon_years=1.0, steps=252, n_paths=20_000, seed=42)


# ---------------------------------------------------------------- shape / basics


@pytest.mark.parametrize(
    "simulate,kwargs",
    [
        (simulate_gbm, {"mu": 0.08, "sigma": 0.2}),
        (simulate_heston, {"mu": 0.08, "v0": 0.04, "kappa": 3.0, "theta": 0.04,
                           "sigma_v": 0.3, "rho": -0.7}),
        (simulate_merton_jump, {"mu": 0.08, "sigma": 0.2, "lambda_jump": 1.0,
                                "mu_jump": -0.05, "sigma_jump": 0.1}),
    ],
)
def test_paths_have_expected_shape_and_start_at_s0(simulate, kwargs, cfg):
    paths = simulate(S0, cfg=cfg, **kwargs)
    assert paths.shape == (cfg.n_paths, cfg.steps + 1)
    np.testing.assert_allclose(paths[:, 0], S0)


@pytest.mark.parametrize(
    "simulate,kwargs",
    [
        (simulate_gbm, {"mu": 0.08, "sigma": 0.2}),
        (simulate_heston, {"mu": 0.08, "v0": 0.04, "kappa": 3.0, "theta": 0.04,
                           "sigma_v": 0.3, "rho": -0.7}),
        (simulate_merton_jump, {"mu": 0.08, "sigma": 0.2, "lambda_jump": 1.0,
                                "mu_jump": -0.05, "sigma_jump": 0.1}),
    ],
)
def test_prices_are_strictly_positive(simulate, kwargs, cfg):
    """REGRESSION: the old baseline used additive Euler, which admits S <= 0.

    ``dS = mu*S*dt + sigma*S*dW; S += dS`` (legacy gbm.py:693-696) can drive a
    path negative, after which downstream ``np.log(paths)`` produced NaN that
    the global warning filter hid.
    """
    paths = simulate(S0, cfg=cfg, **kwargs)
    assert np.all(paths > 0.0)
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize(
    "simulate,kwargs",
    [
        (simulate_gbm, {"mu": 0.08, "sigma": 0.2}),
        (simulate_heston, {"mu": 0.08, "v0": 0.04, "kappa": 3.0, "theta": 0.04,
                           "sigma_v": 0.3, "rho": -0.7}),
        (simulate_merton_jump, {"mu": 0.08, "sigma": 0.2, "lambda_jump": 1.0,
                                "mu_jump": -0.05, "sigma_jump": 0.1}),
    ],
)
def test_same_seed_reproduces_and_different_seed_differs(simulate, kwargs):
    a = simulate(S0, cfg=SimConfig(n_paths=500, seed=1), **kwargs)
    b = simulate(S0, cfg=SimConfig(n_paths=500, seed=1), **kwargs)
    c = simulate(S0, cfg=SimConfig(n_paths=500, seed=2), **kwargs)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_simulators_do_not_touch_global_numpy_rng():
    """REGRESSION: legacy simulators called np.random.seed(42) internally.

    That clobbered global state and made back-to-back calls draw *identical*
    streams, so 'independent' model comparisons shared random numbers.
    """
    # Legacy global-RNG API used on purpose here: this test exists to prove the
    # simulators leave that global state untouched.
    np.random.seed(999)  # noqa: NPY002
    before = np.random.random()  # noqa: NPY002

    np.random.seed(999)  # noqa: NPY002
    simulate_gbm(S0, mu=0.08, sigma=0.2, cfg=SimConfig(n_paths=100, seed=7))
    after = np.random.random()  # noqa: NPY002

    assert before == after


def test_consecutive_calls_draw_independent_streams():
    """REGRESSION: internal reseeding made two different models share draws."""
    cfg = SimConfig(n_paths=2000, seed=None)
    a = simulate_gbm(S0, mu=0.08, sigma=0.2, cfg=cfg)
    b = simulate_gbm(S0, mu=0.08, sigma=0.2, cfg=cfg)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------- GBM moments


def test_gbm_terminal_mean_matches_analytic(cfg):
    """E[S_T] = S0 * exp(mu * T) for the arithmetic drift mu."""
    mu, sigma = 0.08, 0.2
    paths = simulate_gbm(S0, mu=mu, sigma=sigma, cfg=cfg)
    terminal = paths[:, -1]

    expected = S0 * np.exp(mu * cfg.horizon_years)
    stderr = terminal.std(ddof=1) / np.sqrt(cfg.n_paths)
    assert abs(terminal.mean() - expected) < 4 * stderr


def test_gbm_terminal_log_variance_matches_analytic(cfg):
    """Var[log S_T] = sigma^2 * T."""
    mu, sigma = 0.08, 0.2
    paths = simulate_gbm(S0, mu=mu, sigma=sigma, cfg=cfg)
    log_var = np.log(paths[:, -1] / S0).var(ddof=1)
    assert log_var == pytest.approx(sigma**2 * cfg.horizon_years, rel=0.05)


def test_gbm_zero_volatility_is_deterministic_growth():
    cfg = SimConfig(horizon_years=2.0, steps=100, n_paths=10, seed=3)
    paths = simulate_gbm(S0, mu=0.05, sigma=0.0, cfg=cfg)
    expected = S0 * np.exp(0.05 * cfg.horizon_years)
    np.testing.assert_allclose(paths[:, -1], expected, rtol=1e-10)


# ---------------------------------------------------------------- estimation


def test_estimate_parameters_recovers_volatility(synthetic_prices, true_params):
    """Volatility is estimated precisely even from a few years of daily data."""
    est = estimate_parameters(synthetic_prices["Close"])
    assert est.sigma == pytest.approx(true_params["sigma"], rel=0.05)


def test_estimate_parameters_applies_ito_correction_to_log_returns(synthetic_prices):
    """REGRESSION: legacy code added the Ito term to *simple* returns.

    gbm.py:1878-1880 computed ``Returns.mean()*252 + 0.5*vol**2`` from
    ``pct_change()``, double-counting drift because simple returns already
    average ~mu rather than mu - sigma^2/2.

    Asserted as an exact algebraic identity against this sample's own log
    returns, so it does not depend on how the path happened to realise.
    """
    close = synthetic_prices["Close"].to_numpy()
    log_returns = np.diff(np.log(close))

    est = estimate_parameters(close)

    assert est.log_drift == pytest.approx(log_returns.mean() * 252)
    assert est.sigma == pytest.approx(log_returns.std(ddof=1) * np.sqrt(252))
    assert est.mu == pytest.approx(est.log_drift + 0.5 * est.sigma**2)

    # The legacy formula, for contrast: it is materially different, which is
    # exactly why reported drifts were wrong.
    simple_returns = np.diff(close) / close[:-1]
    legacy_mu = simple_returns.mean() * 252 + 0.5 * est.sigma**2
    assert abs(legacy_mu - est.mu) > 0.5 * est.sigma**2


def test_estimate_parameters_drift_converges_on_long_sample(true_params):
    """Drift is recoverable, but only with enough data to beat its own SE.

    Drift SE is sigma/sqrt(years); at sigma=0.25 that is 0.025 over 100 years.
    """
    mu, sigma, s0, years = true_params["mu"], true_params["sigma"], true_params["s0"], 100
    n = 252 * years
    dt = 1.0 / 252

    gen = np.random.default_rng(2024)
    z = gen.standard_normal(n)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    close = s0 * np.exp(np.concatenate([[0.0], np.cumsum(increments)]))

    est = estimate_parameters(close)
    stderr = sigma / np.sqrt(years)
    assert abs(est.mu - mu) < 3 * stderr


def test_estimated_drift_reproduces_terminal_mean(synthetic_prices):
    """Round trip: estimate then simulate must preserve E[S_T]."""
    close = synthetic_prices["Close"]
    est = estimate_parameters(close)
    s0 = float(close.iloc[-1])

    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=40_000, seed=11)
    terminal = simulate_gbm(s0, mu=est.mu, sigma=est.sigma, cfg=cfg)[:, -1]

    expected = s0 * np.exp(est.mu * cfg.horizon_years)
    stderr = terminal.std(ddof=1) / np.sqrt(cfg.n_paths)
    assert abs(terminal.mean() - expected) < 4 * stderr


# ---------------------------------------------------------------- Heston


def test_heston_variance_never_negative(cfg):
    """CIR full truncation must keep variance in [0, inf)."""
    paths, variance = simulate_heston(
        S0, mu=0.08, v0=0.04, kappa=1.0, theta=0.04,
        # Deliberately violates Feller (2*kappa*theta < sigma_v^2) to stress it.
        sigma_v=0.9, rho=-0.7, cfg=cfg, return_variance=True,
    )
    assert np.all(variance >= 0.0)
    assert np.all(np.isfinite(variance))
    assert np.all(paths > 0.0)


def test_heston_initial_variance_is_v0_not_theta():
    """REGRESSION: legacy set v[:,0]=theta *before* correcting theta.

    enhanced_gbm.py:204 wrote the initial variance, then :206-211 recreated
    theta, so v0 kept an uncorrected value.
    """
    _, variance = simulate_heston(
        S0, mu=0.05, v0=0.09, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5,
        cfg=SimConfig(n_paths=100, seed=5), return_variance=True,
    )
    np.testing.assert_allclose(variance[:, 0], 0.09)


def test_heston_with_zero_vol_of_vol_collapses_to_gbm():
    """sigma_v=0, v0=theta makes variance constant, so Heston == GBM."""
    sigma = 0.2
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=20_000, seed=17)

    heston = simulate_heston(
        S0, mu=0.06, v0=sigma**2, kappa=1.0, theta=sigma**2,
        sigma_v=0.0, rho=0.0, cfg=cfg,
    )
    expected = S0 * np.exp(0.06 * cfg.horizon_years)
    stderr = heston[:, -1].std(ddof=1) / np.sqrt(cfg.n_paths)
    assert abs(heston[:, -1].mean() - expected) < 4 * stderr


def test_heston_leverage_effect_has_correct_sign(cfg):
    """Negative rho must produce negative price/variance correlation.

    REGRESSION: the legacy scheme used v_{t+1} in the price step for time t,
    an anticipating scheme that corrupts this correlation.
    """
    paths, variance = simulate_heston(
        S0, mu=0.0, v0=0.04, kappa=2.0, theta=0.04,
        sigma_v=0.5, rho=-0.8, cfg=cfg, return_variance=True,
    )
    log_ret = np.log(paths[:, -1] / S0)
    var_change = variance[:, -1] - variance[:, 0]
    assert np.corrcoef(log_ret, var_change)[0, 1] < -0.1


def test_heston_mean_reverts_towards_theta():
    cfg = SimConfig(horizon_years=10.0, steps=2520, n_paths=2000, seed=23)
    _, variance = simulate_heston(
        S0, mu=0.0, v0=0.16, kappa=3.0, theta=0.04, sigma_v=0.2, rho=0.0,
        cfg=cfg, return_variance=True,
    )
    # Started at 4x theta; after 10 years at kappa=3 it should be close to theta.
    assert variance[:, -1].mean() == pytest.approx(0.04, rel=0.25)


# ---------------------------------------------------------------- Merton


def test_merton_is_risk_neutral_martingale():
    """REGRESSION: the compensator was missing entirely.

    Without ``- lambda*k*dt`` where ``k = exp(mu_j + sigma_j^2/2) - 1``, the
    discounted price is not a martingale and every jump-diffusion option price
    is biased. Legacy: enhanced_gbm.py:407 and gbm.py:1791.
    """
    r = 0.05
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=200_000, seed=31)

    terminal = simulate_merton_jump(
        S0, mu=r, sigma=0.2, lambda_jump=2.0, mu_jump=-0.10, sigma_jump=0.15,
        cfg=cfg,
    )[:, -1]

    expected = S0 * np.exp(r * cfg.horizon_years)
    stderr = terminal.std(ddof=1) / np.sqrt(cfg.n_paths)
    assert abs(terminal.mean() - expected) < 4 * stderr


def test_merton_with_zero_intensity_collapses_to_gbm():
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=5000, seed=41)
    kwargs = {"mu": 0.07, "sigma": 0.22, "cfg": cfg}

    jump = simulate_merton_jump(
        S0, lambda_jump=0.0, mu_jump=-0.05, sigma_jump=0.1, **kwargs
    )
    plain = simulate_gbm(S0, **kwargs)
    np.testing.assert_allclose(jump, plain, rtol=1e-12)


def test_merton_allows_more_than_one_jump_per_step():
    """REGRESSION: legacy drew Bernoulli(1-exp(-lambda*dt)), capping at 1 jump.

    enhanced_gbm.py:412-413. With a high intensity and coarse steps, a Poisson
    process must produce multi-jump intervals.
    """
    from gbm.processes.jump import draw_jump_counts

    counts = draw_jump_counts(
        lambda_jump=50.0, dt=0.1, size=(5000, 10), rng=np.random.default_rng(3)
    )
    assert counts.max() > 1
    assert counts.mean() == pytest.approx(50.0 * 0.1, rel=0.05)


def test_merton_jump_sizes_are_not_clamped():
    """REGRESSION: legacy clamped jump multipliers to [0.1, 10].

    That truncates exactly the tail the model exists to represent. With a wide
    jump distribution some multipliers must fall outside those bounds.
    """
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=50_000, seed=53)
    terminal = simulate_merton_jump(
        S0, mu=0.0, sigma=0.05, lambda_jump=3.0, mu_jump=0.0, sigma_jump=1.2,
        cfg=cfg,
    )[:, -1]
    ratio = terminal / S0
    assert ratio.max() > 10.0 or ratio.min() < 0.1


def test_merton_fat_tails_exceed_gbm():
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=50_000, seed=61)
    from scipy.stats import kurtosis

    jump = simulate_merton_jump(
        S0, mu=0.05, sigma=0.15, lambda_jump=2.0, mu_jump=-0.08,
        sigma_jump=0.15, cfg=cfg,
    )[:, -1]
    plain = simulate_gbm(S0, mu=0.05, sigma=0.15, cfg=cfg)[:, -1]

    assert kurtosis(np.log(jump)) > kurtosis(np.log(plain)) + 0.5


# ---------------------------------------------------------------- regime switching


def test_regime_switching_shapes_and_positivity(cfg):
    paths, regimes = simulate_regime_switching(
        S0,
        mu_states=np.array([0.10, -0.05, -0.20]),
        sigma_states=np.array([0.12, 0.25, 0.45]),
        transition_matrix=np.array(
            [[0.98, 0.015, 0.005], [0.05, 0.90, 0.05], [0.02, 0.18, 0.80]]
        ),
        cfg=cfg,
        return_regimes=True,
    )
    assert paths.shape == (cfg.n_paths, cfg.steps + 1)
    assert regimes.shape == (cfg.n_paths, cfg.steps + 1)
    assert np.all(paths > 0.0)
    assert set(np.unique(regimes)).issubset({0, 1, 2})


def test_regime_switching_single_state_equals_gbm():
    cfg = SimConfig(horizon_years=1.0, steps=252, n_paths=5000, seed=71)
    paths = simulate_regime_switching(
        S0,
        mu_states=np.array([0.06]),
        sigma_states=np.array([0.2]),
        transition_matrix=np.array([[1.0]]),
        cfg=cfg,
    )
    expected = S0 * np.exp(0.06 * cfg.horizon_years)
    stderr = paths[:, -1].std(ddof=1) / np.sqrt(cfg.n_paths)
    assert abs(paths[:, -1].mean() - expected) < 4 * stderr


def test_regime_switching_rejects_non_stochastic_matrix():
    with pytest.raises(ValueError, match="rows must sum to 1"):
        simulate_regime_switching(
            S0,
            mu_states=np.array([0.1, -0.1]),
            sigma_states=np.array([0.2, 0.4]),
            transition_matrix=np.array([[0.5, 0.2], [0.3, 0.7]]),
            cfg=SimConfig(n_paths=10, seed=1),
        )


def test_regime_switching_visits_all_states(cfg):
    _, regimes = simulate_regime_switching(
        S0,
        mu_states=np.array([0.10, -0.05, -0.20]),
        sigma_states=np.array([0.12, 0.25, 0.45]),
        transition_matrix=np.array(
            [[0.90, 0.07, 0.03], [0.10, 0.80, 0.10], [0.05, 0.25, 0.70]]
        ),
        cfg=cfg,
        return_regimes=True,
    )
    assert set(np.unique(regimes)) == {0, 1, 2}


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [{"steps": 0}, {"n_paths": 0}, {"horizon_years": -1.0}])
def test_simconfig_rejects_invalid(bad):
    with pytest.raises(ValueError):
        SimConfig(**bad)


def test_negative_volatility_rejected():
    with pytest.raises(ValueError, match="sigma"):
        simulate_gbm(S0, mu=0.05, sigma=-0.2, cfg=SimConfig(n_paths=10, seed=1))


def test_negative_initial_price_rejected():
    with pytest.raises(ValueError, match="s0"):
        simulate_gbm(-1.0, mu=0.05, sigma=0.2, cfg=SimConfig(n_paths=10, seed=1))

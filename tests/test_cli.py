"""CLI and analysis-orchestration tests. Network is always mocked."""

from __future__ import annotations

import numpy as np
import pytest

from gbm.analysis import MODEL_NAMES, analyse, compare_models
from gbm.cli import build_parser, main


@pytest.fixture(autouse=True)
def offline(monkeypatch, synthetic_prices):
    """Serve synthetic data in place of yfinance for every test here."""

    class FakeYF:
        @staticmethod
        def download(ticker, **kwargs):
            return synthetic_prices.copy()

        class Ticker:
            def __init__(self, symbol):
                pass

            def history(self, **kwargs):
                return synthetic_prices.copy()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)


@pytest.fixture(autouse=True)
def no_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("gbm.data.DEFAULT_CACHE_DIR", tmp_path / "cache")


# ---------------------------------------------------------------- analysis API


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_model_produces_valid_paths(model):
    result = analyse("AAPL", model=model, months=6, n_paths=2000, seed=42)

    # 6 months at 21 trading days = 126 steps, so 127 columns including t=0.
    assert result.paths.shape == (2000, 127)
    assert np.all(result.paths > 0)
    assert result.model == model
    assert result.spot > 0


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_model_reports_coherent_risk(model):
    m = analyse("AAPL", model=model, months=6, n_paths=5000, seed=42).metrics

    assert m.cvar_5 <= m.var_5
    assert -1.0 < m.max_drawdown <= 0.0
    assert 0.0 <= m.profit_probability <= 1.0


def test_analysis_is_reproducible():
    a = analyse("AAPL", n_paths=1000, seed=7)
    b = analyse("AAPL", n_paths=1000, seed=7)
    np.testing.assert_array_equal(a.paths, b.paths)


def test_seed_none_is_not_reproducible():
    a = analyse("AAPL", n_paths=1000, seed=None)
    b = analyse("AAPL", n_paths=1000, seed=None)
    assert not np.array_equal(a.paths, b.paths)


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        analyse("AAPL", model="blackscholes")


def test_summary_mentions_the_ticker_and_model():
    text = analyse("AAPL", model="heston", n_paths=500, seed=1).summary()
    assert "AAPL" in text
    assert "heston" in text
    assert "VaR" in text


# ---------------------------------------------------------------- comparison


def test_comparison_runs_every_model():
    result = compare_models("AAPL", months=6, n_paths=2000, seed=42)
    assert set(result.results) == set(MODEL_NAMES)


def test_comparison_models_use_independent_random_streams():
    """REGRESSION: the legacy code reseeded to 42 inside every simulator, so a
    model and its 'independent' baseline drew identical normals.
    """
    result = compare_models("AAPL", n_paths=2000, seed=42)

    # compare_models offsets the seed per model, so no two models may share a
    # random stream. Compare the standardised shocks rather than the prices,
    # which differ anyway because the models differ.
    def shocks(paths):
        increments = np.diff(np.log(paths), axis=1)
        return (increments - increments.mean()) / increments.std()

    gbm_shocks = shocks(result.results["gbm"].paths)
    jump_shocks = shocks(result.results["jump"].paths)

    assert not np.allclose(gbm_shocks, jump_shocks)


def test_ranking_puts_safest_model_first():
    """REGRESSION: legacy sorted signed CVaR ascending under 'lower is better',
    so rank 1 was the riskiest model.
    """
    result = compare_models("AAPL", n_paths=5000, seed=42)
    ranked = result.ranked_by_risk()

    cvars = [result.results[n].metrics.cvar_5 for n in ranked]
    assert cvars == sorted(cvars, reverse=True)  # least negative (safest) first


def test_comparison_summary_is_tabular():
    text = compare_models("AAPL", n_paths=1000, seed=42).summary()
    assert "Model" in text and "CVaR" in text
    for name in MODEL_NAMES:
        assert name in text


# ---------------------------------------------------------------- CLI


def test_analyze_command_succeeds(capsys, tmp_path):
    code = main(["analyze", "AAPL", "--sims", "500", "--no-plots", "--output", str(tmp_path)])

    assert code == 0
    assert "AAPL" in capsys.readouterr().out


def test_ticker_is_a_required_argument():
    """REGRESSION: legacy gbm.py hardcoded ticker = 'XLU'."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze"])


def test_cli_never_blocks_on_stdin(monkeypatch, tmp_path):
    """REGRESSION: enhanced_gbm.py called input(), making it un-scriptable."""

    def explode(*args, **kwargs):
        raise AssertionError("CLI must never read from stdin")

    monkeypatch.setattr("builtins.input", explode)
    assert main(["analyze", "AAPL", "--sims", "200", "--no-plots",
                 "--output", str(tmp_path)]) == 0


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_model_selectable_from_cli(model, tmp_path):
    assert main(["analyze", "AAPL", "--model", model, "--sims", "200",
                 "--no-plots", "--output", str(tmp_path)]) == 0


def test_compare_command_succeeds(capsys, tmp_path):
    code = main(["compare", "AAPL", "--sims", "500", "--no-plots", "--output", str(tmp_path)])

    assert code == 0
    assert "Model comparison" in capsys.readouterr().out


def test_options_command_reports_all_greeks(capsys, tmp_path):
    code = main(["options", "AAPL", "--strike", "100", "--sims", "2000",
                 "--no-plots", "--output", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    for greek in ("Delta", "Gamma", "Vega", "Theta", "Rho"):
        assert greek in out, f"{greek} missing from options output"


def test_options_monte_carlo_agrees_with_black_scholes(capsys, tmp_path):
    """The two prices printed side by side must actually agree."""
    import re

    main(["options", "AAPL", "--sims", "40000", "--no-plots", "--output", str(tmp_path)])
    out = capsys.readouterr().out

    analytic = float(re.search(r"Black-Scholes price: \$([\d,.]+)", out).group(1).replace(",", ""))
    mc = float(re.search(r"Monte Carlo price:   \$([\d,.]+)", out).group(1).replace(",", ""))
    se = float(re.search(r"\+/- ([\d.]+) se", out).group(1))

    assert abs(analytic - mc) < 5 * se


def test_plots_are_written_when_not_disabled(tmp_path):
    code = main(["analyze", "AAPL", "--sims", "500", "--output", str(tmp_path)])

    assert code == 0
    written = list((tmp_path / "plots").glob("*.png"))
    assert len(written) >= 2
    assert all(p.stat().st_size > 1000 for p in written)


def test_bad_ticker_exits_nonzero_without_traceback(monkeypatch, tmp_path):
    import pandas as pd

    class DeadYF:
        @staticmethod
        def download(ticker, **kwargs):
            return pd.DataFrame()

        class Ticker:
            def __init__(self, symbol):
                pass

            def history(self, **kwargs):
                return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", DeadYF)

    assert main(["analyze", "NOPE", "--sims", "100", "--no-plots",
                 "--output", str(tmp_path)]) == 1


def test_quiet_and_verbose_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-q", "-v", "analyze", "AAPL"])


@pytest.mark.parametrize(
    "argv",
    [
        ["-q", "analyze", "AAPL"],
        ["analyze", "AAPL", "-q"],
    ],
)
def test_quiet_accepted_before_or_after_subcommand(argv):
    """Global flags must work in either position; requiring them before the
    subcommand is a trap users hit constantly.
    """
    assert build_parser().parse_args(argv).quiet is True


@pytest.mark.parametrize(
    "argv",
    [
        ["-v", "analyze", "AAPL"],
        ["analyze", "AAPL", "-v"],
        ["-vv", "analyze", "AAPL"],
    ],
)
def test_verbose_accepted_before_or_after_subcommand(argv):
    assert build_parser().parse_args(argv).verbose >= 1


def test_verbosity_defaults_are_present_without_flags():
    args = build_parser().parse_args(["analyze", "AAPL"])
    assert args.verbose == 0
    assert args.quiet is False


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0


def test_negative_seed_means_random():
    parser = build_parser()
    assert parser.parse_args(["analyze", "AAPL", "--seed", "-1"]).seed == -1

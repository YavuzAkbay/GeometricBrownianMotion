"""Explainability and plotting tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from gbm.config import OutputConfig, SimConfig  # noqa: E402
from gbm.explain import (  # noqa: E402
    analyse_attention,
    calibration,
    permutation_importance,
)
from gbm.features import build_dataset, build_features  # noqa: E402
from gbm.models import TrainConfig, train_model  # noqa: E402
from gbm.processes import simulate_gbm  # noqa: E402
from gbm.risk import RiskMetrics, terminal_returns  # noqa: E402
from gbm.viz import (  # noqa: E402
    plot_attention,
    plot_feature_importance,
    plot_paths,
    plot_return_distribution,
    plot_training_history,
    save_figure,
)


@pytest.fixture(scope="module")
def trained(synthetic_prices_module):
    features = build_features(synthetic_prices_module)
    dataset = build_dataset(features, sequence_length=20)
    return train_model(
        dataset,
        TrainConfig(epochs=2, d_model=32, n_heads=4, n_layers=1, seed=11, device="cpu"),
    )


# ---------------------------------------------------------------- importance


def test_permutation_importance_shapes_and_names(trained):
    ds = trained.dataset
    imp = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names,
        device=trained.device, n_repeats=2,
    )

    assert imp.names == ds.feature_names
    assert imp.scores.shape == (ds.n_features,)
    assert imp.method == "permutation"


def test_permutation_importance_is_non_negative(trained):
    ds = trained.dataset
    imp = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names,
        device=trained.device, n_repeats=2,
    )
    assert np.all(imp.scores >= 0)


def test_permutation_importance_runs_without_dropout(trained):
    """REGRESSION: legacy computed permutation importance with dropout active
    and gradients tracked, so scores were randomised by dropout noise.

    Repeating with the same seed must give identical results.
    """
    ds = trained.dataset
    kwargs = {
        "device": trained.device, "n_repeats": 2, "seed": 99,
    }
    a = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names, **kwargs
    )
    b = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names, **kwargs
    )
    np.testing.assert_allclose(a.scores, b.scores)
    assert not trained.model.training


def test_importance_normalisation_sums_to_one(trained):
    ds = trained.dataset
    imp = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names,
        device=trained.device, n_repeats=2,
    )
    assert imp.normalised.sum() == pytest.approx(1.0)


def test_importance_ranking_is_descending(trained):
    ds = trained.dataset
    imp = permutation_importance(
        trained.model, ds.x_test[:64], ds.y_test[:64], ds.feature_names,
        device=trained.device, n_repeats=2,
    )
    scores = [s for _, s in imp.ranked]
    assert scores == sorted(scores, reverse=True)


def test_mismatched_feature_names_rejected(trained):
    ds = trained.dataset
    with pytest.raises(ValueError, match="names for"):
        permutation_importance(
            trained.model, ds.x_test[:8], ds.y_test[:8], ("only", "two"),
            device=trained.device,
        )


def test_missing_shap_raises_rather_than_degrading(trained, monkeypatch):
    """REGRESSION: a failed analysis must never silently become a different
    analysis, or worse, placeholder numbers on a chart.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    from gbm.explain import shap_importance

    with pytest.raises(ImportError, match="gbm-quant\\[explain\\]"):
        shap_importance(
            trained.model, trained.dataset.x_test[:8], trained.dataset.feature_names,
            device=trained.device,
        )


# ---------------------------------------------------------------- attention


def test_attention_analysis_shapes(trained):
    analysis = analyse_attention(
        trained.model, trained.dataset.x_test[:64], device=trained.device
    )
    assert analysis.mean_weights.shape == (20,)
    assert analysis.per_sample.shape == (64, 20)


def test_attention_stability_is_bounded(trained):
    analysis = analyse_attention(
        trained.model, trained.dataset.x_test[:64], device=trained.device
    )
    assert 0.0 <= analysis.stability <= 1.0
    assert 0 <= analysis.most_attended_step < 20


def test_attention_handles_more_samples_than_grid_slots(trained):
    """REGRESSION: legacy allocated a fixed 2x3 subplot grid then looped over
    every sample, raising IndexError for more than six.
    """
    analysis = analyse_attention(
        trained.model, trained.dataset.x_test[:50], device=trained.device
    )
    assert analysis.per_sample.shape[0] == min(50, len(trained.dataset.x_test))


# ---------------------------------------------------------------- calibration


def test_calibration_detects_informative_uncertainty():
    """Uncertainty that tracks error must score a high reliability."""
    rng = np.random.default_rng(0)
    n = 2000
    uncertainty = rng.uniform(0.1, 2.0, n)
    targets = np.zeros(n)
    predictions = rng.normal(0.0, uncertainty)

    result = calibration(predictions, targets, uncertainty)

    assert result.reliability > 0.3
    assert result.correlation > 0


def test_calibration_reliability_is_not_inverted():
    """REGRESSION: legacy scored reliability as 1 - abs(corr), so a perfectly
    informative uncertainty head scored 0.0 -- exactly backwards.
    """
    rng = np.random.default_rng(1)
    n = 2000
    uncertainty = rng.uniform(0.1, 2.0, n)
    informative = calibration(rng.normal(0, uncertainty), np.zeros(n), uncertainty)
    uninformative = calibration(
        rng.normal(0, 1.0, n), np.zeros(n), np.full(n, 0.5)
    )

    assert informative.reliability > uninformative.reliability


def test_calibration_error_is_small_when_well_calibrated():
    rng = np.random.default_rng(2)
    n = 5000
    uncertainty = rng.uniform(0.5, 1.5, n)
    result = calibration(rng.normal(0, uncertainty), np.zeros(n), uncertainty)

    assert result.calibration_error < 0.3


def test_calibration_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        calibration(np.zeros(10), np.zeros(9), np.ones(10))


def test_calibration_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least"):
        calibration(np.zeros(3), np.zeros(3), np.ones(3), n_bins=10)


# ---------------------------------------------------------------- plotting


@pytest.fixture
def paths():
    return simulate_gbm(100.0, 0.08, 0.25, SimConfig(n_paths=2000, steps=126, seed=5))


def test_every_plot_is_actually_written(tmp_path, paths, trained):
    """REGRESSION: legacy plot functions ended at tight_layout() with no
    savefig, so under the forced Agg backend every figure was silently
    discarded while the code printed 'Creating visualization...'.
    """
    output = OutputConfig(root=tmp_path)
    metrics = RiskMetrics.from_paths(paths, horizon_years=0.5)

    analysis = analyse_attention(
        trained.model, trained.dataset.x_test[:32], device=trained.device
    )
    ds = trained.dataset
    importance = permutation_importance(
        trained.model, ds.x_test[:32], ds.y_test[:32], ds.feature_names,
        device=trained.device, n_repeats=1,
    )

    figures = {
        "paths": plot_paths(paths),
        "returns": plot_return_distribution(
            terminal_returns(paths), metrics.var_5, metrics.cvar_5
        ),
        "importance": plot_feature_importance(importance),
        "attention": plot_attention(analysis),
        "history": plot_training_history(trained.history),
    }

    for name, fig in figures.items():
        path = save_figure(fig, name, output)
        assert path.exists()
        assert path.stat().st_size > 1000


def test_save_figure_respects_configured_dpi(tmp_path, paths):
    """REGRESSION: legacy hardcoded dpi=400 for every figure."""
    low = save_figure(plot_paths(paths), "low", OutputConfig(root=tmp_path, dpi=72))
    high = save_figure(plot_paths(paths), "high", OutputConfig(root=tmp_path, dpi=200))

    assert high.stat().st_size > low.stat().st_size


def test_output_directories_are_created_on_demand(tmp_path, paths):
    output = OutputConfig(root=tmp_path / "nested" / "run")
    assert not output.plots.exists()

    save_figure(plot_paths(paths), "x", output)
    assert output.plots.exists()


def test_plot_paths_handles_fewer_paths_than_requested():
    small = simulate_gbm(100.0, 0.05, 0.2, SimConfig(n_paths=3, steps=10, seed=1))
    fig = plot_paths(small, n_shown=500)
    assert fig is not None
    matplotlib.pyplot.close(fig)

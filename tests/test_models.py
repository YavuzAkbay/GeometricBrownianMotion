"""Model and training tests. CPU-only, small, and deterministic."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gbm.features import build_dataset, build_features  # noqa: E402
from gbm.models import (  # noqa: E402
    TrainConfig,
    TransformerPredictor,
    evaluate,
    train_model,
)

SEQ = 20


@pytest.fixture
def dataset(synthetic_prices):
    features = build_features(synthetic_prices)
    return build_dataset(features, sequence_length=SEQ)


@pytest.fixture
def config() -> TrainConfig:
    return TrainConfig(
        epochs=3, batch_size=64, d_model=32, n_heads=4, n_layers=1,
        seed=7, device="cpu",
    )


# ---------------------------------------------------------------- architecture


def test_forward_returns_price_logvar_and_attention(dataset):
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    x = torch.tensor(dataset.x_train[:8], dtype=torch.float32)

    price, log_var, attention = model(x)

    assert price.shape == (8,)
    assert log_var.shape == (8,)
    assert attention.shape == (8, SEQ)


def test_attention_weights_sum_to_one(dataset):
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    weights = model.attention_weights(torch.tensor(dataset.x_train[:16], dtype=torch.float32))

    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)
    assert np.all(weights >= 0)


def test_predicted_volatility_is_positive(dataset):
    """REGRESSION: the legacy vol head was a sigmoid*0.5 that was never trained,
    so its output was a random number that still passed the sanity gate.

    Predicting log-variance makes positivity structural.
    """
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    out = model.predict(torch.tensor(dataset.x_train[:32], dtype=torch.float32))

    assert np.all(out.volatility > 0)
    assert np.all(np.isfinite(out.volatility))


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        TransformerPredictor(n_features=5, d_model=30, n_heads=7)


# ---------------------------------------------------------------- MC dropout


def test_mc_dropout_leaves_norm_layers_in_eval_mode(dataset):
    """REGRESSION: legacy called self.train() inside predict_with_uncertainty,
    re-enabling normalisation running-stat updates while predicting on test data.
    """
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    model.enable_mc_dropout()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.training, "dropout should be active for MC sampling"
        elif isinstance(module, torch.nn.LayerNorm | torch.nn.BatchNorm1d):
            assert not module.training, "normalisation must stay in eval mode"


def test_uncertainty_has_both_components(dataset):
    model = TransformerPredictor(
        n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1, dropout=0.3
    )
    x = torch.tensor(dataset.x_test[:16], dtype=torch.float32)

    result = model.predict_with_uncertainty(x, n_samples=10)

    assert result.mean.shape == (16,)
    assert np.all(result.epistemic >= 0)
    assert np.all(result.aleatoric > 0)
    assert np.all(result.total_uncertainty >= result.epistemic)


def test_confidence_is_bounded(dataset):
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    result = model.predict_with_uncertainty(
        torch.tensor(dataset.x_test[:16], dtype=torch.float32), n_samples=5
    )
    confidence = result.confidence()

    assert np.all(confidence > 0)
    assert np.all(confidence <= 1.0)


def test_model_returns_to_eval_after_uncertainty_sampling(dataset):
    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    model.predict_with_uncertainty(
        torch.tensor(dataset.x_test[:8], dtype=torch.float32), n_samples=3
    )
    assert not model.training


# ---------------------------------------------------------------- training


def test_training_records_both_curves(dataset, config):
    trained = train_model(dataset, config)

    assert len(trained.history.train_loss) > 0
    assert len(trained.history.val_loss) == len(trained.history.train_loss)


def test_validation_loss_is_tracked_separately(dataset, config):
    """REGRESSION: legacy scheduler and early stopping keyed off training loss,
    leaving the test set as the only holdout -- and also the reported result.
    """
    trained = train_model(dataset, config)

    assert trained.history.val_loss != trained.history.train_loss
    assert np.isfinite(trained.history.best_val_loss)


def test_volatility_head_receives_gradient(dataset, config):
    """REGRESSION: the legacy loss used only the price head, so the volatility
    head was never trained -- yet its output was fed into the simulators.
    """
    from gbm.models.train import _gaussian_nll

    model = TransformerPredictor(n_features=dataset.n_features, d_model=32, n_heads=4, n_layers=1)
    x = torch.tensor(dataset.x_train[:32], dtype=torch.float32)
    y = torch.tensor(dataset.y_train[:32], dtype=torch.float32)

    pred, log_var, _ = model(x)
    _gaussian_nll(pred, log_var, y, torch).backward()

    grads = [
        p.grad for p in model.log_var_head.parameters() if p.grad is not None
    ]
    assert grads, "log-variance head received no gradient at all"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_training_is_reproducible(dataset, config):
    """REGRESSION: torch.manual_seed was never called anywhere in the legacy
    code, so weight init and dropout were non-reproducible even when NumPy
    was seeded.
    """
    a = train_model(dataset, config)
    b = train_model(dataset, config)

    np.testing.assert_allclose(a.history.train_loss, b.history.train_loss, rtol=1e-5)


def test_best_weights_are_restored_not_final(dataset):
    config = TrainConfig(
        epochs=8, batch_size=64, d_model=32, n_heads=4, n_layers=1, seed=3, device="cpu"
    )
    trained = train_model(dataset, config)

    best = min(trained.history.val_loss)
    assert trained.history.val_loss[trained.history.best_epoch] == pytest.approx(best)


def test_early_stopping_triggers_on_no_improvement(dataset):
    # Zero LR and zero dropout make the model completely static, so validation
    # loss is identical every epoch and patience must cut training short.
    config = TrainConfig(
        epochs=100, patience=2, batch_size=64, d_model=16, n_heads=2,
        n_layers=1, learning_rate=0.0, dropout=0.0, seed=5, device="cpu",
    )
    trained = train_model(dataset, config)

    assert trained.history.stopped_early
    assert len(trained.history.train_loss) < 100


@pytest.mark.parametrize("bad", [{"epochs": 0}, {"batch_size": -1}])
def test_invalid_train_config_rejected(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


# ---------------------------------------------------------------- evaluation


def test_evaluation_metrics_are_in_raw_units(dataset, config):
    """REGRESSION: legacy trained on normalised targets but evaluated against
    raw ones, so every reported R2/MAE/IC compared mismatched scales.

    Raw daily log returns are ~1e-2, so MAE must be small in absolute terms.
    """
    trained = train_model(dataset, config)
    metrics = evaluate(trained, "test")

    assert metrics.mae < 0.5
    assert metrics.n_samples == len(dataset.x_test)
    assert np.isfinite(metrics.r2)


def test_evaluation_bounds_are_sane(dataset, config):
    trained = train_model(dataset, config)
    metrics = evaluate(trained, "test")

    assert 0.0 <= metrics.directional_accuracy <= 1.0
    assert -1.0 <= metrics.information_coefficient <= 1.0
    assert metrics.mse >= 0
    assert metrics.mae >= 0


def test_evaluate_rejects_unknown_split(dataset, config):
    trained = train_model(dataset, config)
    with pytest.raises(ValueError, match="Unknown split"):
        evaluate(trained, "holdout")


def test_metrics_dict_is_serialisable(dataset, config):
    import json

    trained = train_model(dataset, config)
    json.dumps(evaluate(trained, "test").to_dict())

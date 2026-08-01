"""Training loop.

Fixes relative to the legacy trainer:

* The LR scheduler and early stopping keyed off **training** loss, so the test
  set was the only holdout and was also what got reported. Both now key off a
  separate validation split.
* Only the price head was in the loss; the volatility head was an untrained
  random projection whose output was nonetheless fed into the simulators. The
  loss is now a Gaussian negative log-likelihood, which trains both heads
  jointly and makes the predicted variance meaningful.
* Metrics were computed on raw targets while the model was trained on
  normalised ones. Evaluation now inverse-transforms first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend import get_device, require_torch, seed_everything, to_numpy
from ..features import Dataset
from ..logging import get_logger
from .transformer import TransformerPredictor

log = get_logger(__name__)

__all__ = ["TrainConfig", "TrainedModel", "TrainingHistory", "evaluate", "train_model"]


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip: float = 1.0
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.2
    seed: int | None = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


@dataclass
class TrainingHistory:
    """Per-epoch losses. Both curves are recorded so overfitting is visible."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("nan")


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation metrics, all computed in raw target units."""

    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    information_coefficient: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "mae": self.mae,
            "r2": self.r2,
            "directional_accuracy": self.directional_accuracy,
            "information_coefficient": self.information_coefficient,
            "n_samples": float(self.n_samples),
        }


@dataclass
class TrainedModel:
    """A trained model plus the data it was trained on."""

    model: object
    dataset: Dataset
    history: TrainingHistory
    config: TrainConfig
    device: object


def _gaussian_nll(pred, log_var, target, torch):
    """Gaussian negative log-likelihood with a learned per-sample variance.

    Trains the price and variance heads jointly. ``log_var`` is clamped to keep
    the exponential from overflowing early in training, when the variance head
    is still effectively random.
    """
    log_var = torch.clamp(log_var, min=-10.0, max=10.0)
    precision = torch.exp(-log_var)
    return (precision * (pred - target) ** 2 + log_var).mean()


def train_model(dataset: Dataset, config: TrainConfig | None = None) -> TrainedModel:
    """Train the transformer predictor on a prepared dataset.

    Args:
        dataset: Output of :func:`gbm.features.build_dataset`.
        config: Hyperparameters.

    Returns:
        A :class:`TrainedModel` holding the best-validation weights, not the
        final-epoch weights.
    """
    torch = require_torch()
    config = config or TrainConfig()

    seed_everything(config.seed)
    device = get_device(config.device)
    log.info("Training on %s", device)

    model = TransformerPredictor(
        n_features=dataset.n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)

    def as_tensors(x, y):
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )

    x_train, y_train = as_tensors(dataset.x_train, dataset.y_train)
    x_val, y_val = as_tensors(dataset.x_val, dataset.y_val)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(config.patience // 3, 1)
    )

    history = TrainingHistory()
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    n_train = len(x_train)

    for epoch in range(config.epochs):
        model.train()
        # Shuffling indices on-device avoids a host round-trip per epoch.
        permutation = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, config.batch_size):
            idx = permutation[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)

            pred, log_var, _ = model(x_train[idx])
            loss = _gaussian_nll(pred, log_var, y_train[idx], torch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred, val_log_var, _ = model(x_val)
            val_loss = float(_gaussian_nll(val_pred, val_log_var, y_val, torch).item())

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        # Scheduling and stopping both key off validation, not training, loss.
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            log.info(
                "epoch %3d/%d  train %.5f  val %.5f", epoch + 1, config.epochs,
                train_loss, val_loss,
            )

        if epochs_without_improvement >= config.patience:
            log.info("Early stopping at epoch %d (best epoch %d)", epoch + 1, history.best_epoch + 1)
            history.stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedModel(
        model=model, dataset=dataset, history=history, config=config, device=device
    )


def evaluate(trained: TrainedModel, split: str = "test") -> EvalMetrics:
    """Evaluate on a split, in raw target units.

    Args:
        trained: A :class:`TrainedModel`.
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        :class:`EvalMetrics`. Predictions and targets are both
        inverse-transformed first — the legacy code compared normalised
        predictions against raw targets, invalidating every reported figure.
    """
    torch = require_torch()

    try:
        x = getattr(trained.dataset, f"x_{split}")
        y = getattr(trained.dataset, f"y_{split}")
    except AttributeError as exc:
        raise ValueError(f"Unknown split {split!r}; expected train, val or test") from exc

    model = trained.model
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=trained.device)
        pred_scaled, _, _ = model(tensor)

    predictions = trained.dataset.inverse_transform_y(to_numpy(pred_scaled))
    targets = trained.dataset.inverse_transform_y(y)

    errors = predictions - targets
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))

    total_variance = float(np.var(targets))
    r2 = 1.0 - mse / total_variance if total_variance > 0 else 0.0

    # Sign agreement between predicted and realised returns. Each row is an
    # independent forecast, so this compares signs directly rather than taking
    # np.diff over row order as the legacy metric did.
    directional = float(np.mean(np.sign(predictions) == np.sign(targets)))

    if predictions.std() > 0 and targets.std() > 0:
        ic = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        ic = 0.0

    return EvalMetrics(
        mse=mse,
        mae=mae,
        r2=r2,
        directional_accuracy=directional,
        information_coefficient=ic,
        n_samples=len(targets),
    )

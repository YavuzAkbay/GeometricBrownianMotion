"""Explainability: feature importance, attention analysis, calibration.

**This module computes; it never plots.** The legacy code shipped
``create_feature_importance_analysis`` alongside a copy-pasted
``..._no_plot`` twin whose two permutation implementations had silently
diverged (``mean()`` in one, ``mean(abs())`` in the other). Rendering lives in
:mod:`gbm.viz`.

A failed computation raises. It never substitutes placeholder data: the legacy
code had eleven ``except`` blocks that fed invented numbers into charts
(``y=[1.0]*10`` labelled "Feature Importance (Placeholder)", a fabricated
single-point calibration curve, zeroed importances), producing plausible-looking
plots of values that were never measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import require_torch, to_numpy
from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "AttentionAnalysis",
    "CalibrationResult",
    "FeatureImportance",
    "analyse_attention",
    "calibration",
    "permutation_importance",
    "shap_importance",
]


@dataclass(frozen=True)
class FeatureImportance:
    """Per-feature importance scores, sorted descending."""

    names: tuple[str, ...]
    scores: np.ndarray
    method: str

    @property
    def ranked(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.scores)
        return [(self.names[i], float(self.scores[i])) for i in order]

    @property
    def normalised(self) -> np.ndarray:
        """Scores scaled to sum to 1, for cumulative-importance analysis."""
        total = self.scores.sum()
        if total <= 0:
            return np.full_like(self.scores, 1.0 / len(self.scores))
        return self.scores / total

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return self.ranked[:n]


@dataclass(frozen=True)
class AttentionAnalysis:
    """Attention weights over timesteps, aggregated across samples."""

    mean_weights: np.ndarray
    std_weights: np.ndarray
    per_sample: np.ndarray

    @property
    def stability(self) -> float:
        """Consistency of attention across samples, in [0, 1].

        1 means every sample attends identically; 0 means maximal disagreement.
        """
        mean_of_std = float(self.std_weights.mean())
        mean_of_mean = float(self.mean_weights.mean())
        if mean_of_mean <= 0:
            return 0.0
        return float(1.0 / (1.0 + mean_of_std / mean_of_mean))

    @property
    def most_attended_step(self) -> int:
        return int(np.argmax(self.mean_weights))


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration of predicted uncertainty against realised error."""

    expected_confidence: np.ndarray
    observed_frequency: np.ndarray
    calibration_error: float
    correlation: float

    @property
    def reliability(self) -> float:
        """How informative the uncertainty estimate is, in [0, 1].

        Higher is better: uncertainty that tracks realised error scores near 1.
        The legacy ``reliability_score`` was ``1 - abs(corr)``, which scored a
        *perfectly* informative uncertainty head as 0.0 -- exactly backwards.
        """
        return float(abs(self.correlation))


def _batched_forward(model, x, device, batch_size: int = 512) -> np.ndarray:
    """Run the price head over ``x`` in batches.

    The legacy explainability code looped one single-row forward pass per
    sample, in eight separate places, at 1000 forwards each.
    """
    torch = require_torch()
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            price, _, _ = model(chunk)
            outputs.append(to_numpy(price))

    return np.concatenate(outputs)


def permutation_importance(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    device=None,
    n_repeats: int = 5,
    seed: int | None = 42,
) -> FeatureImportance:
    """Importance as the increase in MSE when a feature is shuffled.

    Args:
        model: A trained predictor.
        x: Input sequences, shape ``(n, seq, n_features)``.
        y: Targets, shape ``(n,)``.
        feature_names: One name per feature.
        device: Torch device. Defaults to the model's.
        n_repeats: Shuffles per feature; results are averaged.
        seed: RNG seed for the shuffles.

    Returns:
        A :class:`FeatureImportance` with ``method="permutation"``.

    Note:
        Runs entirely under ``eval()`` and ``no_grad()``. The legacy version
        called the model with dropout still active and gradients tracked, so
        its importances were randomised by dropout noise.
    """
    if len(feature_names) != x.shape[-1]:
        raise ValueError(
            f"Got {len(feature_names)} names for {x.shape[-1]} features"
        )

    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)

    baseline = float(np.mean((_batched_forward(model, x, device) - y) ** 2))

    scores = np.zeros(len(feature_names))
    for j in range(len(feature_names)):
        deltas = []
        for _ in range(n_repeats):
            shuffled = x.copy()
            # Permute this feature across samples, preserving its time
            # structure within each sample.
            shuffled[:, :, j] = shuffled[rng.permutation(len(x)), :, j]
            mse = float(np.mean((_batched_forward(model, shuffled, device) - y) ** 2))
            deltas.append(mse - baseline)
        scores[j] = float(np.mean(deltas))

    # Importance is a magnitude; a negative delta means the feature is noise.
    return FeatureImportance(
        names=tuple(feature_names), scores=np.maximum(scores, 0.0), method="permutation"
    )


def shap_importance(
    model,
    x: np.ndarray,
    feature_names: tuple[str, ...],
    device=None,
    background_size: int = 100,
    sample_size: int = 200,
    seed: int | None = 42,
) -> FeatureImportance:
    """Mean absolute SHAP value per feature.

    Args:
        model: A trained predictor.
        x: Input sequences.
        feature_names: One name per feature.
        device: Torch device.
        background_size: Reference samples for the explainer.
        sample_size: Samples to explain.
        seed: RNG seed for subsampling.

    Returns:
        A :class:`FeatureImportance` with ``method="shap"``.

    Raises:
        ImportError: If ``shap`` is not installed. The caller decides how to
            proceed; this never silently degrades to a different method.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            'SHAP is required for this analysis. Install with:\n'
            '    pip install "gbm-quant[explain]"'
        ) from exc

    torch = require_torch()
    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)

    n_background = min(background_size, len(x))
    n_explain = min(sample_size, len(x))

    background = torch.tensor(
        x[rng.choice(len(x), n_background, replace=False)], dtype=torch.float32, device=device
    )
    to_explain = torch.tensor(
        x[rng.choice(len(x), n_explain, replace=False)], dtype=torch.float32, device=device
    )

    class _PriceHead(torch.nn.Module):
        """Expose only the scalar price output, as SHAP requires."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, batch):
            return self.inner(batch)[0].unsqueeze(-1)

    model.eval()
    explainer = shap.DeepExplainer(_PriceHead(model), background)
    values = explainer.shap_values(to_explain, check_additivity=False)

    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)

    # Shape is (n, seq, features[, 1]); collapse everything but the feature axis.
    while values.ndim > 3:
        values = values[..., 0]
    scores = np.abs(values).mean(axis=(0, 1))

    return FeatureImportance(names=tuple(feature_names), scores=scores, method="shap")


def analyse_attention(model, x: np.ndarray, device=None, max_samples: int = 500) -> AttentionAnalysis:
    """Aggregate attention weights over a batch of samples.

    Args:
        model: A trained predictor exposing ``attention_weights``.
        x: Input sequences.
        device: Torch device.
        max_samples: Cap on samples analysed.

    Returns:
        An :class:`AttentionAnalysis` covering ``min(len(x), max_samples)`` rows.
    """
    torch = require_torch()
    device = device or next(model.parameters()).device

    subset = x[: min(len(x), max_samples)]
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(subset, dtype=torch.float32, device=device)
        weights = model.attention_weights(tensor)

    return AttentionAnalysis(
        mean_weights=weights.mean(axis=0),
        std_weights=weights.std(axis=0),
        per_sample=weights,
    )


def calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Assess whether predicted uncertainty tracks realised error.

    Samples are bucketed by predicted uncertainty; within each bucket the mean
    predicted uncertainty is compared against the realised RMSE. A
    well-calibrated model has these equal, so the calibration error is their
    mean absolute difference.

    The legacy ECE instead binarised the target at its own median, which fixes
    the base rate at 0.5 by construction and makes the statistic carry no
    calibration information -- yet it gated the report's recommendations.

    Args:
        predictions: Point predictions.
        targets: Realised values.
        uncertainty: Predicted standard deviation per sample.
        n_bins: Number of uncertainty buckets.

    Returns:
        A :class:`CalibrationResult`.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()
    uncertainty = np.asarray(uncertainty, dtype=float).ravel()

    if not len(predictions) == len(targets) == len(uncertainty):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, "
            f"targets={len(targets)}, uncertainty={len(uncertainty)}"
        )
    if len(predictions) < n_bins:
        raise ValueError(f"Need at least {n_bins} samples to form {n_bins} bins")

    errors = np.abs(predictions - targets)

    # Equal-count bins, so every bucket carries the same statistical weight.
    order = np.argsort(uncertainty)
    bins = np.array_split(order, n_bins)

    expected, observed = [], []
    for idx in bins:
        if idx.size == 0:
            continue
        expected.append(float(uncertainty[idx].mean()))
        observed.append(float(np.sqrt((errors[idx] ** 2).mean())))

    expected_arr = np.array(expected)
    observed_arr = np.array(observed)

    if uncertainty.std() > 0 and errors.std() > 0:
        correlation = float(np.corrcoef(uncertainty, errors)[0, 1])
    else:
        correlation = 0.0

    return CalibrationResult(
        expected_confidence=expected_arr,
        observed_frequency=observed_arr,
        calibration_error=float(np.abs(expected_arr - observed_arr).mean()),
        correlation=correlation,
    )

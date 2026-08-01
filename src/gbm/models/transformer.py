"""Transformer price predictor with calibrated uncertainty.

Two defects from the legacy model are fixed here:

* The old network exposed ``volatility_predictor`` and ``drift_predictor``
  heads that were **never included in the loss** — training optimised only the
  price head. Their outputs were untrained random projections, yet ``ml_vol``
  was fed straight into the simulators, where a random sigmoid output landed in
  ``[0, 0.5]`` and passed the ``0.05 <= vol <= 1.0`` sanity gate. Here the
  volatility head is trained against realised forward volatility and is part of
  the loss; there is no separate drift head, because a drift estimate belongs
  to :func:`gbm.processes.estimate_parameters`, not to a price model.
* ``predict_with_uncertainty`` called ``self.train()`` inside the loop, which
  re-enabled BatchNorm running-stat updates while predicting on test data.
  Dropout is now toggled explicitly, layer by layer, leaving normalisation
  layers in eval mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..backend import require_torch, to_numpy

__all__ = ["ModelOutput", "PositionalEncoding", "TransformerPredictor", "UncertainPrediction"]


def _nn():
    return require_torch().nn


@dataclass(frozen=True)
class ModelOutput:
    """Raw forward-pass outputs, in scaled target units."""

    price: np.ndarray
    log_variance: np.ndarray

    @property
    def volatility(self) -> np.ndarray:
        """Predicted per-step volatility, always positive."""
        return np.exp(0.5 * self.log_variance)


@dataclass(frozen=True)
class UncertainPrediction:
    """MC-dropout prediction with its uncertainty decomposition."""

    mean: np.ndarray
    epistemic: np.ndarray
    aleatoric: np.ndarray

    @property
    def total_uncertainty(self) -> np.ndarray:
        """Combined standard deviation of the predictive distribution."""
        return np.sqrt(self.epistemic**2 + self.aleatoric**2)

    def confidence(self) -> np.ndarray:
        """A bounded confidence score in (0, 1], decreasing in uncertainty.

        Reported as-is. The legacy pipeline rescaled confidence into [0.1, 0.9]
        and then printed "Confidence rescaling disabled" on the same line.
        """
        spread = self.total_uncertainty
        return 1.0 / (1.0 + spread / (np.abs(self.mean).mean() + 1e-12))


def _build_positional_encoding(max_len: int, d_model: int):
    torch = require_torch()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def PositionalEncoding(d_model: int, max_len: int = 5000):
    """Return a positional-encoding module for ``d_model`` features."""
    nn = _nn()

    class _PositionalEncoding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("pe", _build_positional_encoding(max_len, d_model))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    return _PositionalEncoding()


def TransformerPredictor(
    n_features: int,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    dropout: float = 0.2,
):
    """Build the transformer predictor.

    Constructed by a factory rather than declared at module scope so that
    importing :mod:`gbm` does not require torch.

    Args:
        n_features: Number of input features per timestep.
        d_model: Model width. Must be divisible by ``n_heads``.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        dropout: Dropout probability.

    Returns:
        A ``torch.nn.Module`` whose ``forward`` returns ``(price, log_var,
        attention)``.
    """
    torch = require_torch()
    nn = torch.nn

    if d_model % n_heads:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    class _TransformerPredictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            self.input_projection = nn.Linear(n_features, d_model)
            self.positional_encoding = PositionalEncoding(d_model)
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor is explicitly off: it is incompatible with
            # norm_first=True (which we want for training stability), and
            # leaving it on only emits a warning saying it was ignored.
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers, enable_nested_tensor=False
            )

            # Attention pooling over timesteps, so the weights are inspectable
            # for explainability instead of being reverse-engineered.
            self.attention_pool = nn.Linear(d_model, 1)

            self.price_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            # Predicts log-variance, so the exponentiated value is positive
            # without a sigmoid clamp that silently bounds the range.
            self.log_var_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, x):
            """Run a forward pass.

            Args:
                x: Tensor of shape ``(batch, sequence, n_features)``.

            Returns:
                ``(price, log_var, attention_weights)``. ``price`` and
                ``log_var`` are ``(batch,)``; attention is ``(batch, sequence)``.
            """
            h = self.input_projection(x)
            h = self.positional_encoding(h)
            h = self.dropout(h)
            h = self.encoder(h)

            scores = self.attention_pool(h).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            pooled = (h * weights.unsqueeze(-1)).sum(dim=1)

            return (
                self.price_head(pooled).squeeze(-1),
                self.log_var_head(pooled).squeeze(-1),
                weights,
            )

        def enable_mc_dropout(self) -> None:
            """Put *only* dropout layers into training mode.

            The legacy code called ``self.train()``, which also re-enabled
            normalisation running-stat updates while predicting on test data.
            """
            self.eval()
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

        @torch.no_grad()
        def predict(self, x) -> ModelOutput:
            """Deterministic prediction with dropout disabled."""
            self.eval()
            price, log_var, _ = self(x)
            return ModelOutput(price=to_numpy(price), log_variance=to_numpy(log_var))

        @torch.no_grad()
        def predict_with_uncertainty(self, x, n_samples: int = 50) -> UncertainPrediction:
            """MC-dropout prediction separating epistemic from aleatoric noise.

            Args:
                x: Input batch.
                n_samples: Stochastic forward passes.

            Returns:
                An :class:`UncertainPrediction`. Epistemic uncertainty is the
                spread across passes; aleatoric is the mean predicted noise.
            """
            self.enable_mc_dropout()

            means, variances = [], []
            for _ in range(n_samples):
                price, log_var, _ = self(x)
                means.append(price)
                variances.append(torch.exp(log_var))

            stacked = torch.stack(means)
            self.eval()

            return UncertainPrediction(
                mean=to_numpy(stacked.mean(dim=0)),
                epistemic=to_numpy(stacked.std(dim=0)),
                aleatoric=to_numpy(torch.stack(variances).mean(dim=0).sqrt()),
            )

        @torch.no_grad()
        def attention_weights(self, x) -> np.ndarray:
            """Attention over timesteps, batched.

            The legacy explainability code ran one single-row forward pass per
            sample, in eight separate places.
            """
            self.eval()
            _, _, weights = self(x)
            return to_numpy(weights)

    return _TransformerPredictor()

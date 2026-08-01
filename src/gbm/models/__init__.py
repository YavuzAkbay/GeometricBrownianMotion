"""Neural predictors.

Importing this subpackage requires torch (``pip install "gbm-quant[ml]"``).
The top-level :mod:`gbm` package does not import it, so the numerical core
works without torch installed.
"""

from __future__ import annotations

from .train import EvalMetrics, TrainConfig, TrainedModel, TrainingHistory, evaluate, train_model
from .transformer import ModelOutput, TransformerPredictor, UncertainPrediction

__all__ = [
    "EvalMetrics",
    "ModelOutput",
    "TrainConfig",
    "TrainedModel",
    "TrainingHistory",
    "TransformerPredictor",
    "UncertainPrediction",
    "evaluate",
    "train_model",
]

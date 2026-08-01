"""Plotting.

Every function takes already-computed data and returns a
``matplotlib.figure.Figure``. Nothing here computes a statistic, and nothing
here calls ``plt.show()``.

Two legacy defects are addressed:

* ``plot_gbm_analysis`` and ``compare_multiple_stocks`` ended at
  ``plt.tight_layout()`` with no ``savefig`` and no ``close``. Under the forced
  ``Agg`` backend every figure was silently discarded and leaked, so the CLI
  printed "Creating visualization..." and produced nothing. :func:`save_figure`
  is the single exit point and it verifies the file exists.
* The backend was forced to ``Agg`` at import, which broke interactive use for
  every importer. It is now selected by the caller (the CLI does this), never
  as an import side effect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import OutputConfig
from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "plot_attention",
    "plot_feature_importance",
    "plot_paths",
    "plot_return_distribution",
    "plot_training_history",
    "save_figure",
    "use_headless_backend",
]


def use_headless_backend() -> None:
    """Select the non-interactive Agg backend.

    Call this from an entry point before importing pyplot-dependent code when
    running without a display. Library modules must not call it: doing so at
    import time is what broke interactive use of the legacy modules.
    """
    import matplotlib

    matplotlib.use("Agg")


def _plt():
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, name: str, output: OutputConfig | None = None) -> Path:
    """Save a figure to the plots directory and close it.

    Args:
        fig: The figure to save.
        name: Filename stem, without extension.
        output: Output settings. Defaults to :class:`OutputConfig`.

    Returns:
        The path written.

    Raises:
        OSError: If the file was not created.
    """
    output = output or OutputConfig()
    output.ensure()

    path = output.plots / f"{name}.png"
    fig.savefig(path, dpi=output.dpi, bbox_inches="tight", facecolor="white")
    _plt().close(fig)

    if not path.exists():
        raise OSError(f"Figure {name!r} was not written to {path}")

    log.info("Saved plot: %s", path)
    return path


def plot_paths(
    paths: np.ndarray,
    time_axis: np.ndarray | None = None,
    title: str = "Simulated price paths",
    n_shown: int = 200,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
):
    """Plot a sample of paths with a percentile fan.

    Args:
        paths: Price paths, shape ``(n_paths, n_steps + 1)``.
        time_axis: X values. Defaults to step indices.
        title: Figure title.
        n_shown: Individual paths drawn faintly behind the fan.
        percentiles: Percentile bands to overlay.
    """
    plt = _plt()
    if time_axis is None:
        time_axis = np.arange(paths.shape[1])

    fig, ax = plt.subplots(figsize=(11, 6))

    shown = min(n_shown, len(paths))
    step = max(len(paths) // shown, 1)
    ax.plot(time_axis, paths[::step].T, color="steelblue", alpha=0.05, linewidth=0.6)

    bands = np.percentile(paths, percentiles, axis=0)
    for pct, band in zip(percentiles, bands, strict=True):
        style = "-" if pct == 50 else "--"
        width = 2.0 if pct == 50 else 1.2
        ax.plot(time_axis, band, style, linewidth=width, label=f"{pct:g}th percentile")

    ax.axhline(paths[0, 0], color="black", linewidth=1.0, alpha=0.5, label="Start")
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_return_distribution(
    returns: np.ndarray,
    var_5: float | None = None,
    cvar_5: float | None = None,
    title: str = "Terminal return distribution",
):
    """Histogram of terminal returns with VaR and CVaR markers."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.hist(returns, bins=80, color="steelblue", alpha=0.75, edgecolor="none")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(
        float(np.mean(returns)), color="darkgreen", linewidth=1.6,
        label=f"Mean {np.mean(returns):+.2%}",
    )

    if var_5 is not None:
        ax.axvline(var_5, color="darkorange", linestyle="--", linewidth=1.6,
                   label=f"VaR 5% {var_5:.2%}")
    if cvar_5 is not None:
        ax.axvline(cvar_5, color="firebrick", linestyle="--", linewidth=1.6,
                   label=f"CVaR 5% {cvar_5:.2%}")

    ax.set_title(title)
    ax.set_xlabel("Total return")
    ax.set_ylabel("Paths")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance, top_n: int = 15, title: str | None = None):
    """Horizontal bar chart of the top features.

    Args:
        importance: A :class:`gbm.explain.FeatureImportance`.
        top_n: Features to show.
        title: Overrides the default, which names the method used.
    """
    plt = _plt()
    ranked = importance.top(top_n)
    names = [n for n, _ in ranked][::-1]
    scores = [s for _, s in ranked][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.35 * len(names))))
    ax.barh(names, scores, color="steelblue")
    ax.set_title(title or f"Feature importance ({importance.method})")
    ax.set_xlabel("Importance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_attention(analysis, title: str = "Attention over timesteps"):
    """Mean attention per timestep with a one-standard-deviation band."""
    plt = _plt()
    steps = np.arange(len(analysis.mean_weights))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, analysis.mean_weights, color="steelblue", linewidth=1.8, label="Mean")
    ax.fill_between(
        steps,
        analysis.mean_weights - analysis.std_weights,
        analysis.mean_weights + analysis.std_weights,
        color="steelblue", alpha=0.2, label="+/- 1 sd",
    )
    ax.axvline(
        analysis.most_attended_step, color="firebrick", linestyle="--", linewidth=1.2,
        label=f"Peak at step {analysis.most_attended_step}",
    )

    ax.set_title(f"{title}  (stability {analysis.stability:.2f})")
    ax.set_xlabel("Timestep in sequence")
    ax.set_ylabel("Attention weight")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_history(history, title: str = "Training history"):
    """Training and validation loss curves, with the best epoch marked."""
    plt = _plt()
    epochs = np.arange(1, len(history.train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history.train_loss, label="Train", linewidth=1.6)
    ax.plot(epochs, history.val_loss, label="Validation", linewidth=1.6)
    ax.axvline(
        history.best_epoch + 1, color="firebrick", linestyle="--", linewidth=1.2,
        label=f"Best epoch {history.best_epoch + 1}",
    )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

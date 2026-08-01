"""Feature engineering and leakage-free dataset construction.

Three leaks in the legacy pipeline are closed here:

1. **Overlapping sequences straddled the split.** Sequences were built from the
   full series and then cut at ``int(0.8 * n)``, so the first ``sequence_length``
   test windows contained rows that also appeared in training windows. A gap of
   ``sequence_length`` is now inserted between splits.
2. **No validation set.** The LR scheduler and early stopping both keyed off
   *training* loss, so the only holdout was also the reported test set. There
   are now three disjoint splits.
3. **Scalers were fit on everything.** ``fit_transform`` ran on the full array
   before any split. Scalers are now fit on train only and applied to the rest.

Every indicator is causal: it uses ``rolling``/``ewm`` over past values only,
never a centred window or a negative ``shift``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .logging import get_logger

log = get_logger(__name__)

FEATURE_COLUMNS = (
    "log_return",
    "volatility_20",
    "volatility_60",
    "momentum_10",
    "momentum_30",
    "rsi_14",
    "macd",
    "macd_signal",
    "bollinger_position",
    "volume_ratio",
    "high_low_range",
    "close_to_sma20",
    "close_to_sma50",
)


def relative_strength_index(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI over a trailing window.

    Returns values in [0, 100]; a flat series yields 50 rather than NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    # Where there are no losses RSI is 100 by definition; guard the division
    # rather than letting it emit a RuntimeWarning that used to be suppressed.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the causal technical feature set.

    Args:
        frame: OHLCV frame indexed chronologically.

    Returns:
        A frame containing :data:`FEATURE_COLUMNS` plus ``target``, with warmup
        rows dropped. ``target`` is the *next* day's log return, so row ``t``
        predicts ``t + 1`` and no feature on row ``t`` sees it.
    """
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in frame.columns:
            raise ValueError(f"Missing required column {col!r}")

    close = frame["Close"]
    out = pd.DataFrame(index=frame.index)

    out["log_return"] = np.log(close / close.shift(1))

    out["volatility_20"] = out["log_return"].rolling(20).std() * np.sqrt(252)
    out["volatility_60"] = out["log_return"].rolling(60).std() * np.sqrt(252)

    out["momentum_10"] = close / close.shift(10) - 1.0
    out["momentum_30"] = close / close.shift(30) - 1.0

    out["rsi_14"] = relative_strength_index(close, 14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    # Position within the Bollinger band; 0.5 when the band has no width.
    band_width = (2 * std_20).replace(0.0, np.nan)
    out["bollinger_position"] = ((close - (sma_20 - 2 * std_20)) / band_width).fillna(0.5)

    volume_sma = frame["Volume"].rolling(20).mean().replace(0.0, np.nan)
    out["volume_ratio"] = (frame["Volume"] / volume_sma).fillna(1.0)

    out["high_low_range"] = (frame["High"] - frame["Low"]) / close

    out["close_to_sma20"] = close / sma_20 - 1.0
    out["close_to_sma50"] = close / close.rolling(50).mean() - 1.0

    # Target is strictly forward-looking; features are strictly backward-looking.
    out["target"] = out["log_return"].shift(-1)

    return out.dropna()


@dataclass(frozen=True)
class Dataset:
    """Scaled, split sequence data ready for training.

    Splits are chronological and disjoint, separated by ``gap`` rows so no
    training window overlaps a validation or test window.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    x_scaler: StandardScaler
    y_scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to raw log-return units.

        Reporting metrics requires this: the legacy code trained on normalised
        targets but evaluated against raw ones, so every reported R2, MAE and
        IC compared mismatched units.
        """
        return self.y_scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()


def make_sequences(
    values: np.ndarray, targets: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows.

    Returns:
        ``(x, y)`` where ``x`` has shape ``(n, sequence_length, n_features)``
        and ``y[i]`` is the target aligned with the last row of ``x[i]``.
    """
    n = len(values) - sequence_length + 1
    if n <= 0:
        raise ValueError(
            f"Need at least {sequence_length} rows to build a sequence, got {len(values)}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=sequence_length, axis=0
    )
    # sliding_window_view gives (n, n_features, seq); models expect (n, seq, feat).
    x = np.ascontiguousarray(windows.transpose(0, 2, 1))
    y = targets[sequence_length - 1 :]
    return x, y


def build_dataset(
    frame: pd.DataFrame,
    sequence_length: int = 60,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> Dataset:
    """Build scaled, gapped, chronologically split sequence data.

    Args:
        frame: Output of :func:`build_features`.
        sequence_length: Window length. Also the size of the gap between
            splits, which is what prevents window overlap.
        train_frac: Fraction of rows for training.
        val_frac: Fraction for validation. Test gets the remainder.
        feature_columns: Columns to use as inputs.

    Returns:
        A :class:`Dataset` whose scalers were fit on the training split only.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1, got {train_frac + val_frac}"
        )

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing feature columns {missing}")

    x_raw = frame[list(feature_columns)].to_numpy(dtype=float)
    y_raw = frame["target"].to_numpy(dtype=float).reshape(-1, 1)

    n = len(frame)
    gap = sequence_length
    train_end = int(n * train_frac)
    val_start = train_end + gap
    val_end = val_start + int(n * val_frac)
    test_start = val_end + gap

    min_rows = sequence_length + 1
    if n - test_start < min_rows:
        raise ValueError(
            f"Not enough data: {n} rows leaves {n - test_start} for the test "
            f"split, need at least {min_rows}. Use a longer period or a "
            f"shorter sequence_length."
        )

    # Scalers see training rows only. The legacy pipeline fit on the full array
    # before splitting, leaking test-period mean and variance into training.
    x_scaler = StandardScaler().fit(x_raw[:train_end])
    y_scaler = StandardScaler().fit(y_raw[:train_end])

    x_scaled = x_scaler.transform(x_raw)
    y_scaled = y_scaler.transform(y_raw).ravel()

    def slice_split(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        return make_sequences(x_scaled[start:end], y_scaled[start:end], sequence_length)

    x_train, y_train = slice_split(0, train_end)
    x_val, y_val = slice_split(val_start, val_end)
    x_test, y_test = slice_split(test_start, n)

    log.debug(
        "Dataset: train=%d val=%d test=%d sequences (gap=%d rows between splits)",
        len(x_train), len(x_val), len(x_test), gap,
    )

    return Dataset(
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        feature_names=tuple(feature_columns),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

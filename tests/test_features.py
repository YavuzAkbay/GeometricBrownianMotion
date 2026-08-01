"""Tests for feature engineering and split construction.

The point of this file is leakage: features must be causal, splits must be
disjoint and gapped, and scalers must never see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbm.features import (
    FEATURE_COLUMNS,
    build_dataset,
    build_features,
    make_sequences,
    relative_strength_index,
)


@pytest.fixture
def features(synthetic_prices) -> pd.DataFrame:
    return build_features(synthetic_prices)


# ---------------------------------------------------------------- causality


def test_features_are_causal(synthetic_prices):
    """Truncating the future must not change any past feature value.

    This is the definitive look-ahead test: compute features on the full
    series, then on a prefix, and require the overlap to match exactly.
    """
    full = build_features(synthetic_prices)
    prefix = build_features(synthetic_prices.iloc[:800])

    common = prefix.index.intersection(full.index)
    assert len(common) > 100

    # `target` is deliberately forward-looking, so it is excluded.
    cols = list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        full.loc[common, cols], prefix.loc[common, cols], check_exact=False, rtol=1e-9
    )


def test_target_is_next_period_log_return(features, synthetic_prices):
    """target[t] must equal log_return[t+1], never log_return[t]."""
    close = synthetic_prices["Close"]
    log_returns = np.log(close / close.shift(1))

    idx = features.index[10]
    next_idx = log_returns.index[log_returns.index.get_loc(idx) + 1]

    assert features.loc[idx, "target"] == pytest.approx(log_returns.loc[next_idx])


def test_all_declared_features_present(features):
    for col in FEATURE_COLUMNS:
        assert col in features.columns
    assert "target" in features.columns


def test_features_have_no_nan_or_inf(features):
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy()).all()


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Close"):
        build_features(pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Volume": [1]}))


# ---------------------------------------------------------------- RSI


def test_rsi_bounded_and_finite(synthetic_prices):
    rsi = relative_strength_index(synthetic_prices["Close"], 14).dropna()
    assert rsi.between(0.0, 100.0).all()


def test_rsi_of_monotone_rise_is_one_hundred():
    rising = pd.Series(np.arange(1, 101, dtype=float))
    assert relative_strength_index(rising, 14).iloc[-1] == pytest.approx(100.0)


def test_rsi_of_monotone_fall_is_zero():
    falling = pd.Series(np.arange(100, 0, -1, dtype=float))
    assert relative_strength_index(falling, 14).iloc[-1] == pytest.approx(0.0)


# ---------------------------------------------------------------- sequences


def test_make_sequences_shapes_and_alignment():
    values = np.arange(50, dtype=float).reshape(25, 2)
    targets = np.arange(25, dtype=float)

    x, y = make_sequences(values, targets, sequence_length=5)

    assert x.shape == (21, 5, 2)
    assert y.shape == (21,)
    # Window i ends at row i+4, so its target is targets[i+4].
    np.testing.assert_array_equal(x[0], values[0:5])
    assert y[0] == targets[4]
    np.testing.assert_array_equal(x[-1], values[20:25])
    assert y[-1] == targets[24]


def test_make_sequences_rejects_too_short_input():
    with pytest.raises(ValueError, match="at least"):
        make_sequences(np.zeros((3, 2)), np.zeros(3), sequence_length=10)


# ---------------------------------------------------------------- splits


def test_splits_are_non_empty(features):
    ds = build_dataset(features, sequence_length=60)
    assert len(ds.x_train) > 0
    assert len(ds.x_val) > 0
    assert len(ds.x_test) > 0


def test_a_validation_split_exists(features):
    """REGRESSION: the legacy pipeline had no validation set, so the LR
    scheduler and early stopping both keyed off *training* loss.
    """
    ds = build_dataset(features, sequence_length=60)
    assert len(ds.x_val) > 0
    assert not np.array_equal(ds.x_val, ds.x_test)


def test_train_and_test_windows_never_share_rows(features):
    """REGRESSION: sequences were built before an index split, so the first
    `sequence_length` test windows contained training rows (legacy gbm.py:1288).

    Each window's final row is identifiable by its feature vector; requiring
    disjoint sets of window contents proves the gap works.
    """
    seq = 60
    ds = build_dataset(features, sequence_length=seq)

    def row_keys(x):
        # Hash every row of every window; overlapping windows share row hashes.
        flat = x.reshape(-1, x.shape[-1])
        return {hash(r.tobytes()) for r in flat}

    assert row_keys(ds.x_train).isdisjoint(row_keys(ds.x_test))
    assert row_keys(ds.x_train).isdisjoint(row_keys(ds.x_val))
    assert row_keys(ds.x_val).isdisjoint(row_keys(ds.x_test))


def test_scaler_is_fit_on_training_data_only(features):
    """REGRESSION: legacy called fit_transform on the full array first.

    If the scaler saw only training rows, the training split standardises to
    ~0 mean / ~1 std while later splits generally do not.
    """
    ds = build_dataset(features, sequence_length=60)

    train_rows = ds.x_train.reshape(-1, ds.n_features)
    assert np.abs(train_rows.mean(axis=0)).max() < 0.35
    assert np.abs(train_rows.std(axis=0) - 1.0).max() < 0.45

    # Refitting on everything would give a materially different transform.
    from sklearn.preprocessing import StandardScaler

    all_raw = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    full_scaler = StandardScaler().fit(all_raw)
    assert not np.allclose(full_scaler.mean_, ds.x_scaler.mean_, rtol=1e-6)


def test_splits_are_chronological(features):
    """Training data must precede validation, which must precede test."""
    seq = 60
    ds = build_dataset(features, sequence_length=seq)
    raw = ds.x_scaler.transform(features[list(FEATURE_COLUMNS)].to_numpy(dtype=float))

    def first_row_index(window_stack):
        target = window_stack[0, 0]
        return int(np.argmin(np.abs(raw - target).sum(axis=1)))

    assert first_row_index(ds.x_train) < first_row_index(ds.x_val)
    assert first_row_index(ds.x_val) < first_row_index(ds.x_test)


def test_inverse_transform_recovers_raw_target_scale(features):
    """REGRESSION: legacy trained on normalised targets but evaluated against
    raw ones, so every reported R2/MAE/IC compared mismatched units.
    """
    ds = build_dataset(features, sequence_length=60)
    recovered = ds.inverse_transform_y(ds.y_train)

    # Raw daily log returns are tiny; scaled ones are ~unit variance.
    assert np.abs(recovered).max() < 0.5
    assert ds.y_train.std() == pytest.approx(1.0, abs=0.25)


def test_insufficient_data_raises_actionable_error(features):
    with pytest.raises(ValueError, match="Not enough data"):
        build_dataset(features.iloc[:100], sequence_length=60)


@pytest.mark.parametrize(
    "kwargs", [{"train_frac": 0.9, "val_frac": 0.2}, {"train_frac": 1.5}, {"val_frac": 0.0}]
)
def test_invalid_split_fractions_rejected(features, kwargs):
    with pytest.raises(ValueError):
        build_dataset(features, sequence_length=60, **kwargs)


def test_missing_feature_column_rejected(features):
    with pytest.raises(ValueError, match="missing feature columns"):
        build_dataset(features.drop(columns=["rsi_14"]), sequence_length=60)

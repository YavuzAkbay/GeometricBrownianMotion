"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Six years of daily OHLCV generated from a GBM with known parameters.

    ``mu=0.12`` and ``sigma=0.25`` are the ground truth that
    ``estimate_parameters`` must recover.
    """
    n = 252 * 6
    mu, sigma, s0 = 0.12, 0.25, 100.0
    dt = 1.0 / 252

    gen = np.random.default_rng(7)
    z = gen.standard_normal(n)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    close = s0 * np.exp(np.concatenate([[0.0], np.cumsum(log_increments)]))

    index = pd.bdate_range("2018-01-01", periods=len(close))
    noise = gen.uniform(0.995, 1.005, size=len(close))

    return pd.DataFrame(
        {
            "Open": close * noise,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": gen.integers(1_000_000, 5_000_000, size=len(close)),
        },
        index=index,
    )


@pytest.fixture
def true_params() -> dict[str, float]:
    """Ground-truth parameters behind ``synthetic_prices``."""
    return {"mu": 0.12, "sigma": 0.25, "s0": 100.0}

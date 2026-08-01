"""Data layer tests. yfinance is always mocked; nothing here hits the network."""

from __future__ import annotations

import pandas as pd
import pytest

from gbm.data import DataFetchError, PriceData, fetch


@pytest.fixture
def fake_yf(monkeypatch, synthetic_prices):
    """Install a fake yfinance module and record how it was called."""

    calls: list[dict] = []

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append({"api": "history", "symbol": self.symbol, **kwargs})
            return synthetic_prices.copy()

    class FakeYF:
        frame = synthetic_prices.copy()

        @staticmethod
        def download(ticker, **kwargs):
            calls.append({"api": "download", "ticker": ticker, **kwargs})
            return FakeYF.frame.copy()

        Ticker = FakeTicker

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    FakeYF.calls = calls
    return FakeYF


def test_fetch_returns_validated_price_data(fake_yf, tmp_path):
    data = fetch("aapl", period="5y", cache_dir=tmp_path)

    assert isinstance(data, PriceData)
    assert data.ticker == "AAPL"  # normalised
    assert not data.from_cache
    assert data.latest_price > 0
    assert list(data.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_pins_auto_adjust_and_progress(fake_yf, tmp_path):
    """REGRESSION: legacy left these at yfinance defaults in one module and set
    them in another, so the two computed different prices for one ticker.
    """
    fetch("AAPL", cache_dir=tmp_path)

    call = fake_yf.calls[0]
    assert call["auto_adjust"] is True
    assert call["progress"] is False


def test_second_fetch_hits_the_cache(fake_yf, tmp_path):
    first = fetch("AAPL", cache_dir=tmp_path)
    calls_after_first = len(fake_yf.calls)

    second = fetch("AAPL", cache_dir=tmp_path)

    assert not first.from_cache
    assert second.from_cache
    assert len(fake_yf.calls) == calls_after_first  # no new download
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_force_refresh_bypasses_cache(fake_yf, tmp_path):
    fetch("AAPL", cache_dir=tmp_path)
    before = len(fake_yf.calls)

    data = fetch("AAPL", cache_dir=tmp_path, force_refresh=True)

    assert not data.from_cache
    assert len(fake_yf.calls) > before


def test_cache_can_be_disabled(fake_yf, tmp_path):
    fetch("AAPL", cache_dir=None)
    fetch("AAPL", cache_dir=None)
    assert len(fake_yf.calls) == 2
    assert not any(tmp_path.iterdir())


def test_corrupt_cache_entry_is_ignored(fake_yf, tmp_path):
    fetch("AAPL", cache_dir=tmp_path)
    for path in tmp_path.rglob("*.csv.gz"):
        path.write_bytes(b"not a parquet file")

    data = fetch("AAPL", cache_dir=tmp_path)
    assert not data.from_cache  # silently re-fetched rather than crashing


def test_multiindex_columns_are_flattened(fake_yf, tmp_path, synthetic_prices):
    frame = synthetic_prices.copy()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])
    fake_yf.frame = frame

    data = fetch("AAPL", cache_dir=tmp_path)
    assert list(data.frame.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_empty_download_falls_back_to_history(fake_yf, tmp_path):
    fake_yf.frame = pd.DataFrame()

    data = fetch("AAPL", cache_dir=tmp_path)

    assert len(data) > 0
    assert any(c["api"] == "history" for c in fake_yf.calls)


def test_total_failure_raises_actionable_error(monkeypatch, tmp_path):
    """REGRESSION: legacy returned None here, which callers dereferenced."""

    class DeadYF:
        @staticmethod
        def download(ticker, **kwargs):
            return pd.DataFrame()

        class Ticker:
            def __init__(self, symbol):
                pass

            def history(self, **kwargs):
                return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", DeadYF)

    with pytest.raises(DataFetchError, match="No price data"):
        fetch("NOSUCHTICKER", cache_dir=tmp_path)


def test_missing_columns_raise(fake_yf, tmp_path, synthetic_prices):
    fake_yf.frame = synthetic_prices[["Open", "High"]].copy()

    with pytest.raises(DataFetchError, match="missing required columns"):
        fetch("AAPL", cache_dir=tmp_path)


def test_non_positive_prices_rejected(fake_yf, tmp_path, synthetic_prices):
    """Log returns are undefined; better to fail loudly than emit NaN."""
    frame = synthetic_prices.copy()
    frame.iloc[5, frame.columns.get_loc("Close")] = -1.0
    fake_yf.frame = frame

    with pytest.raises(DataFetchError, match="non-positive"):
        fetch("AAPL", cache_dir=tmp_path)


def test_too_few_rows_rejected(fake_yf, tmp_path, synthetic_prices):
    fake_yf.frame = synthetic_prices.iloc[:1].copy()

    with pytest.raises(DataFetchError, match="at least 2"):
        fetch("AAPL", cache_dir=tmp_path)


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_ticker_rejected(bad, tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        fetch(bad, cache_dir=tmp_path)

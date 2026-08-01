"""Market data fetching with an on-disk cache.

Differences from the legacy ``fetch_and_clean_data``:

* Failures raise :class:`DataFetchError` with an actionable message instead of
  returning ``None`` for callers to dereference.
* ``auto_adjust`` and ``progress`` are pinned explicitly. The legacy code left
  them at yfinance defaults in one module and set them in another, so the two
  computed different prices for the same ticker.
* Responses are cached to disk, keyed by ticker, period and date. The legacy
  code re-downloaded on every call.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

DEFAULT_CACHE_DIR = Path("cache") / "market_data"


class DataFetchError(RuntimeError):
    """Raised when market data cannot be obtained or is unusable."""


@dataclass(frozen=True)
class PriceData:
    """A validated OHLCV frame plus its provenance."""

    ticker: str
    period: str
    frame: pd.DataFrame
    from_cache: bool

    @property
    def close(self) -> pd.Series:
        return self.frame["Close"]

    @property
    def latest_price(self) -> float:
        return float(self.frame["Close"].iloc[-1])

    def __len__(self) -> int:
        return len(self.frame)


def _cache_path(ticker: str, period: str, cache_dir: Path) -> Path:
    """Cache key includes today's date so data refreshes daily.

    Gzipped CSV rather than parquet: parquet needs pyarrow (~100 MB) and would
    otherwise fail silently, and rather than pickle, which would deserialise
    arbitrary objects from disk.
    """
    key = f"{ticker.upper()}|{period}|{date.today().isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / f"{ticker.upper()}_{period}_{digest}.csv.gz"


def _normalise(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten MultiIndex columns, validate, and drop unusable rows."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.loc[:, ~frame.columns.duplicated()]

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataFetchError(
            f"Data for {ticker!r} is missing required columns {missing}. "
            f"Got columns: {list(frame.columns)}"
        )

    frame = frame[list(REQUIRED_COLUMNS)].dropna()

    if frame.empty:
        raise DataFetchError(f"All rows for {ticker!r} were dropped as incomplete.")

    non_positive = (frame["Close"] <= 0).sum()
    if non_positive:
        raise DataFetchError(
            f"{non_positive} non-positive close prices for {ticker!r}; "
            "log returns are undefined."
        )

    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)

    # A DatetimeIndex freq does not survive the CSV cache round-trip, so clear
    # it here to guarantee a cached frame equals a freshly fetched one. Real
    # yfinance responses never carry one anyway.
    if getattr(frame.index, "freq", None) is not None:
        frame.index = frame.index.copy()
        frame.index.freq = None

    return frame


def _download(ticker: str, period: str) -> pd.DataFrame:
    """Fetch from yfinance, falling back to the Ticker.history endpoint.

    The two endpoints fail independently often enough that trying both is
    worthwhile, but a failure of both is reported rather than swallowed.
    """
    import yfinance as yf

    try:
        frame = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as exc:
        log.debug("yf.download raised for %s: %s", ticker, exc)
        frame = pd.DataFrame()

    if frame is not None and not frame.empty:
        return frame

    log.debug("yf.download returned nothing for %s; trying Ticker.history", ticker)
    try:
        frame = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception as exc:
        raise DataFetchError(
            f"Could not fetch data for {ticker!r}.\n"
            f"  - Check the symbol is valid and currently listed.\n"
            f"  - Try: pip install --upgrade yfinance\n"
            f"  - Underlying error: {exc}"
        ) from exc

    if frame is None or frame.empty:
        raise DataFetchError(
            f"No price data returned for {ticker!r} over period {period!r}.\n"
            f"  - Check the symbol is valid and currently listed.\n"
            f"  - Try: pip install --upgrade yfinance"
        )

    return frame


def fetch(
    ticker: str,
    period: str = "5y",
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> PriceData:
    """Fetch daily OHLCV data for a ticker.

    Args:
        ticker: Symbol, e.g. ``"AAPL"``.
        period: yfinance period string, e.g. ``"1y"``, ``"5y"``, ``"max"``.
        cache_dir: Directory for the parquet cache. ``None`` disables caching.
            Created on first write, never on import.
        force_refresh: Bypass any cached copy.

    Returns:
        A validated :class:`PriceData`.

    Raises:
        DataFetchError: If data cannot be fetched or fails validation.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    ticker = ticker.strip().upper()

    path = _cache_path(ticker, period, cache_dir) if cache_dir else None

    if path is not None and path.exists() and not force_refresh:
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            if frame.empty or list(frame.columns) != list(REQUIRED_COLUMNS):
                raise ValueError(f"cache entry has unexpected columns {list(frame.columns)}")
            log.debug("Loaded %s from cache: %s", ticker, path)
            return PriceData(ticker, period, frame, from_cache=True)
        except Exception as exc:
            # A corrupt cache entry must never be fatal; re-fetch instead.
            log.warning("Ignoring unreadable cache entry %s: %s", path, exc)

    log.info("Fetching %s (%s) from Yahoo Finance", ticker, period)
    frame = _normalise(_download(ticker, period), ticker)

    if len(frame) < 2:
        raise DataFetchError(
            f"Only {len(frame)} usable rows for {ticker!r}; need at least 2 "
            "to compute returns."
        )

    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path)
            log.debug("Cached %s to %s", ticker, path)
        except Exception as exc:
            # Caching is an optimisation; a read-only filesystem must not break
            # the analysis.
            log.warning("Could not write cache entry %s: %s", path, exc)

    return PriceData(ticker, period, frame, from_cache=False)

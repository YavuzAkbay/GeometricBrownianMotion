"""Configuration objects.

Every tunable that used to be a magic number scattered through the old scripts
lives here. In particular the risk-free rate, which was hard-coded as ``0.03``
in six separate places, is now a single field with one default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Trading days per year. Used for every annualisation in the package; the old
# code mixed 252 and 365 depending on the call site.
TRADING_DAYS = 252

DEFAULT_RISK_FREE_RATE = 0.03


@dataclass(frozen=True)
class SimConfig:
    """Parameters shared by every path simulator.

    Attributes:
        horizon_years: Simulation horizon T, in years.
        steps: Number of time steps N. ``dt = horizon_years / steps``.
        n_paths: Number of Monte Carlo paths.
        seed: Seed for the local RNG. ``None`` means non-reproducible.
            Simulators build their own ``numpy.random.default_rng`` from this;
            they never touch the global NumPy or torch RNG state.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
    """

    horizon_years: float = 0.5
    steps: int = 126
    n_paths: int = 10_000
    seed: int | None = None
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise ValueError(f"horizon_years must be > 0, got {self.horizon_years}")
        if self.steps <= 0:
            raise ValueError(f"steps must be > 0, got {self.steps}")
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be > 0, got {self.n_paths}")

    @property
    def dt(self) -> float:
        return self.horizon_years / self.steps

    @classmethod
    def from_months(cls, months: int, **kwargs) -> SimConfig:
        """Build a config from a horizon in months, at daily resolution."""
        if months <= 0:
            raise ValueError(f"months must be > 0, got {months}")
        return cls(horizon_years=months / 12.0, steps=months * 21, **kwargs)


@dataclass(frozen=True)
class OptionSpec:
    """A single European option contract."""

    strike: float
    maturity_years: float
    option_type: str = "call"
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.maturity_years < 0:
            raise ValueError(f"maturity_years must be >= 0, got {self.maturity_years}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")


@dataclass(frozen=True)
class RiskConfig:
    """Risk-metric conventions.

    ``confidence_levels`` are tail probabilities: 0.05 means the 5% worst
    outcomes. All returned VaR/CVaR figures follow the sign convention
    documented in :mod:`gbm.risk` — losses are negative.
    """

    confidence_levels: tuple[float, ...] = (0.01, 0.05, 0.10)
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE

    def __post_init__(self) -> None:
        for a in self.confidence_levels:
            if not 0.0 < a < 1.0:
                raise ValueError(f"confidence level must be in (0, 1), got {a}")


@dataclass
class OutputConfig:
    """Where artefacts are written, and how.

    Nothing is created on import — directories are made only when
    :meth:`ensure` is called, which the CLI does and libraries do not.
    """

    root: Path = field(default_factory=lambda: Path("output"))
    dpi: int = 150
    save_plots: bool = True

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> OutputConfig:
        """Create the output tree. Call this explicitly, never at import."""
        for p in (self.plots, self.data, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self

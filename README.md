# Geometric Brownian Motion — Quantitative Modelling Toolkit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/static/v1?label=license&message=GPL-3.0)](LICENSE.TXT)

Monte Carlo equity forecasting, options pricing and risk analysis. Implements
Geometric Brownian Motion alongside Heston stochastic volatility,
regime-switching and Merton jump-diffusion, with Black–Scholes pricing, Greeks,
VaR/CVaR, a portfolio option overlay, and an optional transformer predictor with
calibrated uncertainty.

**Author:** Yavuz Akbay — [GitHub](https://github.com/YavuzAkbay) ·
[LinkedIn](https://www.linkedin.com/in/yavuzakbay/) · akbay.yavuz@gmail.com

---

## Install

```bash
git clone https://github.com/YavuzAkbay/GeometricBrownianMotion.git
cd GeometricBrownianMotion

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e .                  # numerical core + CLI
pip install -e ".[ml]"            # + transformer model (PyTorch)
pip install -e ".[ml,explain]"    # + SHAP / plotly explainability
```

The numerical core — every process, pricer and risk metric — depends only on
NumPy, pandas and SciPy. PyTorch is needed only for the neural predictor.

## Command line

```bash
gbm analyze AAPL --model heston --months 6 --sims 10000 --seed 42
gbm compare AAPL --sims 20000
gbm options AAPL --strike 300 --type put
```

```
AAPL - heston model
Spot price:          $308.91
Estimated drift:     +19.55% annualised
Estimated vol:       27.95% annualised
Horizon:             0.50 years (126 steps, 5,000 paths)

Expected return:     +10.98%
Volatility:          21.32%
Sharpe ratio:        0.630  (annualised, excess of rf)
VaR (5%):            -22.27%
CVaR (5%):           -29.57%
Max drawdown:        -17.31%  (mean over paths)
Probability of gain: 68.3%
```

Common flags — valid before or after the subcommand:

| Flag | Meaning |
|---|---|
| `--model` | `gbm`, `heston`, `regime`, `jump` |
| `--months` | Forecast horizon (default 6) |
| `--sims` | Monte Carlo paths (default 10 000) |
| `--seed` | RNG seed; `-1` for a random run (default 42) |
| `--period` | History window to fit on (default `5y`) |
| `--rate` | Annualised risk-free rate (default 0.03) |
| `--output` | Artefact directory (default `./output`) |
| `--no-plots` | Skip figure generation |
| `--dpi` | Figure resolution (default 150) |
| `-v` / `-q` | More detail / warnings only |

Runs are deterministic: the same `--seed` reproduces byte-identical output.
Market data is cached under `./cache`, keyed by ticker, period and date.

## Library

```python
from gbm import SimConfig, estimate_parameters, simulate_heston, RiskMetrics
from gbm.data import fetch

data = fetch("AAPL", period="5y")
params = estimate_parameters(data.close)

cfg = SimConfig.from_months(6, n_paths=50_000, seed=42)
paths = simulate_heston(
    data.latest_price,
    mu=params.mu, v0=params.sigma**2, kappa=3.0,
    theta=params.sigma**2, sigma_v=0.4, rho=-0.7,
    cfg=cfg,
)

metrics = RiskMetrics.from_paths(paths, horizon_years=cfg.horizon_years)
print("\n".join(metrics.summary_lines()))
```

Importing `gbm` has no side effects: it creates no directories, prints nothing,
and does not import torch.

### Options pricing

```python
from gbm import SimConfig, black_scholes_price, greeks, monte_carlo_price, simulate_gbm

price = black_scholes_price(s=100, k=100, t=1.0, r=0.05, sigma=0.2, option_type="call")
g = greeks(100, 100, 1.0, 0.05, 0.2, "call")
# g.delta, g.gamma, g.vega, g.theta (per day), g.rho

rn_paths = simulate_gbm(100, mu=0.05, sigma=0.2, cfg=SimConfig(seed=1))
mc = monte_carlo_price(rn_paths, strike=100, maturity=1.0, rate=0.05)
print(mc.price, mc.std_error, (mc.ci_low, mc.ci_high))
```

Monte Carlo pricing requires risk-neutral paths — simulate with `mu` set to the
risk-free rate, as above.

### Portfolio with an option overlay

```python
import numpy as np
from gbm import SimConfig
from gbm.portfolio import OptionPosition, analyse_portfolio

result = analyse_portfolio(
    holdings={"AAPL": 100, "MSFT": 50},
    spots={"AAPL": 180.0, "MSFT": 380.0},
    mus={"AAPL": 0.10, "MSFT": 0.09},
    sigmas={"AAPL": 0.28, "MSFT": 0.25},
    correlation=np.array([[1.0, 0.62], [0.62, 1.0]]),
    options=[OptionPosition("AAPL", strike=170, maturity_years=0.5, option_type="put")],
    cfg=SimConfig.from_months(12, seed=42),
)
print(result.var_reduction, result.cvar_reduction)  # positive means safer
```

Assets are ordered by `sorted(holdings)`; the correlation matrix must use that
same order.

### Neural predictor

```python
from gbm.data import fetch
from gbm.features import build_dataset, build_features
from gbm.models import TrainConfig, evaluate, train_model

dataset = build_dataset(build_features(fetch("AAPL").frame), sequence_length=60)
trained = train_model(dataset, TrainConfig(epochs=50, seed=42))
print(evaluate(trained, "test").to_dict())
```

Splits are chronological and separated by a `sequence_length` gap, so no
training window overlaps a validation or test window. Scalers are fit on the
training split only, and metrics are reported in raw target units.

## Conventions

These are applied uniformly and are worth knowing before reading any output.

- **Losses are negative.** A VaR of `-0.18` means the 5% worst outcomes lose at
  least 18%. Lower is worse, so ranking goes through `rank_by_risk()` rather
  than a bare `min`/`sorted`.
- **Drift `mu` is arithmetic**, the one in `E[S_T] = S_0 e^{mu T}`. It is
  estimated from log returns with the Itô correction `mu = m + sigma^2 / 2`.
- **Max drawdown is per path**, a peak-to-trough decline in `[-1, 0]`,
  aggregated across paths by mean (or `q05` for a tail figure).
- **Sharpe is annualised and excess of the risk-free rate**, compounded to the
  horizon: `sqrt(1/T) * (mean(R) - rf_T) / std(R)`.
- **Theta is per calendar day**; vega and rho are per 1.00 of volatility and
  rate respectively.
- Simulators use private `numpy.random.Generator` instances and never touch
  global RNG state.

## Devices

The neural predictor auto-selects CUDA, then Apple MPS, then CPU:

```python
TrainConfig(device="auto")   # or "cpu", "cuda", "mps"
```

An explicitly requested device that is unavailable raises rather than silently
falling back. The numerical core is NumPy and runs on CPU regardless.

## Development

```bash
pip install -e ".[dev,explain]"
pytest                 # no network access required
ruff check src tests
```

CI runs on Python 3.10–3.12 and additionally asserts that importing the package
creates no directories and emits no output, that no module silences warnings
globally, and that there are no bare `except:` clauses.

Tests are property-based where it matters: put–call parity, Greeks against
finite differences, Monte Carlo convergence to Black–Scholes, the Merton
martingale property under the risk-neutral measure, CIR variance positivity,
and causality of every engineered feature.

## Project layout

```
src/gbm/
  config.py      logging.py     backend.py
  data.py        features.py    analysis.py     cli.py
  pricing.py     risk.py        portfolio.py
  explain.py     viz.py
  processes/     gbm · heston · jump · regime
  models/        transformer · train
tests/
```

## Version 2.0

Version 2.0 is a full rewrite. The previous release was three flat scripts
(~8 300 lines) in which every simulator and risk function existed twice — a CPU
copy and a hand-ported GPU copy that had drifted into different discretisation
schemes under the same names.

The rewrite fixed roughly forty correctness defects. The ones that changed
published numbers:

- The "traditional GBM" baseline in every comparison chart used additive Euler
  with no Itô term, so it was biased and could produce negative prices.
- Merton jump-diffusion had no compensator, biasing every jump option price;
  jumps were also capped at one per step and their sizes clamped, truncating
  the tail the model exists to represent.
- Heston stepped the price with `v_{t+1}` rather than `v_t`, corrupting the
  leverage effect, and wrote `v_0` before correcting `theta`.
- The Itô correction was applied to simple rather than log returns, double
  counting drift.
- Model rankings sorted signed CVaR ascending under "lower is better", so the
  riskiest model was reported as rank 1.
- Portfolio option payoffs were written to a column that was never read, so the
  options-contribution report was structurally zero on every run.
- Max drawdown had three mutually incompatible definitions, none measuring a
  peak-to-trough decline; Sharpe had no risk-free rate and no annualisation.
- The volatility head was never in the training loss, yet its untrained output
  was fed into the simulators.
- Train and test windows overlapped by `sequence_length` rows, scalers were fit
  on the full dataset, and metrics compared normalised predictions against raw
  targets.

Also removed: global `warnings.filterwarnings('ignore')` (which hid the
`invalid value in sqrt` warnings these bugs produced), import-time `mkdir` side
effects, eleven `except` blocks that charted fabricated placeholder data, a
`.gitignore` rule that made it impossible to commit tests, and a blocking
`input()` prompt that made the tool un-scriptable.

The previous version remains available at the `pre-rewrite-v1.3` tag.

## Contributing

Issues and pull requests welcome. Please keep `pytest` and `ruff check` green,
and add a test for any behaviour change — regression tests in this repo name
the defect they pin.

## License

GPL-3.0 — see [LICENSE.TXT](LICENSE.TXT).

## Disclaimer

For research and education. Model output is not investment advice. Past
performance does not predict future results.

"""Command-line interface.

Replaces the legacy entry points: ``gbm.py`` hardcoded ``ticker = "XLU"``, and
``enhanced_gbm.py`` blocked on ``input()`` behind two decorative menus whose
selections were never read, making it impossible to script, cron or CI.

Every run is deterministic given ``--seed``, and all output goes through the
logging module so ``--quiet`` and ``--verbose`` work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_RISK_FREE_RATE, OutputConfig
from .logging import configure, get_logger

log = get_logger(__name__)


def _add_verbosity(parser: argparse.ArgumentParser, suppress_defaults: bool = False) -> None:
    """Add -v/--verbose and -q/--quiet.

    Added to both the top-level parser and every subcommand so that
    ``gbm -q analyze AAPL`` and ``gbm analyze AAPL -q`` both work. The
    subcommand copies use SUPPRESS defaults, so an unspecified flag leaves the
    top-level value intact instead of overwriting it with a default.
    """
    default_v = argparse.SUPPRESS if suppress_defaults else 0
    default_q = argparse.SUPPRESS if suppress_defaults else False

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="count", default=default_v,
                       help="More detail; repeat for module names")
    group.add_argument("-q", "--quiet", action="store_true", default=default_q,
                       help="Warnings and errors only")


def _add_common(parser: argparse.ArgumentParser) -> None:
    _add_verbosity(parser, suppress_defaults=True)
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--months", type=int, default=6, help="Forecast horizon (default: 6)")
    parser.add_argument(
        "--sims", type=int, default=10_000, dest="n_paths",
        help="Monte Carlo paths (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed; use --seed -1 for a non-reproducible run (default: 42)",
    )
    parser.add_argument("--period", default="5y", help="History window to fit on (default: 5y)")
    parser.add_argument(
        "--rate", type=float, default=DEFAULT_RISK_FREE_RATE, dest="risk_free_rate",
        help=f"Annualised risk-free rate (default: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output"),
        help="Directory for plots and reports (default: ./output)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gbm",
        description="Geometric Brownian Motion and advanced stochastic models "
                    "for equity forecasting, options pricing and risk analysis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    _add_verbosity(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    analyse_cmd = sub.add_parser("analyze", help="Simulate one model and report risk")
    _add_common(analyse_cmd)
    analyse_cmd.add_argument(
        "--model", default="gbm", choices=["gbm", "heston", "regime", "jump"],
        help="Stochastic model (default: gbm)",
    )

    compare_cmd = sub.add_parser("compare", help="Run every model and rank by risk")
    _add_common(compare_cmd)

    options_cmd = sub.add_parser("options", help="Price an option and show its Greeks")
    _add_common(options_cmd)
    options_cmd.add_argument(
        "--strike", type=float,
        help="Strike price (default: at the money, i.e. the spot price)",
    )
    options_cmd.add_argument(
        "--type", default="call", choices=["call", "put"], dest="option_type",
        help="Option type (default: call)",
    )

    return parser


def _resolve_seed(seed: int) -> int | None:
    return None if seed < 0 else seed


def _run_analyze(args) -> int:
    from .analysis import analyse

    result = analyse(
        args.ticker, model=args.model, months=args.months, n_paths=args.n_paths,
        seed=_resolve_seed(args.seed), period=args.period,
        risk_free_rate=args.risk_free_rate,
    )
    print(result.summary())

    if not args.no_plots:
        _save_analysis_plots(result, args)
    return 0


def _run_compare(args) -> int:
    from .analysis import compare_models

    result = compare_models(
        args.ticker, months=args.months, n_paths=args.n_paths,
        seed=_resolve_seed(args.seed), period=args.period,
        risk_free_rate=args.risk_free_rate,
    )
    print(result.summary())

    if not args.no_plots:
        output = OutputConfig(root=args.output, dpi=args.dpi)
        from .risk import terminal_returns
        from .viz import plot_return_distribution, save_figure

        for name, res in result.results.items():
            fig = plot_return_distribution(
                terminal_returns(res.paths), res.metrics.var_5, res.metrics.cvar_5,
                title=f"{result.ticker} {name}: terminal returns",
            )
            save_figure(fig, f"{result.ticker}_{name}_returns", output)
    return 0


def _run_options(args) -> int:
    from .analysis import analyse
    from .pricing import black_scholes_price, greeks, monte_carlo_price

    result = analyse(
        args.ticker, model="gbm", months=args.months, n_paths=args.n_paths,
        seed=_resolve_seed(args.seed), period=args.period,
        risk_free_rate=args.risk_free_rate,
    )

    strike = args.strike if args.strike is not None else result.spot
    maturity = args.months / 12.0
    rate = args.risk_free_rate
    sigma = result.params.sigma

    analytic = black_scholes_price(result.spot, strike, maturity, rate, sigma, args.option_type)
    g = greeks(result.spot, strike, maturity, rate, sigma, args.option_type)

    # Risk-neutral paths: drift must be the risk-free rate, not the fitted mu.
    from .config import SimConfig
    from .processes import simulate_gbm

    rn_paths = simulate_gbm(
        result.spot, mu=rate, sigma=sigma,
        cfg=SimConfig.from_months(args.months, n_paths=args.n_paths,
                                  seed=_resolve_seed(args.seed)),
    )
    mc = monte_carlo_price(rn_paths, strike, maturity, rate, args.option_type)

    print(
        "\n".join(
            [
                f"{result.ticker} {args.option_type} option",
                f"Spot:                ${result.spot:,.2f}",
                f"Strike:              ${strike:,.2f}",
                f"Maturity:            {maturity:.3f} years",
                f"Implied by history:  {sigma:.2%} volatility",
                f"Risk-free rate:      {rate:.2%}",
                "",
                f"Black-Scholes price: ${analytic:,.4f}",
                f"Monte Carlo price:   ${mc.price:,.4f} "
                f"(+/- {mc.std_error:.4f} se, {mc.n_paths:,} paths)",
                f"95% CI:              [${mc.ci_low:,.4f}, ${mc.ci_high:,.4f}]",
                "",
                "Greeks",
                f"  Delta:  {g.delta:+.4f}   per $1 of spot",
                f"  Gamma:  {g.gamma:+.6f}   per $1 of spot, squared",
                f"  Vega:   {g.vega:+.4f}   per 1.00 of volatility",
                f"  Theta:  {g.theta:+.4f}   per calendar day",
                f"  Rho:    {g.rho:+.4f}   per 1.00 of rate",
            ]
        )
    )
    return 0


def _save_analysis_plots(result, args) -> None:
    from .risk import terminal_returns
    from .viz import plot_paths, plot_return_distribution, save_figure

    output = OutputConfig(root=args.output, dpi=args.dpi)
    stem = f"{result.ticker}_{result.model}"

    save_figure(
        plot_paths(result.paths, title=f"{result.ticker} {result.model}: simulated paths"),
        f"{stem}_paths", output,
    )
    save_figure(
        plot_return_distribution(
            terminal_returns(result.paths), result.metrics.var_5, result.metrics.cvar_5,
            title=f"{result.ticker} {result.model}: terminal returns",
        ),
        f"{stem}_returns", output,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        A process exit code: 0 on success, 1 on a handled error, 130 on Ctrl-C.
    """
    args = build_parser().parse_args(argv)
    configure(verbosity=args.verbose, quiet=args.quiet)

    if not args.no_plots:
        # Selected here, at the entry point, rather than as an import side
        # effect -- forcing Agg at import broke interactive use for importers.
        from .viz import use_headless_backend

        use_headless_backend()

    handlers = {"analyze": _run_analyze, "compare": _run_compare, "options": _run_options}

    from .data import DataFetchError

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        return 130
    except (DataFetchError, ValueError, OSError) as exc:
        # Expected, actionable failures: bad ticker, bad arguments, unwritable
        # output. The traceback is available under -v but is noise by default.
        # Anything else propagates -- an unexpected bug must not be reported as
        # a tidy error message, which is what the legacy catch-all did.
        log.error("%s", exc)
        log.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

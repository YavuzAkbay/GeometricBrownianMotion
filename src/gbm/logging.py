"""Logging for the package.

Replaces the 868 bare ``print()`` calls in the old scripts. Library code emits
through these loggers and never configures handlers; only :func:`configure`
(called by the CLI) attaches one, so importing the package stays silent.
"""

from __future__ import annotations

import logging
import sys

_ROOT = "gbm"


def get_logger(name: str) -> logging.Logger:
    """Return the package logger for a module.

    Args:
        name: Usually ``__name__``. A leading ``gbm.`` is preserved; anything
            else is nested under the package root.
    """
    if name == _ROOT or name.startswith(f"{_ROOT}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")


def configure(verbosity: int = 0, quiet: bool = False, stream=None) -> None:
    """Attach a single stderr handler to the package logger.

    Args:
        verbosity: 0 = INFO, 1 = DEBUG, 2+ = DEBUG with module names.
        quiet: Suppress everything below WARNING. Overrides ``verbosity``.
        stream: Destination stream. Defaults to ``sys.stderr`` so that piping
            stdout to a file yields data, not progress chatter.
    """
    logger = logging.getLogger(_ROOT)

    if quiet:
        level = logging.WARNING
    elif verbosity >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = "%(message)s" if verbosity < 2 else "%(levelname)s %(name)s: %(message)s"

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    # Idempotent: repeated calls (e.g. tests) replace rather than stack handlers.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# Importing the package must not print anything. A NullHandler prevents
# "No handlers could be found" warnings when the CLI never runs.
logging.getLogger(_ROOT).addHandler(logging.NullHandler())

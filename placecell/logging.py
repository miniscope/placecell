"""Logging utilities for placecell."""

import logging

_ROOT = "placecell"


def init_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a logger with rich console output.

    Parameters
    ----------
    name:
        Logger name (e.g. ``__name__``). Automatically namespaced
        under ``placecell.``.
    level:
        Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"

    # Initialise root logger once
    root = logging.getLogger(_ROOT)
    if not root.handlers:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=True)
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
            )
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger

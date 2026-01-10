"""Shared utilities for CLI commands."""

import logging
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for CLI commands."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_template(name: str) -> str:
    """Load HTML template from templates directory."""
    template_path = Path(__file__).parent / "templates" / f"{name}.html"
    return template_path.read_text(encoding="utf-8")

"""Shared utilities for CLI commands."""

from pathlib import Path


def load_template(name: str) -> str:
    """Load HTML template from templates directory."""
    template_path = Path(__file__).parent / "templates" / f"{name}.html"
    return template_path.read_text(encoding="utf-8")

# Configuration file for the Sphinx documentation builder.

from importlib.metadata import version as get_version

project = "placecell"
copyright = "2025, Takuya Sasatani"
author = "t-sasatani"
release = get_version("placecell")
version = ".".join(release.split(".")[:2])
# Shorten dev version: 0.1.1.dev2+g3321b8196.d20260117 -> 0.1.1.dev2+g3321b81
if "+" in release:
    base, local = release.split("+", 1)
    # Extract git hash (g3321b8196) and shorten to 7 chars
    parts = local.split(".")
    git_part = next((p for p in parts if p.startswith("g")), None)
    short_hash = git_part[:8] if git_part else ""  # g + 7 chars
    short_release = f"{base}+{short_hash}" if short_hash else base
else:
    short_release = release

extensions = [
    "myst_parser",
    "sphinx_click",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "html_admonition",
]

myst_fence_as_directive = ["mermaid"]

html_theme = "sphinx_book_theme"
html_title = "placecell"

html_theme_options = {
    "show_toc_level": 2,
    "toc_title": "On this page",
    "repository_url": "https://github.com/miniscope/placecell",
    "use_repository_button": True,
    "show_navbar_depth": 1,
    "announcement": f"Version: v{short_release}",
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "both"

# Autosummary settings
autosummary_generate = True

# Configuration file for the Sphinx documentation builder.

project = "placecell"
copyright = "2025, Takuya Sasatani"
author = "t-sasatani"

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

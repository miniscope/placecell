# Configuration file for the Sphinx documentation builder.

project = "placecell"
copyright = "2025, Takuya Sasatani"
author = "t-sasatani"

extensions = [
    "myst_parser",
    "sphinx_click",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]

html_theme = "sphinx_book_theme"
html_title = "pcell"

autosummary_generate = True

"""Sphinx configuration for the viprodyne documentation."""

from __future__ import annotations

project = "viprodyne"
author = "Henrik Dahl Pinholt"
copyright = "2026, Henrik Dahl Pinholt"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "viprodyne"
html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

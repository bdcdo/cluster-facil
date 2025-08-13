# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Point Sphinx to the source code directory to find the module
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Cluster Fácil'
copyright = '2024, Bruno Dias' # Atualize o ano/autor se necessário
author = 'Bruno Dias'

# Get the version from the package __init__ file
try:
    from cluster_facil import __version__ as version
except ImportError:
    version = '0.1.0' # Fallback version
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    'sphinx.ext.napoleon', # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode', # Add links to source code
    'myst_parser',         # Enable parsing Markdown files (.md)
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'pt_BR'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # Use the Read the Docs theme
html_static_path = ['_static']

# -- Options for MyST parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = 'bysource' # Order members by source code order

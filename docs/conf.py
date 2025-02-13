# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "NiChart_Project"
copyright = "2024, Guray Erus, George Aidinis, Alexander Getka, Kyunglok Baik Spiros Maggioros"
author = " Guray Erus, George Aidinis, Alexander Getka, Kyunglok Baik, Spiros Maggioros"
release = "2024"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_toolbox.sidebar_links",
    "sphinx_toolbox.github",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

github_username = "CBICA"
github_repository = "github.com/CBICA/NiChart_Project"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

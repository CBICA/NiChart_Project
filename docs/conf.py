# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NiChart: Neuroimaging Chart'
copyright = 'https://www.med.upenn.edu/cbica/software-agreement.html'
author = 'Guray Erus, Vishnu Bashyam, Alexander Getka, George Aidinis, Kyunglok Baik, Wu Di, Spiros Maggioros'
release = '0.0.1'
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary", "sphinx.ext.todo","sphinx.ext.autodoc", "sphinx.ext.viewcode","sphinx.ext.intersphinx"]

todo_include_todos = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

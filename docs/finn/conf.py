# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sphinx_rtd_theme  # noqa: F401 (imported for theme availability)
import sys

sys.path.insert(0, os.path.abspath("../../src/"))


# -- Project information -----------------------------------------------------

project = "FINN"
copyright = "2020-2022, Xilinx, 2022-2024, AMD"
author = "Y. Umuroglu and J. Petri-Koenig"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []
extensions.append("sphinx.ext.autodoc")
extensions.append("sphinx.ext.autosectionlabel")

# Prefix section labels with document name to avoid duplicates
# e.g., "concepts:overview" instead of just "overview"
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

master_doc = "index"

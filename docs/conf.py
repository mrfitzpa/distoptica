# -*- coding: utf-8 -*-
# Copyright 2024 Matthew Fitzpatrick.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
r"""Configuration file for the Sphinx documentation builder.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For aborting the python script with a specific error code.
import sys

# For accessing terminal environment variables.
import os

# For running terminal commands.
import subprocess

# For pattern matching.
import re



##################################
## Define classes and functions ##
##################################



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

## Check to see whether distoptica can be imported.
try:
    import distoptica
except:
    print("ERROR: can't import distoptica.")
    sys.exit(1)






## Project information.

project = "distoptica"
copyright = "2024, Matthew Fitzpatrick"
author = "Matthew Fitzpatrick"






## General configuration.

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.extlinks",
              "sphinx.ext.intersphinx",
              "sphinx.ext.todo",
              "sphinx.ext.coverage",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "sphinx_autodoc_typehints",
              "sphinx.ext.githubpages",
              "numpydoc"]



# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]



# Avoid a bunch of warnings when using properties with doc strings in classes.
# see https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False



autosummary_generate = True
autoclass_content = "both"
html_show_sourcelink = False
autodoc_inherit_docstrings = True
set_type_checking_flag = True
add_module_names = False



# For equation numbering by section.
numfig = True
math_numfig = True
numfig_secnum_depth = 6



# Cross links to other sphinx documentation websites.
intersphinx_mapping = \
    {"python": ("https://docs.python.org/3", None),
     "numpy": ("https://numpy.org/doc/stable", None),
     "torch": ("http://pytorch.org/docs/stable/", None),
     "fancytypes": ("https://mrfitzpa.github.io/fancytypes", None)}



# External links.
extlinks = {"arxiv": ("https://arxiv.org/abs/%s", "arXiv:%s"),
            "doi": ("https://dx.doi.org/%s", "doi:%s")}



## Options for HTML output.

html_theme = "sphinx_rtd_theme"
html_theme_options = {"display_version": False}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]



# If not "", a "Last updated on:" timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"



# Override some CSS settings.
html_css_files = ["readthedocs_custom.css"]



# Adapted from
# ``https://github.com/ThoSe1990/SphinxExample/blob/main/docs/conf.py``.
build_all_docs = os.environ.get("build_all_docs", None)
pages_root = os.environ.get("pages_root", "")

if build_all_docs is not None:
    current_language = os.environ.get("current_language")
    current_version = os.environ.get("current_version")

    html_context = {"current_language" : current_language,
                    "languages" : [],
                    "current_version" : current_version,
                    "versions" : []}

    if (current_version == "latest"):
        html_context["languages"].append(["en", pages_root])

    if (current_language == "en"):
        html_context["versions"].append(["latest", pages_root])

    cmd_output_as_bytes = subprocess.check_output("git tag", shell=True)
    cmd_output = cmd_output_as_bytes.decode("utf-8")
    tags = cmd_output.rstrip("\n").split("\n")

    pattern = r"v[0-9]+\.[0.9]+\.[0-9]+"
    release_tags = tuple(tag for tag in tags if re.fullmatch(pattern, tag))

    if (current_version != "latest"):
        language = "en"
        path = pages_root+"/"+current_version+"/"+language
        html_context["languages"].append([language, path])

    for tag in release_tags:
        version = tag[1:]
        path = pages_root+"/"+version+"/"+current_language
        html_context["versions"].append([version, path])

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
r"""Build the static documentation website files.

To generate the documentation, you will also need to install several other
packages. This can be done by running the following command from the root of the
repository::

  pip install .[docs]

Next, assuming that you are in the root of the repository, and that you have
installed all the prerequisite packages, you can generate the static
documentation website files by issuing the following commands::

  cd docs
  python build_docs

The generated static documentation files will be stored in ``<root>/pages``,
where ``<root>`` is the root of the repository.

The code below was adopted from
``https://github.com/ThoSe1990/SphinxExample/blob/main/docs/build_docs.py``.

"""



#####################################
## Load libraries/packages/modules ##
#####################################

# For running terminal commands.
import subprocess

# For storing terminal environment variables.
import os

# For pattern matching.
import re



##################################
## Define classes and functions ##
##################################

def _build_doc(version, language, tag):
    os.environ["current_version"] = version
    os.environ["current_language"] = language
    
    subprocess.run("git checkout " + tag, shell=True)
    subprocess.run("git checkout main -- conf.py", shell=True)
    subprocess.run("git checkout main -- _templates/versions.html", shell=True)

    subprocess.run("cd ..; pip install .; cd docs", shell=True)
    
    os.environ["SPHINXOPTS"] = "-D language='{}'".format(language)
    subprocess.run("make html", shell=True)

    subprocess.run("git checkout -f main", shell=True)

    return None

def _mvdir(src, dst):
    subprocess.run(["mkdir", "-p", dst])
    subprocess.run("mv "+src+"* "+dst, shell=True)

    return None



###########################
## Define error messages ##
###########################



#########################
## Main body of script ##
#########################

os.environ["build_all_docs"] = str(True)
os.environ["pages_root"] = "https://mrfitzpa.github.io/distoptica" 

_build_doc("latest", "en", "main")
_mvdir("./_build/html/", "./pages/")

cmd_output_as_bytes = subprocess.check_output("git tag", shell=True)
cmd_output = cmd_output_as_bytes.decode("utf-8")
tags = cmd_output.rstrip("\n").split("\n")

pattern = r"v[0-9]+\.[0.9]+\.[0-9]+"
release_tags = tuple(tag for tag in tags if re.fullmatch(pattern, tag))

for tag in release_tags:
    version = tag[1:]
    language = "en"
    _build_doc(version, language, tag)
    _mvdir("./_build/html/", "./pages/"+version+"/"+language+"/")

subprocess.run("cd ..; pip install .; cd docs", shell=True)
_mvdir("./pages/", "../pages/")

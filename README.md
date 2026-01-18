# DistOptica

[![Test library](https://github.com/mrfitzpa/distoptica/actions/workflows/test_library.yml/badge.svg)](https://github.com/mrfitzpa/distoptica/actions/workflows/test_library.yml)
[![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mrfitzpa/e3d00c6ff78c39d52c8b3f1ca5da9065/raw/distoptica_coverage_badge.json)](https://github.com/mrfitzpa/distoptica/actions/workflows/measure_code_coverage.yml)
[![Documentation](https://img.shields.io/badge/docs-read-brightgreen)](https://mrfitzpa.github.io/distoptica)
[![PyPi Version](https://img.shields.io/pypi/v/distoptica.svg)](https://pypi.org/project/distoptica)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/distoptica.svg)](https://anaconda.org/conda-forge/distoptica)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18286791-blue.svg)](https://doi.org/10.5281/zenodo.18286791)

`distoptica` is a Python library for modelling optical distortions.

Visit the `distoptica` [website](https://mrfitzpa.github.io/distoptica) for a
web version of the installation instructions, the reference guide, and the
examples archive.

The source code can be found in the [`distoptica` GitHub
repository](https://github.com/mrfitzpa/distoptica).



## Table of contents

- [Instructions for installing and uninstalling
  `distoptica`](#instructions-for-installing-and-uninstalling-distoptica)
  - [Installing `distoptica`](#installing-distoptica)
    - [Installing `distoptica` using `pip`](#installing-distoptica-using-pip)
    - [Installing `distoptica` using
      `conda`](#installing-distoptica-using-conda)
  - [Uninstalling `distoptica`](#uninstalling-distoptica)
- [Learning how to use `distoptica`](#learning-how-to-use-distoptica)



## Instructions for installing and uninstalling `distoptica`



### Installing `distoptica`

For all installation scenarios, first open up the appropriate command line
interface. On Unix-based systems, you could open e.g. a terminal. On Windows
systems you could open e.g. an Anaconda Prompt as an administrator.

Before installing `distoptica`, it is recommended that users install `PyTorch`
in the same environment that they intend to install `distoptica` according to
the instructions given [here](https://pytorch.org/get-started/locally/) for
their preferred PyTorch installation option.



#### Installing `distoptica` using `pip`

Before installing `distoptica`, make sure that you have activated the (virtual)
environment in which you intend to install said package. After which, simply run
the following command:

    pip install distoptica

The above command will install the latest stable version of `distoptica`.

To install the latest development version from the main branch of the
[distoptica GitHub repository](https://github.com/mrfitzpa/distoptica), one must
first clone the repository by running the following command:

    git clone https://github.com/mrfitzpa/distoptica.git

Next, change into the root of the cloned repository, and then run the following
command:

    pip install .

Note that you must include the period as well. The above command executes a
standard installation of `distoptica`.

Optionally, for additional features in `distoptica`, one can install additional
dependencies upon installing `distoptica`. To install a subset of additional
dependencies (along with the standard installation), run the following command
from the root of the repository:

    pip install .[<selector>]

where `<selector>` can be one of the following:

* `tests`: to install the dependencies necessary for running unit tests;
* `examples`: to install the dependencies necessary for executing files stored
  in `<root>/examples`, where `<root>` is the root of the repository;
* `docs`: to install the dependencies necessary for documentation generation;
* `all`: to install all of the above optional dependencies.

Alternatively, one can run:

    pip install distoptica[<selector>]

elsewhere in order to install the latest stable version of `distoptica`, along
with the subset of additional dependencies specified by `<selector>`. 



#### Installing `distoptica` using `conda`

Before proceeding, make sure that you have activated the (virtual) `conda`
environment in which you intend to install said package. For Windows systems,
users must install `PyTorch` separately prior to following the remaining
instructions below.

To install `distoptica` using the `conda` package manager, run the following
command:

    conda install -c conda-forge distoptica

The above command will install the latest stable version of `distoptica`.



### Uninstalling `distoptica`

If `distoptica` was installed using `pip`, then to uninstall, run the following
command:

    pip uninstall distoptica

If `distoptica` was installed using `conda`, then to uninstall, run the
following command:

    conda remove distoptica



## Learning how to use `distoptica`

For those new to the `distoptica` library, it is recommended that they take a
look at the [Examples](https://mrfitzpa.github.io/distoptica/examples.html)
page, which contain code examples that show how one can use the `distoptica`
library. While going through the examples, readers can consult the [distoptica
reference
guide](https://mrfitzpa.github.io/distoptica/_autosummary/distoptica.html) to
understand what each line of code is doing.
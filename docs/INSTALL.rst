.. _installation_instructions_sec:

Installation instructions
=========================

Installing distoptica
---------------------

For all installation scenarios, first open up the appropriate command line
interface. On Unix-based systems, you would open a terminal. On Windows systems
you would open an Anaconda Prompt as an administrator.

Installing distoptica using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install ``distoptica`` using ``pip`` is to run the following
command::

  pip install distoptica

The above command will install the latest stable version of ``distoptica``.

To install the latest development version from the main branch of the
`distoptica GitHub repository <https://github.com/mrfitzpa/distoptica>`_, one
must first clone the repository by running the following command::

  git clone https://github.com/mrfitzpa/distoptica.git

Next, change into the root of the cloned repository, and then run the following
command::

  pip install .

Note that you must include the period as well. The above command executes a
standard installation of ``distoptica``.

Optionally, for additional features in ``distoptica``, one can install additional
dependencies upon installing ``distoptica``. To install a subset of additional
dependencies (along with the standard installation), run the following command
from the root of the repository::

  pip install .[<selector>]

where ``<selector>`` can be one of the following:

* ``tests``: to install the dependencies necessary for running unit tests;
* ``examples``: to install the dependencies necessary for running the jupyter
  notebooks stored in ``<root>/examples``, where ``<root>`` is the root of the
  repository;
* ``docs``: to install the dependencies necessary for documentation generation;
* ``all``: to install all of the above optional dependencies.

Installing distoptica using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install ``distoptica`` using the ``conda`` package manager, run the following
command::

  conda install -c conda-forge distoptica

The above command will install the latest stable version of ``distoptica``.

Uninstalling distoptica
-----------------------

If ``distoptica`` was installed using ``pip``, then to uninstall, run the
following command from the root of the repository::

  pip uninstall distoptica

If ``distoptica`` was installed using ``conda``, then to uninstall, run the
following command from the root of the repository::

  conda remove distoptica

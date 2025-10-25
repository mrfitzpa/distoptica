#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright 2025 Matthew Fitzpatrick.
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



# The current script will run all of the unit tests of `distoptica` and measure
# the code coverage resulting from running said unit tests.
#
# To run this script correctly, first you must create or activate an environment
# that has `distoptica` installed, along with all of the dependencies requried
# to run the unit tests and measure code coverage. See the root-level README for
# installation instructions.



cd tests
COVERAGE_FILE=".coverage"
python -m pytest --cov --cov-config=../.coveragerc -vv
python -m coverage json
python -m coverage report --fail-under=100
unset COVERAGE_FILE
cd ..

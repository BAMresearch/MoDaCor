.. SPDX-License-Identifier: BSD-3-Clause
.. /usr/bin/env python3
.. -*- coding: utf-8 -*-
..
.. from __future__ import annotations
..
.. __coding__ = "utf-8"
.. __authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
.. __copyright__ = "Copyright 2026, The MoDaCor team"
.. __date__ = "20/01/2025"
.. __status__ = "Development"  # "Development", "Production"
.. end of header and standard imports
..
========
Overview
========

New modular data corrections for any neutron or X-ray technique that produces 1D or 2D scattering/diffraction/imaging data.

.. start-badges

| |version| |commits-since| |license|
| |supported-versions| |wheel| |downloads|
| |cicd| |coverage|

.. |version| image:: https://img.shields.io/pypi/v/modacor.svg
    :target: https://test.pypi.org/project/modacor
    :alt: PyPI Package latest release

.. |commits-since| image:: https://img.shields.io/github/commits-since/BAMresearch/modacor/v1.0.0.svg
    :target: https://github.com/BAMresearch/modacor/compare/v1.0.0...main
    :alt: Commits since latest release

.. |license| image:: https://img.shields.io/pypi/l/modacor.svg
    :target: https://en.wikipedia.org/wiki/MIT_license
    :alt: License

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/modacor.svg
    :target: https://test.pypi.org/project/modacor
    :alt: Supported versions

.. |wheel| image:: https://img.shields.io/pypi/wheel/modacor.svg
    :target: https://test.pypi.org/project/modacor#files
    :alt: PyPI Wheel

.. |downloads| image:: https://img.shields.io/pypi/dw/modacor.svg
    :target: https://test.pypi.org/project/modacor/
    :alt: Weekly PyPI downloads

.. |cicd| image:: https://github.com/BAMresearch/modacor/actions/workflows/ci-cd.yml/badge.svg
    :target: https://github.com/BAMresearch/modacor/actions/workflows/ci-cd.yml
    :alt: Continuous Integration and Deployment Status

.. |coverage| image:: https://img.shields.io/endpoint?url=https://BAMresearch.github.io/modacor/coverage-report/cov.json
    :target: https://BAMresearch.github.io/modacor/coverage-report/
    :alt: Coverage report

.. end-badges

Summary
=======

MoDaCor is a library for traceable, stepwise data corrections, with propagation of units as well as (multiple) uncertainties. The data sources can be files or data streams. The correction follows an optionally branching and merging graph that is to be configured per application. The computational steps can reuse pre-calculated information calculated in the first run to minimize unnecessary overhead. The modular approach allows very detailed introspection into the effect of each step on the data, units and uncertainties.

This is intended to take on modular data correction workflow tasks as described in this work: https://doi.org/10.1107/S1600576717015096
It can be used either directly, or used as a standard to check faster, more integrated data correction implementations against.

Installation
============

::

    pip install modacor

You can also install the in-development version with::

    pip install git+https://github.com/BAMresearch/modacor.git@main


Documentation
=============

The full documentation is published at https://BAMresearch.github.io/modacor and is
organised around three audiences:

- **Getting started** – a Quickstart walkthrough that runs a sample pipeline with the
  bundled MOUSE dataset and highlights pipeline tracing.
- **Pipeline operations** – guidance for building and maintaining YAML pipeline
  configurations, understanding branching/merging graphs, and using tracing exports.
- **Extension development** – tutorials for creating new correction modules, tests, and
  IO sources, plus an auto-generated reference page per module.

Application-specific walkthroughs (e.g. MOUSE and SAXSess instruments) show how to
combine these resources for real experiments, with links to the corresponding example
pipelines and metadata files.

Planned documentation stubs now live in the repository to signal the intended structure:

- ``docs/getting_started/`` – Quickstart guides (populated with ``quickstart.rst``).
- ``docs/pipeline_operations/`` – pipeline basics, configuration reference, and tracing/debugging placeholders.
- ``docs/extending/`` – module-author guide, IO extension notes, and contribution checklist placeholders.
- ``docs/examples/`` – instrument-specific walkthrough placeholders for MOUSE and SAXSess pipelines.

These pages currently contain explicit TODO notes and will be expanded during the documentation refresh.

Application Examples
====================

MoDaCor ships with two end-to-end instrument walkthroughs:

- **MOUSE beamline** – demonstrates a transmission-scattering workflow, using
  `tests/testdata/MOUSE_20250324_1_160_stacked.nxs` plus static metadata from
  `src/modacor/tests/io/yaml/static_data_example.yaml`.
- **SAXSess laboratory instrument** – showcases parallel branches for intensity and
  calibration corrections, based on the design captured in the project blog posts and
  accompanying diagrams in `docs/full_flow.*`.

Each example documents the relevant pipeline configuration, expected inputs, and
resulting data products.

Development
===========

For coding contributions, we strongly recommend:
  - using flake8 and/or black for consistent formatting.
  - writing tests for every added functionality -> towards test-driven coding practices.

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

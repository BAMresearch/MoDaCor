========
Overview
========

new modular data corrections for any neutron or xray technique that produces 1D or 2D scattering/diffraction/imaging
data

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

https://BAMresearch.github.io/modacor

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

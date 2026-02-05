# MoDaCor (v1.0.0)

## Overview

New modular data corrections for any neutron or X-ray technique that produces 1D or 2D scattering/diffraction/imaging
data.

[![PyPI Package latest release](https://img.shields.io/pypi/v/modacor.svg)](https://pypi.org/project/modacor)
[![Commits since latest release](https://img.shields.io/github/commits-since/BAMresearch/MoDaCor/v1.0.0.svg)](https://github.com/BAMresearch/MoDaCor/compare/v1.0.0...main)
[![License](https://img.shields.io/pypi/l/modacor.svg)](https://en.wikipedia.org/wiki/BSD-3-Clause)
[![Supported versions](https://img.shields.io/pypi/pyversions/modacor.svg)](https://pypi.org/project/modacor)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/modacor.svg)](https://pypi.org/project/modacor#files)
[![Weekly PyPI downloads](https://img.shields.io/pypi/dw/modacor.svg)](https://pypi.org/project/modacor/)
[![Continuous Integration and Deployment Status](https://github.com/BAMresearch/MoDaCor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/BAMresearch/MoDaCor/actions/workflows/ci-cd.yml)
[![Coverage report](https://img.shields.io/endpoint?url=https://BAMresearch.github.io/MoDaCor/coverage-report/cov.json)](https://BAMresearch.github.io/MoDaCor/coverage-report/)

## Summary

MoDaCor is a library for traceable, stepwise data corrections, with propagation of units as well as (multiple)
uncertainties. The data sources can be files or data streams. The correction follows an optionally branching and merging
graph that is configured per application. The computational steps can reuse pre-calculated information produced during a
first run to minimise unnecessary overhead. The modular approach allows detailed introspection into the effect of each
step on the data, units, and uncertainties.

This implements modular data correction workflow concepts discussed in
[https://doi.org/10.1107/S1600576717015096](https://doi.org/10.1107/S1600576717015096). It can be used either directly
or as a reference to validate faster, more integrated data-correction implementations.

## Installation

    pip install modacor

### Optional extras

Support for the Tiled-backed IoSource is provided via an optional dependency set. Install it with:

```bash
pip install modacor[tiled]
```

To install the in-development version:

    pip install git+https://github.com/BAMresearch/MoDaCor.git@main

## Documentation structure

The published documentation at <https://BAMresearch.github.io/MoDaCor> is organised around three main tracks:

- **Getting started** – a Quickstart walkthrough that runs a sample pipeline with the bundled MOUSE dataset and
  highlights pipeline tracing.
- **Pipeline operations** – guidance for building and maintaining YAML pipeline configurations, understanding
  branching/merging graphs, and using tracing exports.
- **Extension development** – tutorials for creating new correction modules, tests, and IO sources, plus an
  auto-generated reference page per module.

Instrument-specific walkthroughs (for example the MOUSE and SAXSess instruments) show how to combine these resources for
real experiments, with links to example pipelines and metadata files.

Planned documentation stubs now live in the repository to signal the intended structure:

- `docs/getting_started/` – Quickstart guides (populated with `quickstart.md`).
- `docs/pipeline_operations/` – pipeline basics, configuration reference, and tracing/debugging placeholders.
- `docs/extending/` – module-author guide, IO extension notes, and contribution checklist placeholders.
- `docs/examples/` – instrument-specific walkthrough placeholders for MOUSE and SAXSess pipelines.

These pages currently contain explicit TODO notes and will be expanded during the documentation refresh.

Documentation contributors can follow `docs/README.md` for local build instructions and authoring notes.

## Development

For coding contributions, we strongly recommend:

- using `flake8` and/or `black` for consistent formatting;
- writing tests for every added functionality to encourage test-driven development practices.

### Testing

See which tests are available (arguments after `--` get passed to *pytest* which runs the tests):

    tox -e py -- --co

Run a specific test only:

    tox -e py -- -k <test_name from listing before>

Run all tests with:

    tox -e py

### Package Version

Get the next version number and how the GIT history would be interpreted for that:

    pip install python-semantic-release
    semantic-release -v version --print

This prints its interpretation of the commits in detail. Make sure to supply the `--print`
argument to not raise the version number which is done automatically by the *release* job
of the GitHub Action Workflows.

### Project template

Update the project configuration from the *copier* template and make sure the required packages
are installed:

    pip install copier jinja2-time
    copier update --trust --skip-answered


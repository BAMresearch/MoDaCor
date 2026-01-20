<!--
SPDX-License-Identifier: BSD-3-Clause
/usr/bin/env python3
-*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports
-->

# Overview

New modular data corrections for any neutron or X-ray technique that produces 1D or 2D scattering/diffraction/imaging
data.

[![PyPI Package latest release](https://img.shields.io/pypi/v/modacor.svg)](https://test.pypi.org/project/modacor)<!-- -->
[![Commits since latest release](https://img.shields.io/github/commits-since/BAMresearch/modacor/v1.0.0.svg)](https://github.com/BAMresearch/modacor/compare/v1.0.0...main)<!-- -->
[![License](https://img.shields.io/pypi/l/modacor.svg)](https://en.wikipedia.org/wiki/MIT_license)<!-- -->
[![Supported Python versions](https://img.shields.io/pypi/pyversions/modacor.svg)](https://test.pypi.org/project/modacor)<!-- -->
[![PyPI Wheel](https://img.shields.io/pypi/wheel/modacor.svg)](https://test.pypi.org/project/modacor#files)<!-- -->
[![Weekly PyPI downloads](https://img.shields.io/pypi/dw/modacor.svg)](https://test.pypi.org/project/modacor/)<!-- -->
[![CI/CD status](https://github.com/BAMresearch/modacor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/BAMresearch/modacor/actions/workflows/ci-cd.yml)<!-- -->
[![Coverage report](https://img.shields.io/endpoint?url=https://BAMresearch.github.io/modacor/coverage-report/cov.json)](https://BAMresearch.github.io/modacor/coverage-report/)

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

```bash
pip install modacor
```

To install the in-development version:

```bash
pip install git+https://github.com/BAMresearch/modacor.git@main
```

## Documentation structure

The published documentation at <https://BAMresearch.github.io/modacor> is organised around three main tracks:

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

## Development

For coding contributions, we strongly recommend:

- using `flake8` and/or `black` for consistent formatting;
- writing tests for every added functionality to encourage test-driven development practices.

To run all the tests:

```bash
tox
```

To combine coverage data from all `tox` environments:

- **Windows**

  ```cmd
  set PYTEST_ADDOPTS=--cov-append
  tox
  ```

- **Other platforms**

  ```bash
  PYTEST_ADDOPTS=--cov-append tox
  ```

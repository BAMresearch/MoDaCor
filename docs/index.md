# MoDaCor documentation

MoDaCor provides reference-quality, traceable corrections for monochromatic
X-ray and neutron scattering, diffraction, and imaging, with explicit physical
units and multiple named uncertainty contributions.

## Choose a route

- **First run:** [install MoDaCor](getting-started/installation.md) and complete
  the [synthetic Quickstart](getting-started/quickstart.md).
- **Pipeline author:** start with the [data model](data-model/index.md), then
  learn the [processing framework](processing/index.md).
- **Existing pipeline author:** review the
  [breaking interface migrations](processing/interface-migrations.md) before
  running older YAML with the current release.
- **Facility or service operator:** read the
  [client-server architecture](server/architecture.md) and
  [installation guidance](server/installation-and-deployment.md).
- **Contributor:** use the [development guides](development/index.md) and
  current [design records](development/design/index.md).

```{toctree}
:maxdepth: 2
:caption: Learn and use

introduction/index
getting-started/index
data-model/index
processing/index
modules/index
server/index
examples/index
reference/index
development/index
```

```{toctree}
:maxdepth: 1
:caption: Project

project/index
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

# Documentation architecture

Status: approved on 2026-09-28

This document defines the target information architecture for MoDaCor's
documentation. It is a structural design, not an implementation plan. The
structure and the decisions at the end of this document should be agreed before
pages are moved, rewritten, or generated differently.

## Goals

The documentation should:

- give a new user a short, reliable path from installation to a successful
  correction run;
- explain why MoDaCor exists before introducing its classes and configuration;
- make units, multiple named uncertainties, traceability, and reproducibility
  visible as core design concerns rather than incidental API details;
- separate concepts, task-oriented instructions, examples, reference material,
  and contributor documentation;
- cover local execution and client-server operation as equally supported ways
  to use the same processing model;
- provide an exhaustive generated module catalogue while reserving hand-written
  module guides for workflows, scientific assumptions, and interactions that
  cannot be communicated well by generated field tables;
- keep instrument-specific examples and their data outside the package
  repository, in the dedicated
  [MoDaCor-examples repository](https://github.com/BAMResearch/MoDaCor-examples);
- make it obvious where a new page belongs and what must be updated when a new
  module, I/O backend, server operation, or example is added; and
- detect broken examples, stale generated pages, and navigation problems in
  automated checks.

## Non-goals

This reorganization should not:

- change runtime behavior or public Python, CLI, or HTTP APIs;
- turn the main repository into an instrument-data or notebook repository;
- duplicate the generated module reference in hand-written prose;
- publish active implementation backlogs as if they were user contracts; or
- promise high-throughput behavior that is not part of MoDaCor's quality-first
  design.

## Current-state assessment

The existing documentation contains a substantial amount of useful material,
but its navigation no longer represents that content accurately.

- The root README is longer than it needs to be, includes contributor detail,
  and still describes populated documentation sections as placeholders.
- The current Quickstart depends on a large external MOUSE file that is not part
  of the repository. It is therefore not a reliable first-run experience.
- The data hierarchy is embedded in `Pipeline Basics` and extension guidance;
  it is not available as a focused, user-facing explanation.
- Local pipeline operation, conceptual framework material, server operation,
  an API design contract, and an internal backlog are mixed under `Pipeline
  operations`.
- Server-side coverage is stronger than the navigation suggests, but Python
  client usage, buffers, chunked outputs, live output, security boundaries, and
  the REST reference are not separated by task.
- The generated module pages provide useful configuration tables, but the
  catalogue is a flat alphabetical list. Generated pages currently expose
  machine-specific absolute source paths, and the checked-in index can drift
  from the public module registry.
- Only capillary self-absorption has a dedicated scientific guide. Cross-module
  subjects such as masking, uncertainty handling, geometry, integration, and
  output need curated guides.
- Instrument examples are described in the package documentation, but the
  canonical notebooks, pipeline YAML, manifests, and data acquisition workflow
  live in the separate examples repository. The relationship is not prominent
  enough.
- Design records, completed implementation plans, contributor instructions,
  and user-facing contracts are spread across several top-level sections.
- The current Sphinx build succeeds with warnings treated as errors, which is a
  good baseline, but that check does not detect stale generated output or prove
  that Quickstart code executes.

## Documentation model

The top-level navigation should follow the reader's question rather than the
package layout.

| Section | Reader question | Primary content type |
| --- | --- | --- |
| Introduction | What is MoDaCor, and is it suitable for my work? | Explanation |
| Getting started | How do I install it and obtain a first result? | Tutorial |
| Data model | What does MoDaCor store and preserve? | Concepts |
| Processing framework | How are corrections configured and run? | Concepts and how-to |
| Modules | Which operations exist, and how should related ones be combined? | Catalogue and guides |
| Client-server operation | How do I run and control the service? | Architecture and operations |
| Examples | What do complete workflows look like? | Tutorials and external examples |
| Reference | What is the exact interface or schema? | Generated and concise reference |
| Development | How do I contribute or extend MoDaCor? | Contributor guidance and design records |

Reference-quality science and traceability are relevant to every section, not a
single isolated page. Pages should consistently show units, uncertainty names,
data paths, provenance, and the distinction between measured and derived data.

## Proposed navigation and file structure

The tree below is the target logical and physical structure. Names are chosen to
remain stable as individual areas grow. An `index.md` is a short landing page,
not a second copy of its child pages.

```text
README.md
docs/
├── index.md
├── introduction/
│   ├── index.md
│   ├── purpose-and-scope.md
│   ├── scientific-principles.md
│   └── terminology.md
├── getting-started/
│   ├── index.md
│   ├── installation.md
│   ├── quickstart.md
│   └── where-to-go-next.md
├── data-model/
│   ├── index.md
│   ├── basedata.md
│   ├── databundle.md
│   ├── processingdata.md
│   ├── units-and-dimensionality.md
│   ├── uncertainty-propagation.md
│   └── paths-axes-weights-and-masks.md
├── processing/
│   ├── index.md
│   ├── process-steps.md
│   ├── pipeline-graphs.md
│   ├── pipeline-configuration.md
│   ├── sources.md
│   ├── sinks.md
│   ├── local-execution.md
│   ├── dependencies-and-partial-runs.md
│   ├── tracing-and-provenance.md
│   └── chunked-processing.md
├── modules/
│   ├── index.md
│   ├── choosing-modules.md
│   ├── masking.md
│   ├── uncertainties.md
│   ├── arithmetic-and-normalization.md
│   ├── geometry.md
│   ├── integration-and-reduction.md
│   ├── scattering-corrections.md
│   └── capillary-self-absorption.md
├── server/
│   ├── index.md
│   ├── architecture.md
│   ├── installation-and-deployment.md
│   ├── clients-and-sessions.md
│   ├── operations.md
│   ├── buffers-and-chunked-outputs.md
│   ├── live-output-and-plots.md
│   ├── custom-steps-and-io.md
│   └── security-and-runtime-policy.md
├── examples/
│   ├── index.md
│   ├── local-operation.md
│   └── remote-server-operation.md
├── reference/
│   ├── index.md
│   ├── modules/
│   │   ├── index.md
│   │   └── <generated ProcessStep pages>.md
│   ├── python-api/
│   │   ├── index.md
│   │   ├── data-model.md
│   │   ├── pipeline-and-runner.md
│   │   ├── io.md
│   │   └── client.md
│   ├── cli.md
│   ├── pipeline-schema.md
│   ├── io-capabilities.md
│   ├── server-api.md
│   ├── runtime-service-openapi.yaml
│   └── glossary.md
├── development/
│   ├── index.md
│   ├── contributing.md
│   ├── development-principles.md
│   ├── architecture.md
│   ├── software-guidelines.md
│   ├── module-author-guide.md
│   ├── io-source-sink-guide.md
│   ├── contribution-checklist.md
│   ├── documentation-guide.md
│   ├── testing-and-release.md
│   └── design/
│       ├── index.md
│       ├── <active design records>.md
│       └── completed/
│           ├── index.md
│           └── <completed design records>.md
└── project/
    ├── index.md
    ├── changelog.md
    ├── authors.md
    ├── citation.md
    └── license.md
```

The `Project` pages may be linked from a compact footer or secondary navigation
instead of occupying the same visual prominence as the user journeys.

## Section contracts

### Landing page

`docs/index.md` should state the one-sentence value proposition and offer four
short entry routes:

- new user: install and run the synthetic Quickstart;
- pipeline author: understand the data model and processing framework;
- facility/operator user: configure the client-server runtime; and
- contributor: extend modules or I/O and read design records.

It should not reproduce the README, expose an internal backlog, or begin with a
large undifferentiated toctree.

### Introduction

The introduction establishes intent before API mechanics. It should cover:

- reference-quality, stepwise corrections;
- traceable signals and explicit physical quantities;
- preservation and propagation of multiple named uncertainty contributions;
- detailed bookkeeping for auditability and reproducibility;
- preference for correction quality and inspectability over maximum throughput;
- use as either the primary correction system or a reference implementation for
  validating faster, instrument-specific implementations;
- current focus on monochromatic X-ray and neutron scattering, diffraction, and
  imaging; and
- explicit non-goals and limitations, including the boundary between MoDaCor
  and acquisition/control software.

`scientific-principles.md` explains the quality model without teaching the class
API. `terminology.md` gives short conceptual definitions and links to the exact
reference glossary.

### Getting started

This section owns installation, the first successful run, and only the concepts
needed for that run. It should avoid server setup, real instrument data, custom
modules, and exhaustive option lists.

The Quickstart is one canonical tutorial. The Examples landing page links to it
as the “QuickStart example” rather than maintaining a second copy.

### Data model

The three levels each receive a focused page:

- `BaseData`: signal, units, named one-sigma uncertainties, axes,
  `rank_of_data`, weights, masks, broadcasting, and arithmetic semantics;
- `DataBundle`: related quantities, entry naming, validation, and default plot
  metadata; and
- `ProcessingData`: pipeline-wide bundles, processing paths, mutation, and
  snapshots.

The section also needs cross-cutting pages on units/dimensionality and
uncertainty propagation. This avoids forcing users to reconstruct the
scientific data contract from API docstrings or the module author guide.

`uncertainty-propagation.md` is the authoritative user-facing explanation of
MoDaCor's uncertainty model. It should include:

- the representation of every component as an absolute one-standard-deviation
  array, broadcastable to the associated signal;
- why uncertainty sources remain separately named, and when users should
  combine or preserve them;
- the first-order, uncorrelated propagation formulas for addition,
  subtraction, multiplication, division, and supported unary functions;
- unit conversion of signals and absolute uncertainties;
- the key-matching rules for two operands: matching named components,
  non-matching components, missing components, and the special
  `propagate_to_all` fallback when it is the sole key on an operand;
- division-by-zero, invalid-domain, NaN, broadcasting, and masking behavior;
- propagation through weighted and unweighted means, sums, dimensional
  reduction, and numerical integration;
- the distinction between propagating existing measurement uncertainties and
  estimating new uncertainty from Poisson statistics or observed scatter;
- the purposes and different semantics of `PoissonUncertainties`,
  `CombineUncertainties`, and `CombineUncertaintiesMax`;
- worked examples that retain at least two named uncertainty sources through a
  correction chain; and
- explicit limitations: components are treated as independent unless a module
  defines otherwise, covariance is not represented by the current container,
  and named systematic components must not be combined in quadrature merely for
  convenience.

The page should use a compact operation table followed by worked examples. Its
formulas and edge-case descriptions must be checked against `BaseData`, the
reduction/integration modules, and their focused tests. Module guides and the
generated reference should link to this page instead of restating partial or
potentially divergent propagation rules.

### Processing framework

This section explains and teaches the local processing system:

- `ProcessStep` configuration, execution, declared dependencies, and mutation
  boundary;
- directed acyclic pipeline graphs and scheduling;
- YAML configuration and process-step lookup;
- source and sink registries, supported backends, and `ref::path` addressing;
- the CLI and `run_pipeline_job` as the preferred local execution paths;
- full, selected, partial, and chunked execution;
- tracing, provenance, reproducibility metadata, and debugging; and
- the boundary between configuration, runtime registrations, and in-memory
  `ProcessingData`.

Exact fields and flags belong in Reference; these pages explain how and why to
use them.

### Modules

The Modules landing page has two layers:

1. a generated, exhaustive catalogue of every public `ProcessStep`; and
2. curated guides for topics that span modules or require scientific judgment.

The generated catalogue should be grouped by function (data movement,
arithmetic, masks, uncertainty, geometry, reduction/integration,
visualization/output, and technique-specific corrections) while retaining an
alphabetical index. Each generated page remains the authoritative configuration
reference for one step.

A hand-written special guide is justified when at least one of these applies:

- several modules must be ordered or combined correctly;
- choosing parameters requires domain assumptions;
- masks, units, uncertainty, or geometry semantics need explanation;
- important diagnostics, convergence checks, or failure modes exist; or
- a correction has a literature basis or known limitations that do not fit a
  metadata table.

The initial curated set should cover masking, uncertainty workflows (with the
formal propagation rules kept in the Data model section),
arithmetic/normalization, detector and scattering geometry,
integration/reduction, scattering corrections, and the existing detailed
capillary self-absorption guide. Individual trivial modules should not receive a
second hand-written reference page.

### Client-server operation

This section treats the server as an operational form of the same processing
framework, not as an unrelated product. It should contain:

- an architecture overview of client, HTTP service, sessions, pipeline runner,
  source/sink registries, buffers, and persistent outputs;
- installation for trusted local and restricted deployments;
- examples using the supported Python client first, with CLI and raw HTTP where
  appropriate;
- session creation, source/sink registration, processing, dry runs, reset,
  recovery, history, and error inspection;
- buffer transfers, chunk planning, chunked output lifecycle, and restart
  behavior;
- live Plotly/visualization sinks and event consumption;
- registering reviewed custom steps and I/O implementations;
- concurrency, runtime policy, authentication/TLS boundary, and deployment
  cautions; and
- links to the exact REST/OpenAPI reference.

The current large runtime API design-contract page should be split: stable
operator guidance belongs here, exact endpoints belong in Reference, and
historical or unresolved design material belongs under Development.

### Examples

The package documentation should contain only small, maintained examples that
teach supported interfaces without external scientific datasets:

- the canonical synthetic Quickstart (linked from Getting started);
- a local pipeline example using the runner/CLI; and
- a remote server example using the Python client and the same conceptual
  pipeline.

The Examples landing page must prominently link to
[MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples) as the canonical
home for complete instrument examples. It should explain that those examples
provide notebooks, pipeline YAML, data manifests, download/verification tools,
and facility-specific guidance. MOUSE, SAXSess, I22, B21, and future instrument
walkthroughs should be catalogued there rather than copied into the package
repository.

Links from package pages to a particular external example should target stable
repository paths (and, for releases, preferably a matching tag). The main
documentation should not assert that unreleased example datasets are available.

### Reference

Reference pages are exhaustive and terse. The target set includes:

- all public process steps, generated from `ProcessStepDescriber`;
- the public Python data-model, runner, I/O, and client APIs;
- CLI command and option reference;
- pipeline YAML schema and shared `ProcessStep` configuration;
- an I/O capability matrix covering read/write, attributes, slicing, buffers,
  chunked output, optional dependencies, and runtime registration names;
- REST endpoints plus the downloadable OpenAPI document; and
- a glossary of exact terms and path syntax.

The server implementation package does not need indiscriminate autodoc output.
Only supported public client and extension interfaces should be presented as
public Python API.

### Development

Development material should distinguish present contracts from historical
records:

- contribution workflow and software guidelines;
- architectural boundaries and dependency direction;
- extension guides for modules, numerical models/geometry, and I/O;
- testing, generated-documentation, and release requirements;
- documentation authoring and style guidance;
- active design records; and
- completed design records retained for rationale, clearly marked as
  non-authoritative when runtime code and current guides differ.

The runtime code and focused tests remain the executable contract. Active
backlogs belong here, not in the user/operator navigation.

## README design

The root README is the package's storefront, not the documentation table of
contents or contributor handbook. It should remain compact and contain, in
order:

1. project name, one-sentence purpose, and a restrained badge row;
2. a short “why MoDaCor” paragraph naming units, multiple uncertainties,
   traceability, reproducibility, and the quality-over-throughput emphasis;
3. scope and principal applications;
4. installation (`pip install modacor`);
5. a functional synthetic QuickStart;
6. links to the full documentation and
   [MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples);
7. development/contribution, citation, and license links.

The version should not be embedded in the title because it inevitably becomes
stale. Detailed development environment, lint, test, and release commands move
to Development.

### Synthetic QuickStart contract

The README QuickStart and the fuller Getting-started tutorial should use `uv`
to install Python, create the environment, and install MoDaCor. The correction
example itself should use a tiny NumPy array created in the code. It should:

- require only the base `modacor` installation;
- create a `BaseData` count image, a `DataBundle`, and `ProcessingData`;
- create a scalar exposure-time `BaseData` with seconds as its unit;
- load a short inline pipeline containing `PoissonUncertainties` followed by
  `DivideDatabundles`;
- run it with `run_pipeline_job` and tracing enabled;
- demonstrate that the signal changes from counts to counts per second and that
  the named Poisson uncertainty is propagated;
- print a small, deterministic result and the executed step identifiers; and
  avoid files, downloads, network access, optional extras, and manual scheduler
  loops.

The README may show the compact form, while the documentation explains each
line and the hierarchy it builds. Both must execute the same behavior and be
covered by an automated test so copied code cannot silently rot.

## Mapping existing material

Existing useful content should be moved or split rather than rewritten without
reference to its current contracts.

| Current location | Target |
| --- | --- |
| `README.md` overview | compact README plus `introduction/` |
| `docs/installation.md` | `getting-started/installation.md` |
| `docs/getting_started/quickstart.md` | replace with synthetic `getting-started/quickstart.md` |
| `docs/getting_started/cli_and_runner.md` | split between `processing/local-execution.md` and `reference/cli.md` |
| `docs/pipeline_operations/pipeline_basics.md` | split across `data-model/` and `processing/` |
| `docs/pipeline_operations/configuration_reference.md` | split across module guides, processing guidance, and `reference/pipeline-schema.md` |
| `docs/pipeline_operations/tracing_and_debugging.md` | `processing/tracing-and-provenance.md` |
| `docs/pipeline_operations/server_installation.md` | `server/installation-and-deployment.md` |
| `docs/pipeline_operations/advanced_server_use.md` | `server/custom-steps-and-io.md` and `server/security-and-runtime-policy.md` |
| `docs/pipeline_operations/runtime_service_api.md` | split across `server/`, `reference/server-api.md`, and development design records |
| `docs/pipeline_operations/backlog.md` | active record under `development/design/` |
| `docs/corrections/capillary_self_absorption.md` | `modules/capillary-self-absorption.md` |
| `docs/examples/*.md` instrument pages | concise catalogue links to corresponding MoDaCor-examples paths; canonical content remains external |
| `docs/reference/modules/` | retain as generated reference, improve grouping and drift checks |
| `docs/extending/` | `development/` |
| `docs/design/` | `development/design/` |
| `docs/readme.md` | `development/documentation-guide.md` |
| `docs/usage.md` | absorb into Quickstart and local execution; remove as a standalone page |
| authors, changelog, contributing | `project/` or `development/` as appropriate |

Published links should not be broken silently. The implementation plan should
choose and document one redirect mechanism for moved HTML pages, with an
explicit old-to-new URL map. Redirect support is preferable to leaving many
duplicate compatibility pages in the navigation.

## Extensibility rules

The structure stays maintainable only if additions follow predictable rules.

### Adding a process step

- Add complete `ProcessStepDescriber` metadata.
- Export the supported step through the public registry.
- Regenerate the reference catalogue.
- Add or update a curated module guide only when the step changes a workflow or
  scientific explanation covered there.
- Add focused tests for metadata and generated-reference discovery.

### Adding an I/O backend

- Document its exact interface in the generated/public API reference.
- Add it to the I/O capability matrix and source/sink how-to pages.
- Document required optional dependencies and server registration type.
- Update extension guidance if it introduces a new capability contract.

### Adding a server operation

- Update the OpenAPI contract and exact endpoint reference.
- Update an operations page only when the user workflow changes.
- Add client examples for supported client methods.
- Keep unresolved design discussion in Development.

### Adding an example

- Small synthetic interface examples may live in the package documentation.
- Instrument pipelines, notebooks, manifests, and datasets go to
  MoDaCor-examples.
- Add a package-doc catalogue link only after the external example has a stable
  path and clearly stated readiness/data availability.

## Generation and validation

The documentation build should enforce more than successful rendering.

- Build Sphinx with warnings as errors.
- Run link checking separately, including the MoDaCor-examples link while
  allowing for clearly documented network limitations in local/offline runs.
- Generate module pages into a temporary directory in CI and compare them with
  the checked-in output, or generate them only during the build. The chosen
  model must have a single source of truth.
- Assert that every public registered `ProcessStep` appears exactly once in the
  module catalogue and no removed step remains.
- Render source locations as import paths or repository-relative paths, never
  developer-machine absolute paths.
- Execute the synthetic QuickStart in a focused test using the base dependency
  set.
- Validate internal cross-references, toctree membership, and downloadable
  OpenAPI/YAML assets.
- Keep example output small and assert only stable, meaningful values such as
  units, uncertainty keys, shapes, and executed step IDs.

## Acceptance criteria for the eventual refactor

The documentation refactor is complete when:

- the README is compact and its synthetic QuickStart runs from a clean base
  installation without external files;
- every top-level section has a concise landing page and a clear reader
  question;
- `BaseData`, `DataBundle`, and `ProcessingData` each have focused documentation;
- uncertainty propagation has a dedicated page covering formulas, named-key
  behavior, reductions/integration, assumptions, edge cases, and limitations;
- all processing-framework components named in this design are documented;
- all public modules appear in a generated, grouped catalogue;
- the initial curated module guides exist and link to exact module references;
- local and remote examples run through supported high-level APIs;
- client-server architecture, setup, normal operations, live output, custom
  registration, chunking, and security boundaries are discoverable;
- the docs prominently link to MoDaCor-examples and do not duplicate its
  instrument assets;
- user documentation contains no active backlog as normative guidance;
- old published URLs redirect to their new canonical locations;
- generated pages contain no absolute local paths and cannot drift undetected;
  and
- Sphinx warning-as-error, link, generated-doc, and Quickstart checks pass.

## Confirmed implementation decisions

The following decisions were confirmed on 2026-09-28:

1. Adopt the nine primary sections in the proposed navigation, with Project
   information in secondary navigation.
2. Use a self-contained two-step synthetic counts-to-count-rate pipeline for
   both README and Getting-started Quickstart, with `uv` managing Python and the
   environment.
3. Treat MoDaCor-examples as the canonical home of all instrument-specific
   notebooks, pipelines, manifests, and datasets, linked at
   <https://github.com/BAMResearch/MoDaCor-examples>.
4. Split the current runtime API design contract into operator guidance, exact
   API reference, and development history.
5. Keep generated per-step pages, but add functional grouping, public-registry
   completeness checks, and repository-relative source links.
6. Move active and completed design records below Development and preserve old
   published URLs through redirects.
7. Add curated module guides according to the stated criteria rather than one
   hand-written page per module.
8. Keep generated module pages in Git and fail CI when they differ from fresh
   generated output.
9. Preserve old published URLs using `sphinxext-rediraffe`.
10. Document only supported public Python interfaces, not arbitrary server
    internals.
11. Replace duplicated instrument walkthroughs with concise catalogue entries
    linking to MoDaCor-examples.
12. Continue with one current documentation site; multi-version documentation
    is out of scope.
13. Test the executable Python block extracted directly from the README.
14. Place Project information in secondary navigation.

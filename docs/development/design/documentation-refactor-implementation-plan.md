# Documentation refactor implementation plan

Status: completed (2026-09-28)

This plan implements the approved
[documentation architecture](documentation-architecture.md). It covers the
documentation refactor only. It does not change MoDaCor's runtime contracts or
add instrument implementations, pipelines, notebooks, or datasets to the
package repository.

## Confirmed decisions and constraints

- The target navigation has nine primary sections: Introduction, Getting
  started, Data model, Processing framework, Modules, Client-server operation,
  Examples, Reference, and Development. Project information is secondary.
- The QuickStart is a synthetic counts-to-count-rate correction and uses `uv`
  to install Python, create the environment, and install MoDaCor.
- The executable README Python block is tested directly.
- [MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples) is the
  canonical home for instrument notebooks, pipeline YAML, manifests, and
  datasets. The organization move may not be complete when work begins, but all
  new links use the confirmed destination URL.
- Generated process-step pages remain checked into Git and are compared against
  fresh output in CI.
- Old published documentation URLs redirect to their new canonical locations
  using `sphinxext-rediraffe`.
- Public Python reference covers supported data containers, pipeline/runner,
  I/O interfaces, and client APIs. Server implementation internals are not
  published indiscriminately.
- The current single documentation site continues. Multi-version publishing is
  out of scope.
- Active and completed design records move below Development.
- Existing unrelated working-tree files remain untouched.

## Sources of truth

The refactor must not invent behavior while reorganizing prose.

| Subject | Authoritative source |
| --- | --- |
| Data containers and arithmetic | Runtime classes and focused data-class tests |
| Pipeline configuration and execution | `Pipeline`, `ProcessStep`, runner code, and focused tests |
| Process-step fields | `ProcessStepDescriber` and the public process-step registry |
| I/O capabilities | I/O base classes, concrete implementations, runtime builders, and tests |
| Client-server behavior | Client/server code, OpenAPI document, and end-to-end tests |
| Instrument examples | MoDaCor-examples repository |
| Active package architecture | Current maintainer guides, runtime code, and tests |
| Historical rationale | Completed design records, explicitly marked non-authoritative |

When prose and runtime behavior disagree, reconcile the current contract before
copying text into a new page. Moving stale content unchanged is not completion.

## Delivery strategy

The work is divided into reviewable phases. Every phase leaves the documentation
buildable; no phase should depend on a large final cleanup to restore valid
navigation. Content moves should use Git-aware renames, followed by focused
edits, so history remains useful.

The phases are ordered to establish validation and navigation before the large
content migration:

| Phase | Outcome | Depends on |
| --- | --- | --- |
| 0 | Baseline inventory and migration ledger | None |
| 1 | Build, redirect, generation, and QuickStart guardrails | Phase 0 |
| 2 | Compact README, landing page, Introduction, and Getting started | Phase 1 |
| 3 | Data model and uncertainty-propagation documentation | Phase 2 |
| 4 | Processing-framework documentation | Phase 3 |
| 5 | Generated catalogue and curated module guides | Phases 1 and 4 |
| 6 | Client-server and exact API reference split | Phases 1 and 4 |
| 7 | Examples catalogue and external-repository boundary | Phase 2 |
| 8 | Development, design records, and Project information | Phase 1 |
| 9 | Removal of obsolete pages and full validation | Phases 2–8 |

Phases 5–8 can be implemented in separate branches after their shared
foundations exist, but the final navigation and redirect audit remain a single
integration task.

## Phase 0: baseline and migration ledger

### Work

1. Record the current Sphinx document names, toctrees, internal links,
   downloadable assets, and generated process-step list.
2. Run and retain focused baseline results for:

   - Sphinx HTML with warnings as errors;
   - linkcheck, recording existing external failures separately;
   - the module documentation generator;
   - public module registry completeness tests; and
   - the proposed synthetic QuickStart against the development environment.

3. Create the final old-to-new document-name map before moving files. Include
   every page currently reachable from a toctree or an internal link.
4. Classify each existing page as move, split, replace, externalize, historical,
   or remove-after-redirect.
5. Record non-Markdown assets and decide whether each is retained, regenerated,
   moved under an `_static`/download area, or excluded as an editor backup.

### Deliverable

A checked migration ledger, either appended to this plan during execution or
stored as a small machine-readable file used by redirect tests. It must make
page removal explicit rather than relying on Sphinx's orphan warnings.

### Gate

No page move begins until every currently published docname has a target or an
explicit retirement decision.

## Phase 1: documentation infrastructure and guardrails

### Redirect support

1. Add `sphinxext-rediraffe` to the `docs` and aggregate dependency sets in
   `pyproject.toml`.
2. Enable it in `docs/conf.py`.
3. Add `rediraffe_redirects` using Sphinx document names, not hard-coded site
   URLs.
4. Add a focused test that checks the migration ledger and redirect map agree:

   - every moved/retired internal page has a redirect;
   - every redirect target exists; and
   - redirect chains and loops are forbidden.

Redirects are added in the same change as each move. They are not deferred to
Phase 9.

### Generated module documentation

Extend `scripts/generate_module_doc.py` and its tests so that:

- source locations render as import paths and repository-relative source links,
  never absolute workstation paths;
- all public registered steps are discovered deterministically;
- an explicit documentation-only classification maps each step to a functional
  group;
- every public step must have exactly one group, with an error for missing or
  duplicate classification;
- the generated index contains grouped and alphabetical navigation; and
- a `--check` mode, or equivalent temporary-directory comparison, exits
  non-zero when checked-in output is stale.

Add the generator check to the documentation CI path before Sphinx builds.
Regenerate the checked-in pages only through the supported command.

### QuickStart test harness

Define stable markers around the README's canonical executable Python block and
add a focused test that:

1. extracts exactly one marked Python block;
2. executes it using the installed development package;
3. asserts executed step IDs, output signal values, `count/s` units, and the
   named Poisson uncertainty; and
4. rejects file access, downloads, optional dependencies, or reliance on the
   repository's test data.

The surrounding shell setup is not executed in the unit test. CI already proves
package installation; the commands themselves should also receive a lightweight
syntax/content assertion so the README continues to use `uv` consistently.

### Sphinx configuration

- Preserve MyST Markdown as the preferred format.
- Add stable inter-page labels for pages likely to be referenced widely.
- Configure secondary navigation for Project information using the existing
  theme rather than introducing a new documentation framework.
- Keep the warnings-as-errors HTML build.
- Keep linkcheck separate so transient network failures are distinguishable
  from rendering failures.

### Gate

- Existing docs still build with warnings as errors.
- Redirect-map tests pass with the initial mapping.
- Generated docs pass freshness and public-registry completeness checks.
- The synthetic QuickStart test harness passes before the README is shortened.

## Phase 2: README, landing page, Introduction, and Getting started

### Root README

Rewrite `README.md` to contain only:

1. project name and one-sentence purpose;
2. a restrained badge row;
3. a short value proposition covering units, multiple uncertainties,
   traceability, reproducibility, and quality over throughput;
4. scope: monochromatic X-ray and neutron scattering, diffraction, and imaging;
5. base installation and a synthetic QuickStart;
6. links to the full documentation and MoDaCor-examples; and
7. short contribution, citation, and license links.

Remove the hard-coded version from the title and move lint, test, release, and
template-update instructions to Development.

The environment instructions use:

```text
uv python install 3.12
uv venv --python 3.12
<activate the environment>
uv pip install modacor
```

The README should show POSIX activation compactly and link to the installation
page for Windows and source-checkout variants. The Python block creates the
synthetic data, loads an inline two-step pipeline, runs it with tracing, and
prints deterministic output.

### Documentation landing page

Replace the flat top-level list in `docs/index.md` with:

- the one-sentence value proposition;
- four reader routes: first run, pipeline authoring, service operation, and
  contribution;
- the nine primary sections; and
- secondary Project links.

Do not include the root README as a documentation chapter.

### Introduction

Create:

- `docs/introduction/index.md`;
- `docs/introduction/purpose-and-scope.md`;
- `docs/introduction/scientific-principles.md`; and
- `docs/introduction/terminology.md`.

Reuse accurate overview material from the README and existing pages, adding the
approved intended-use and non-goal statements. Keep exact class behavior out of
this section.

### Getting started

Create or move:

- `docs/getting-started/index.md`;
- `docs/getting-started/installation.md`;
- `docs/getting-started/quickstart.md`; and
- `docs/getting-started/where-to-go-next.md`.

The fuller Quickstart uses the same data and pipeline behavior as the tested
README block, explains the hierarchy and trace briefly, and links onward without
introducing instrument files. Installation covers released, source-checkout,
optional-extra, POSIX, and Windows variants using `uv` as the primary workflow.

### Redirects

At minimum:

```text
installation                              -> getting-started/installation
getting_started/index                     -> getting-started/index
getting_started/quickstart                -> getting-started/quickstart
```

`getting_started/cli_and_runner` redirects later when its content has been
split in Phase 4.

### Gate

- README QuickStart extraction test passes.
- A clean temporary `uv` environment can run the documented QuickStart with the
  base package dependencies.
- The README and documentation link to
  `https://github.com/BAMResearch/MoDaCor-examples`.
- Sphinx and internal-link checks pass.

## Phase 3: data model and uncertainty propagation

### Pages

Create:

- `docs/data-model/index.md`;
- `docs/data-model/basedata.md`;
- `docs/data-model/databundle.md`;
- `docs/data-model/processingdata.md`;
- `docs/data-model/units-and-dimensionality.md`;
- `docs/data-model/uncertainty-propagation.md`; and
- `docs/data-model/paths-axes-weights-and-masks.md`.

Extract the relevant material from `pipeline_basics.md` and the module author
guide, but verify it against current container implementations and tests.

### Uncertainty-propagation page

This page receives a focused technical review because it documents scientific
semantics. It must include:

- absolute one-standard-deviation storage and variance views;
- named independent components;
- operation tables and formulas for addition, subtraction, multiplication,
  division, and supported unary functions;
- unit conversion rules for signals and uncertainties;
- exact matching/non-matching key behavior and sole-key `propagate_to_all`
  behavior;
- broadcasting, NaN, invalid-domain, and division-by-zero behavior;
- weighted/unweighted means and sums;
- dimensional reduction and 1D integration;
- distinction between propagated input uncertainty, Poisson estimates, and
  scatter-derived estimates;
- quadrature and maximum combination modules; and
- independence/covariance limitations and warnings about combining systematic
  components.

Each formula or edge case should cite a focused runtime test through a nearby
source comment or reviewer checklist. Examples must retain multiple named
components through at least one unit-changing correction.

### Gate

- Examples on all data-model pages run in focused tests or doctests.
- The uncertainty tables agree with `BaseData`, reduction, integration, and
  uncertainty-tool tests.
- No user-facing page describes uncertainty arrays as variances.
- Cross-links distinguish masks, weights, and uncertainties consistently.

## Phase 4: processing framework

### Pages

Create:

- `docs/processing/index.md`;
- `docs/processing/process-steps.md`;
- `docs/processing/pipeline-graphs.md`;
- `docs/processing/pipeline-configuration.md`;
- `docs/processing/sources.md`;
- `docs/processing/sinks.md`;
- `docs/processing/local-execution.md`;
- `docs/processing/dependencies-and-partial-runs.md`;
- `docs/processing/tracing-and-provenance.md`; and
- `docs/processing/chunked-processing.md`.

Split current material by reader task:

- concepts and recommended use remain here;
- exact CLI flags and YAML fields move to Reference;
- server-only session behavior moves to Client-server operation;
- unresolved designs and backlogs move to Development; and
- instrument-specific paths/examples move to MoDaCor-examples catalogue links.

Prefer `run_pipeline_job` and the supported CLI for local execution. Retain the
manual scheduler only as an advanced/debugging pattern. Explain exact dependency
contracts and partial rerun invalidation without duplicating per-module fields.

Chunked processing should cover the public planning concepts and local/runtime
boundary. Experimental or future strategies remain in design records.

### Redirects

```text
getting_started/cli_and_runner                 -> processing/local-execution
pipeline_operations/pipeline_basics            -> processing/index
pipeline_operations/configuration_reference    -> processing/pipeline-configuration
pipeline_operations/tracing_and_debugging      -> processing/tracing-and-provenance
usage                                           -> processing/local-execution
```

Where one old page splits into several targets, redirect to the best landing
page and add a short “moved topics” list there.

### Gate

- Local examples use supported high-level APIs.
- Source/sink documentation agrees with current registries and builders.
- Dependency and partial-rerun descriptions agree with execution tests.
- No active backlog appears in the user-facing processing toctree.

## Phase 5: modules

### Generated catalogue

Keep generated pages under `docs/reference/modules/`. Regenerate them after the
Phase 1 generator changes and create:

- a functional grouped index;
- an alphabetical index; and
- cross-links from each curated guide to the exact generated pages.

Initial functional groups are:

- data movement and copying;
- arithmetic and normalization;
- masks;
- uncertainty creation and combination;
- geometry and coordinate construction;
- reduction and integration;
- visualization and output; and
- technique-specific corrections.

The classification is exhaustive and enforced by tests.

### Curated module guides

Create:

- `docs/modules/index.md`;
- `docs/modules/choosing-modules.md`;
- `docs/modules/masking.md`;
- `docs/modules/uncertainties.md`;
- `docs/modules/arithmetic-and-normalization.md`;
- `docs/modules/geometry.md`;
- `docs/modules/integration-and-reduction.md`;
- `docs/modules/scattering-corrections.md`; and
- `docs/modules/capillary-self-absorption.md`.

Move and review the existing capillary guide. Build other guides from current
module behavior and representative tests. The uncertainty guide teaches module
selection and workflow; it links to the Data model page for the formal
propagation rules.

### Redirects

```text
corrections/index                       -> modules/scattering-corrections
corrections/capillary_self_absorption   -> modules/capillary-self-absorption
```

Generated per-module URLs remain unchanged where practical.

### Gate

- Every public registered step appears exactly once in a functional group and
  once in the alphabetical index.
- Checked-in generated pages match fresh output.
- No generated page contains an absolute local path.
- Curated guides do not duplicate full configuration tables.

## Phase 6: client-server operation and exact reference

### Client-server pages

Create:

- `docs/server/index.md`;
- `docs/server/architecture.md`;
- `docs/server/installation-and-deployment.md`;
- `docs/server/clients-and-sessions.md`;
- `docs/server/operations.md`;
- `docs/server/buffers-and-chunked-outputs.md`;
- `docs/server/live-output-and-plots.md`;
- `docs/server/custom-steps-and-io.md`; and
- `docs/server/security-and-runtime-policy.md`.

Use the supported Python client for the primary programmatic examples. CLI
examples remain where they provide an operator-friendly equivalent. Raw HTTP is
reserved for the exact API reference or interoperability examples.

Split the current runtime service design-contract page into:

- stable conceptual/operational guidance under `server/`;
- exact request, response, state, and error reference under
  `reference/server-api.md` and the OpenAPI file; and
- unresolved or historical rationale under `development/design/`.

Document live plots as sink/event workflows, not as a server special case.
Document the trusted/restricted policy boundary and the absence of built-in
authentication/TLS prominently.

### Reference pages

Build the Reference section around exact interfaces:

- `docs/reference/index.md`;
- `docs/reference/python-api/index.md`;
- `docs/reference/python-api/data-model.md`;
- `docs/reference/python-api/pipeline-and-runner.md`;
- `docs/reference/python-api/io.md`;
- `docs/reference/python-api/client.md`;
- `docs/reference/cli.md`;
- `docs/reference/pipeline-schema.md`;
- `docs/reference/io-capabilities.md`;
- `docs/reference/server-api.md`;
- `docs/reference/runtime-service-openapi.yaml`; and
- `docs/reference/glossary.md`.

Use autodoc only for the approved public Python surface. Hand-written reference
introductions should state stability and optional dependencies. The I/O matrix
must be backed by concrete implementation capabilities and tests.

Move the OpenAPI YAML without changing its schema unless a verified mismatch is
found. Any mismatch becomes a separate contract correction rather than being
silently folded into a documentation move.

### Redirects

```text
pipeline_operations/server_installation    -> server/installation-and-deployment
pipeline_operations/advanced_server_use    -> server/custom-steps-and-io
pipeline_operations/runtime_service_api    -> reference/server-api
```

### Gate

- Python client examples pass against the local test server.
- CLI examples use current command names and options.
- OpenAPI download and internal references resolve.
- Security and trust boundaries are visible from server installation and custom
  extension pages.
- No private server class is accidentally presented as supported public API.

## Phase 7: examples and MoDaCor-examples integration

### Package documentation

Create or revise:

- `docs/examples/index.md`;
- `docs/examples/local-operation.md`; and
- `docs/examples/remote-server-operation.md`.

The index links to the canonical synthetic Quickstart and contains concise
catalogue entries for instrument examples. Each entry states facility,
instrument, technique/workflow, readiness, data availability, required extras,
and a stable link into:

<https://github.com/BAMResearch/MoDaCor-examples>

The local and remote examples should process the same small conceptual dataset
so users can see that execution mode changes while the processing model does
not.

### Existing instrument pages

Replace the package-local MOUSE, SAXSess, and I22 walkthrough content with
catalogue entries and external links. Do not move their pipeline YAML or
notebooks into new package directories. Remove the package-local
`docs/examples/MOUSE_solids.yaml` after confirming that no remaining test or
published page relies on it.

Until MoDaCor-examples has a public release tag, link to the confirmed repository
paths on its default branch and label readiness accurately. Once releases
exist, add compatibility/pinning guidance without introducing a multi-version
MoDaCor documentation build.

### Redirects

Old instrument docnames redirect to `examples/index`, whose catalogue supplies
the external destinations:

```text
examples/mouse_pipeline     -> examples/index
examples/saxsess_pipeline   -> examples/index
examples/dls_i22            -> examples/index
```

### Gate

- No instrument notebook, pipeline, manifest, or scientific dataset is added to
  the package repository.
- All catalogue links use the BAMResearch destination.
- Linkcheck or a focused external-link check verifies the repository and stable
  example paths once the organization move is live.
- Readiness statements do not imply unreleased data are downloadable.

## Phase 8: Development, design records, and Project information

### Development

Create:

- `docs/development/index.md`;
- `docs/development/contributing.md`;
- `docs/development/development-principles.md`;
- `docs/development/architecture.md`;
- `docs/development/software-guidelines.md`;
- `docs/development/module-author-guide.md`;
- `docs/development/io-source-sink-guide.md`;
- `docs/development/contribution-checklist.md`;
- `docs/development/documentation-guide.md`; and
- `docs/development/testing-and-release.md`.

Move current extension guidance rather than duplicating it. Consolidate package
boundaries, runtime-contract precedence, attrs/Pint/BaseData rules, tests,
generated docs, and release expectations into clearly scoped pages with
cross-links.

### Design records

Move active and completed records to:

```text
docs/development/design/
docs/development/design/completed/
```

Update all repository references, including `AGENTS.md`, to the canonical new
paths in the same change. Preserve status and historical context. Add a standard
header distinguishing proposed, active, completed, and superseded records.

Move the pipeline-operations backlog into the active design area and remove it
from user navigation.

### Project information

Create:

- `docs/project/index.md`;
- `docs/project/changelog.md`;
- `docs/project/authors.md`;
- `docs/project/citation.md`; and
- `docs/project/license.md`.

Includes may continue to source top-level `CHANGELOG.md`, `CONTRIBUTING.md`, or
license content where appropriate, avoiding manual duplication. Citation should
state what metadata exists today and avoid inventing a DOI.

### Redirects

```text
extending/index                         -> development/index
extending/module_author_guide           -> development/module-author-guide
extending/io_source_sink_guide          -> development/io-source-sink-guide
extending/contribution_checklist        -> development/contribution-checklist
readme                                  -> development/documentation-guide
contributing                            -> development/contributing
authors                                 -> project/authors
changelog                               -> project/changelog
pipeline_operations/backlog             -> development/design/<backlog-name>
design/<record>                          -> development/design/<record>
design/completed/<record>                -> development/design/completed/<record>
```

The concrete redirect map enumerates every design record rather than depending
on wildcard behavior.

### Gate

- Repository-local links and `AGENTS.md` point to the new canonical paths.
- Active design records are distinguishable from completed history.
- Contributor instructions do not appear in the first-run user journey.
- Project pages render from their authoritative sources without duplicated
  manually maintained copies.

## Phase 9: integration, removal, and final validation

### Navigation and cleanup

1. Replace the transitional `docs/index.md` toctree with the final hierarchy.
2. Remove obsolete source pages only after redirect coverage is tested.
3. Remove stale claims about placeholders, old datasets, old repository paths,
   and deprecated navigation names.
4. Remove editor backups and unneeded generated binary assets only when their
   ownership and recovery are clear; do not delete unrelated untracked user
   files as part of the refactor.
5. Check page titles, labels, breadcrumbs, “next” links, and section landing
   pages in the built HTML.
6. As the final content step, run a repository-wide search for every moved
   documentation path and update remaining references in `AGENTS.md`, source
   comments, tests, scripts, templates, contributor files, and workflow
   configuration. Perform this after the documentation tree is stable so
   `AGENTS.md` becomes an index of the final canonical contract locations rather
   than transitional paths.

### Full validation

Run, at minimum:

- focused README QuickStart test;
- focused documentation-structure and redirect tests;
- module generator tests and stale-output check;
- public module contract tests;
- client examples against the local server;
- Sphinx HTML build with warnings as errors;
- Sphinx linkcheck;
- repository formatting/lint checks affected by scripts or tests; and
- a clean-environment QuickStart smoke test using `uv` and only base
  dependencies.

Inspect the rendered site at desktop and narrow widths. Verify that a user can
reach installation, Quickstart, module catalogue, server setup, examples, and
contribution guidance from the landing page without knowing the package layout.

### Final acceptance checklist

- The README is compact, appealing, and executable.
- The synthetic QuickStart needs no files or network after package installation.
- The approved nine-section navigation is present.
- The data hierarchy and uncertainty-propagation rules have dedicated pages.
- The processing framework covers all agreed components.
- Generated module reference is exhaustive, grouped, deterministic, and fresh.
- Curated module guides cover the agreed initial topics.
- Client-server architecture, setup, operations, live output, custom extension,
  chunking, and security are discoverable.
- Exact Python, CLI, pipeline, I/O, and REST reference is separated from
  explanatory guidance.
- Instrument documentation links to BAMResearch/MoDaCor-examples without
  duplicating its assets.
- Active backlog/design content is absent from user-facing navigation.
- `AGENTS.md` and all other repository guidance reference the final canonical
  documentation paths, with no old paths remaining outside the redirect map or
  historical text that intentionally discusses the migration.
- Old published URLs redirect without loops or chains.
- No generated page leaks an absolute developer path.
- All validation gates pass.

## Suggested review and commit boundaries

Keep changes reviewable even when several phases are developed close together:

1. documentation guardrails and redirects;
2. README, Introduction, and Getting started;
3. data model and uncertainty propagation;
4. processing framework;
5. module catalogue and curated guides;
6. client-server and Reference;
7. examples catalogue and external links;
8. Development/design/Project moves; and
9. cleanup and final cross-link audit.

Avoid mixing runtime behavior changes into these commits. If documentation
review exposes a genuine runtime or OpenAPI inconsistency, record and fix it in
a separately scoped change with focused tests, then update the documentation
against the corrected contract.

## Completion record

The refactor was completed on 2026-09-28 without runtime behavior changes.
Final verification covered:

- the generated process-step catalogue in check mode;
- documentation, redirect, README QuickStart, and public-module contract tests;
- runtime-client and local-server API tests;
- a clean Sphinx HTML build with warnings treated as errors;
- Sphinx linkcheck, including the BAMResearch/MoDaCor-examples catalogue;
- a fresh Python 3.12 environment created with `uv`, base-dependency package
  installation, and execution of the README QuickStart; and
- the final canonical-path audit, including `AGENTS.md`.

Stable literature resolvers that reject automated checks, generated GitHub
source links that trigger anonymous rate limiting, and one linkified historical
changelog token are explicitly excluded from linkcheck. Their rationale is
recorded beside the exclusions in `docs/conf.py`.

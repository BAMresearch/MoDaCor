# Future Package Architecture

Status: intended direction; no package split has been implemented yet.

This document records the intended long-term separation of MoDaCor into a
small, usable core and independently maintained packages for domain modules,
concrete I/O integrations, the runtime service, and its client. The first
implementation should remain in one repository and use coordinated releases.
Separate repositories are a later operational choice, not a prerequisite for
the architecture.

The editable draw.io source for the intended structure is available as
{download}`future-package-architecture.drawio <future-package-architecture.drawio>`.

## Motivation

MoDaCor currently ships its data model, execution engine, correction modules,
I/O implementations, runtime service, client, command-line interface, and
documentation as one distribution. The existing source directories already
provide useful conceptual seams, but packaging and discovery still assume that
all implementations live below one installed `modacor` package.

The intended split should:

- make the computational core easier to understand and test;
- keep a minimal installation useful for a complete in-process demonstration;
- isolate domain-specific algorithms and optional storage dependencies;
- make the server and client independently deployable;
- preserve stable pipeline YAML and Python APIs during migration; and
- allow new module and I/O packages to register implementations without
  modifying core.

The split is not intended to make the core an interfaces-only SDK. A core
installation should still be capable of executing a small pipeline from input
to output.

## Intended components

### MoDaCor Core

The existing `modacor` distribution should initially remain the core package.
Keeping that distribution name minimizes disruption for current users; the
product may be described as **MoDaCor Core** without immediately renaming the
published package to `modacor-core`.

Core owns:

- the unit registry and common unit behavior;
- `BaseData`, `DataBundle`, `ProcessingData`, and their supporting contracts;
- `ProcessStep`, configuration and dependency descriptions, and tracing;
- the pipeline graph, process-step registry API, and local runner;
- generic geometry and uncertainty primitives;
- `IoSource`, `IoSink`, and source/sink collection interfaces;
- processing-path and chunk-plan contracts shared with runners and servers;
- generic base modules that form the standard pipeline vocabulary; and
- dependency-light in-memory buffer storage, `BufferSource`, and `BufferSink`.

Buffer I/O deliberately remains in core. It is the reference implementation of
the I/O contracts and permits an end-to-end pipeline such as:

```text
BufferSource -> base processing steps -> BufferSink
```

That path must work without a web server, persistent files, Tiled, or other
optional backends. It provides a fast demonstration path, a test fixture, and a
concrete example for extension authors.

The current buffer store uses a `session_id` as part of every key. During the
migration this may be generalized to an opaque namespace, with the existing
name retained as a compatibility alias. A server can use its session ID as the
namespace, while local execution can use a default or generated namespace.
This refinement is desirable but is not required before package extraction.

The NumPy `.npy` encoder and decoder used by HTTP endpoints are related but
conceptually distinct: they define a wire format rather than in-memory storage.
Their final ownership may remain in core initially or move to a small shared
API-contract package if the client is to remain independent of core.

### Standard domain module libraries

Non-base correction modules should move into installable domain libraries. The
first such library is expected to be a scattering package, provisionally named
`modacor-scattering`.

It should own:

- scattering technique modules;
- scattering-specific helpers;
- attenuation models used by those modules; and
- domain-focused tests and reference documentation.

Generic base operations remain in core initially. This includes arithmetic,
mask combination, uncertainty combination, source/sink boundary steps, and
other operations needed to construct a useful basic pipeline. A later split
into a very small kernel and `modacor-standard-modules` should only be pursued
if the size or release cadence of the base library justifies the additional
package.

Instrument-specific implementations are not part of the general domain
library. They should live in the central MoDaCor examples repository or in
facility-owned extension packages and register themselves through the same
extension mechanism.

### Concrete I/O library

A package provisionally named `modacor-io` should contain concrete integrations
with persistent files and external services. Initial contents may include:

- HDF5 sources and processing sinks, including the chunked HDF5 sink;
- NeXus interpretation helpers;
- CSV and YAML sources or sinks;
- Tiled sources and sinks; and
- visualization sinks.

Core retains the abstract I/O interfaces, buffer I/O, processing-path
contracts, chunk specifications, and registry APIs. This ensures that the
runner and process steps depend only on core contracts rather than on a
particular backend.

`modacor-io` should begin as one distribution with dependency extras where
appropriate. Individual packages such as `modacor-io-hdf` or
`modacor-io-tiled` should be introduced only when dependency weight,
maintainership, or release cadence makes the additional packaging worthwhile.

### Runtime server

The server package, provisionally `modacor-server`, owns:

- runtime sessions and their lifecycle;
- runtime policy and trust-boundary enforcement;
- HTTP and WebSocket routes;
- buffer upload and download endpoints;
- partial-rerun orchestration and error reporting;
- server-managed chunked-output resources;
- process-level resource and concurrency controls;
- the server command-line entry point; and
- container definitions and deployment guidance.

The server depends on core. A standard server installation may also install the
standard domain and I/O packages, but the server implementation must resolve
their capabilities through registries rather than import them as core
internals. Core never imports the server.

### Runtime client

The client package, provisionally `modacor-client`, owns:

- HTTP transport and API error handling;
- session, buffer, plot, and chunked-output clients;
- readiness and lifecycle convenience methods; and
- the remote-client command-line interface.

The client communicates with the server only through the versioned API. It
must not import server implementation classes. A lightweight client that does
not require the full scientific core is preferred. The local-server launcher
may be moved to the server package or exposed as a client extra whose explicit
dependency is `modacor-server`.

### Documentation

Documentation is a separate build product covering all supported packages. It
may remain at the repository root while the packages share a monorepo. The
documentation build installs the relevant packages, generates their reference
pages, and consumes the authoritative server API description.

The OpenAPI contract belongs to the server or to a shared API-contract package,
not solely to documentation. Documentation renders that contract but does not
own it.

## Dependency rules

The intended dependencies are one-way:

1. Core has no dependency on the server, client, domain libraries, or concrete
   I/O library.
2. Domain libraries and concrete I/O implementations depend on core contracts.
3. The server depends on core and composes installed module and I/O providers.
4. The client depends only on its transport dependencies and, if introduced, a
   small shared API-contract package.
5. Documentation may depend on every package for reference generation and
   integration examples.
6. Instrument extensions may depend on core and selected domain or I/O
   libraries, but core never imports instrument code.

These rules should be enforced with import-boundary tests rather than left as
conventions.

## Extension discovery

Package extraction requires replacing assumptions about physical source-tree
locations. In particular, process steps should not need to reside below
`modacor/modules`, and runtime source/sink construction should not rely on a
hard-coded map of every available backend.

Installed providers should register stable names through explicit registries
or Python package entry points. Candidate entry-point groups are:

```text
modacor.process_steps
modacor.io_sources
modacor.io_sinks
```

Existing YAML names should continue to work. Registry behavior must define:

- deterministic name resolution;
- a clear error for duplicate names;
- provider and version information for diagnostics and provenance;
- an explicit curated set for restricted server deployments; and
- a programmatic registration path for tests and embedded applications.

Filesystem discovery may remain temporarily as a compatibility mechanism, but
it should not be the primary cross-package extension contract.

## Command-line ownership

The current command-line interface combines local execution, remote client
commands, and server startup. These responsibilities should become separate
entry points owned by their packages. A compatibility dispatcher may preserve
the existing `modacor` commands for a transition period.

A possible final arrangement is:

```text
modacor run              # core local execution
modacor-client ...       # remote session operations
modacor-server ...       # service startup and administration
```

The exact command names remain an implementation decision. Ownership and
dependency direction are more important than the spelling.

## Packaging and repository strategy

The first physical split should use a monorepo with multiple independently
buildable distributions, for example:

```text
packages/core
packages/scattering
packages/io
packages/server
packages/client
docs
```

Advantages of starting with a monorepo include atomic contract changes, one
integration-test workflow, and simpler coordinated releases. Packages should
use lockstep versions initially. Explicit compatibility ranges and independent
versions can be introduced after the boundaries and release cadence have
proved stable.

Moving packages into separate repositories should be considered only when
there are distinct maintainer groups, access policies, or release cadences.
Repository separation is not necessary to obtain dependency isolation or
smaller installable artifacts.

## Compatibility strategy

The migration should preserve, where practical:

- the `modacor` distribution as the initial core installation;
- existing public imports through re-exports or time-limited compatibility
  shims;
- established process-step names in pipeline YAML;
- the versioned `/v1` server API during the first extraction; and
- an installation path equivalent to the current `modacor[server]` experience.

A convenient aggregate or `standard` installation may install core, the
scattering library, common I/O integrations, and the server. Smaller
installations should remain available for library authors, remote clients, and
specialized deployments.

Deprecation warnings must identify the replacement package or import and remain
for at least one documented compatibility period before old paths are removed.

## Verification strategy

Each distribution needs focused unit tests, while the monorepo retains an
integration matrix that verifies supported combinations. At minimum, CI should
cover:

- core installed alone and running a buffer-backed end-to-end pipeline;
- discovery and execution of a process step supplied by the domain library;
- discovery and round-trip use of each installed I/O backend;
- server startup with only its declared dependencies;
- client/server contract tests against the supported API version;
- wheel installation tests in clean environments;
- restricted-server allowlisting of installed providers; and
- documentation generation against the built packages rather than incidental
  source-tree imports.

## Migration phases

1. **Establish boundaries in the existing distribution.** Separate CLI
   responsibilities, define supported core interfaces, and add import-boundary
   tests.
2. **Introduce provider registries.** Register built-in process steps and I/O
   implementations explicitly, then add entry-point discovery for external
   providers.
3. **Confirm the core end-to-end path.** Keep buffer I/O and representative
   base modules in core and test a complete pipeline in a core-only
   environment.
4. **Create the multi-package workspace.** Extract the scattering, concrete
   I/O, server, and client distributions without changing their public
   behavior.
5. **Publish a compatibility release.** Preserve existing installation and
   import paths while documenting the new package choices.
6. **Evaluate finer splits and repository separation.** Split individual I/O
   backends or repositories only when actual maintenance experience supports
   it.

## Completion criteria

The intended separation is complete when:

- core alone runs the documented buffer-backed demonstration;
- no core module imports an extracted implementation package;
- installed process-step and I/O providers are discovered without filesystem
  scanning of the core package;
- the server and client can be built and tested independently;
- clean-environment wheel tests verify every supported installation profile;
- server/client compatibility is checked against an authoritative API
  contract; and
- users have a documented, warning-backed migration path from the monolithic
  installation.

## Decisions intentionally left open

- Whether a small `modacor-api-contract` distribution is justified.
- Whether `modacor-io` should eventually become one package per backend.
- Whether generic base modules should later move to a standard-module package.
- Whether all distributions should continue using lockstep versions.
- Whether any component ultimately benefits from a separate repository.

These decisions should be made from dependency, maintenance, and deployment
evidence gathered during the staged migration rather than fixed in advance.

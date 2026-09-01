# Architecture Upgrade Plan

This document tracks the design issues identified during the September 2026
architecture review. Keep the status and notes current as each item is fixed.

Status values:

- `Open`: not started.
- `In progress`: actively being changed.
- `Done`: implemented and verified.
- `Deferred`: intentionally postponed with a reason.

## Items

1. `Open` Runtime API trust boundary

   The runtime API currently accepts local filesystem paths for pipeline YAML and
   custom source/sink class import paths from client payloads. This is acceptable
   only for a trusted local helper. A network-facing service needs path
   allowlists, disabled or registry-backed custom imports, and an explicit auth
   model.

2. `Done` Partial rerun invalidation contract

   Dirty-step selection should be based on an explicit per-step dependency
   contract instead of server-owned config heuristics. Steps should declare which
   source refs they read and which `ProcessingData` keys they read or write.

   Outcome: `ProcessStep.dependency_contract()` now provides the partial-rerun
   dependency contract. Boundary steps declare exact contracts for source
   loading, source registration, sink registration, and sink exports. Runtime
   dirty-step detection now consumes the contract and expands matching seeds to
   downstream descendants.

3. `Open` Pipeline scheduler ownership

   `Pipeline` currently inherits from `graphlib.TopologicalSorter`, which mixes
   graph data with one-shot scheduler state. Prefer keeping `Pipeline` as the
   graph/spec holder and creating a fresh scheduler per execution.

4. `Open` Process-step configuration schema

   Step configuration is split between `CONFIG_KEYS` and
   `ProcessStepDescriber.arguments`. Constructor-provided configuration is
   overwritten during initialization, and documented argument types are not
   centrally validated.

5. `Open` Transactional step execution

   `calculate()` is documented as returning a mapping that `execute()` merges,
   but several modules mutate shared `ProcessingData` before that merge. Partial
   rerun rollback and trace semantics would be clearer with an explicit commit
   boundary.

6. `Open` Core data metadata contracts

   `DataBundle` is currently a permissive `dict`, and `BaseData` arithmetic
   inherits axes/rank/weights from the left operand. Mixed-axis operations need
   an explicit metadata policy.

## Update Log

- 2026-09-01: Created tracker and started item 2.
- 2026-09-01: Completed item 2. Verified with targeted `.venv-dev` pytest runs.

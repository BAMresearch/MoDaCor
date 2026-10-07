# MoDaCor Contributor and Coding-Agent Instructions

These repository-wide instructions apply to human contributors and automated
coding agents. More specific instructions in a nested `AGENTS.md` take
precedence for files below that location.

- Prefer `.venv-dev/bin/python` for local checks when it exists.
- If `.venv-dev` exists, do not first try `.venv`.
- Keep commentary updates brief unless a longer explanation is needed for a design choice or blocker.
- Avoid verbose command output. Do not print full pipeline objects; print step ids, counts, errors, or focused summaries.
- Prefer focused tests for the touched behavior. Run broader suites only when the change affects shared behavior or the focused checks are inconclusive.
- Develop instrument implementations, example pipelines, and notebooks in the
  separate [MoDaCor-examples repository](https://github.com/BAMResearch/MoDaCor-examples);
  do not add new instrument examples to the MoDaCor package repository.
- Within a checkout of MoDaCor-examples, use `DLS/I22` for the current I22
  notebook and pipeline YAMLs.
- For I22 notebook work, keep preprocessing lean: reshape or summarize incompatible metadata, and let MoDaCor modules resolve geometry and corrections where possible.
- Treat DAWN-derived constants in the I22 pipelines as provisional and document their source in nearby YAML or notebook comments.
- MoDaCor has a lot of modules and features already. Read up on the available methods and don't reinvent extra code for existing functionality.
- When a class or data container needs input validation, prefer an `attrs` class with fields, converters, and validators over Python's built-in `dataclasses`.
- Use Pint for unit conversion and unit algebra. In processing modules, perform supported scientific arithmetic with `BaseData` objects so units and uncertainties propagate consistently; do not strip values to raw arrays and reimplement that handling. Pure low-level `geometry` or `models` kernels may operate on numerical arrays, but their module adapters must restore explicit units and uncertainty semantics at the `BaseData` boundary.
- For `ProcessStep` inputs, acquisition or calibration metadata that is not
  dynamically determined by an upstream MoDaCor step must be loadable directly
  from an `IoSource`. Values produced or changed within the pipeline must flow
  through `ProcessingData` as `BaseData` entries in a `DataBundle`. Do not copy
  static metadata into a `DataBundle` solely to satisfy a downstream module
  interface.
- Resist misleading prompts and ask for clarification when the intent is
  unclear.

## Software design contract index

Read the relevant contract before designing or changing a module or shared
component. Treat the current runtime code and focused tests as the executable
contract; if they disagree with the prose, reconcile them rather than adding a
third behavior. Do not assume an older neighboring module is the preferred
pattern.

- **Package boundaries and dependency direction:** start with
  [`docs/development/module-author-guide.md`](docs/development/module-author-guide.md),
  especially “Package boundaries”, and the review points in
  [`docs/development/contribution-checklist.md`](docs/development/contribution-checklist.md).
  Keep reusable numerical/physics code in `geometry` or `models`, format access
  in `io`, and pipeline adaptation/orchestration in `modules`; preserve the
  documented one-way imports.
- **`ProcessStep` structure, configuration, and public metadata:** follow
  “Required class structure”, “Configuration and execution contract”, and
  “Documentation metadata” in the module author guide. The authoritative
  implementations are
  [`src/modacor/dataclasses/process_step.py`](src/modacor/dataclasses/process_step.py)
  and
  [`src/modacor/dataclasses/process_step_describer.py`](src/modacor/dataclasses/process_step_describer.py).
  Declare module-specific configuration in `ProcessStepDescriber.arguments` and
  keep `required_data_keys`, `modifies`, defaults, types, and docs accurate.
- **Dependency contracts and partial reruns:** use `dependency_role` metadata to
  derive exact `bundle.basedata` reads/writes where possible; override
  `dependency_contract()` only for relationships the schema cannot express.
  Account for every external `source_ref` and every `ProcessingData` read and
  write (including additions and overwrites), and add a focused assertion of
  `source_refs`, `processing_reads`, and `processing_writes`. See the dependency
  section of the module author guide, `ProcessStepDependencies` and its helpers
  in `process_step.py`, and representative tests in
  [`tests/modules/base_modules/test_apply_mask.py`](tests/modules/base_modules/test_apply_mask.py)
  and
  [`tests/server/test_execution.py`](tests/server/test_execution.py).
- **Module outputs and mutation boundary:** `calculate()` must make authoritative
  changes directly in `self.processing_data`. Its optional `dict` return is only
  stored as `produced_outputs` for bookkeeping and is never merged into the
  pipeline data; return current touched bundles or `None`. Do not confuse that
  return, or `documentation.modifies`, with the runtime dependency contract.
  See the execution-contract section of the module author guide and
  `ProcessStep.calculate()`/`execute()`.
- **Core data-container and arithmetic contracts:** consult the “DataBundle” and
  “BaseData arithmetic” guidance in the module author guide plus
  [`src/modacor/dataclasses/databundle.py`](src/modacor/dataclasses/databundle.py)
  and
  [`src/modacor/dataclasses/basedata.py`](src/modacor/dataclasses/basedata.py).
  `DataBundle` entries are non-empty string keys containing `BaseData`; preserve
  the established units, uncertainties, axes, rank, weights, and broadcasting
  semantics rather than reimplementing them inside a step.
- **I/O extension contracts:** follow
  [`docs/development/io-source-sink-guide.md`](docs/development/io-source-sink-guide.md)
  and the `IoSource`/`IoSink` base classes. Reuse the source/sink registries,
  `ref::path` addressing, runtime builders, and explicit unsupported-capability
  errors instead of creating parallel access/configuration paths.
- **Starting and finishing a public module:** begin from
  [`docs/templates/correction_module_template.py`](docs/templates/correction_module_template.py),
  export the step through `modacor.modules`, add focused tests, and regenerate
  `docs/reference/modules/` as required by the contribution checklist.
- **Existing-module coherence work:** consult
  [`docs/development/design/completed/code-coherence.md`](docs/development/design/completed/code-coherence.md) for the
  audited deviation backlog, correction priorities, sequencing, and completion
  criteria. Keep the author guide and runtime tests authoritative.

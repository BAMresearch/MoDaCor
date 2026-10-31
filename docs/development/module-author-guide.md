# Module author guide

MoDaCor processing modules are `ProcessStep` subclasses. They are instantiated
from pipeline YAML, resolved by name through `modacor.modules` and the
`ProcessStepRegistry`, and documented through their
`ProcessStepDescriber` metadata.

## Package boundaries

The main distinction is whether code describes reusable mathematics, a
scientific model, an external format, or pipeline orchestration. Dependencies
should point in this direction:

```text
modules -> models -> geometry
   |                   ^
   +------> io --------+
```

The arrow means “may import”. `geometry` and `models` must not import pipeline
modules.

| Package | Put this here | Keep this out |
| --- | --- | --- |
| `modacor.geometry` | Unit-vector, ray/shape intersection, affine-transform, and coordinate-frame primitives whose inputs and outputs are ordinary numerical arrays | `BaseData`, `ProcessStep`, `IoSources`, NeXus/HDF5 paths, configuration keys, or attenuation coefficients |
| `modacor.models` | Reusable scientific equations, numerical integration, sensitivity calculations, and physics kernels; for example flat-plate or concentric-cylinder attenuation | Pipeline selection, source resolution, `DataBundle` mutation, YAML semantics, or format traversal |
| `modacor.io` | File- or protocol-specific reading and writing, metadata traversal, and adapters such as the NeXus `depends_on` resolver | Technique correction policy or pipeline mutation |
| `modacor.modules` | `ProcessStep` classes and their configuration, dependency contracts, source resolution, units/metadata adaptation, masking, uncertainty attachment, and `ProcessingData` mutation | A reusable numerical kernel embedded as a private method merely because one step currently calls it |

A useful test is whether a function can be exercised using only NumPy arrays
and scientific parameters. Pure shape and coordinate mathematics normally
belongs in `geometry`; a physical equation or quadrature normally belongs in
`models`. If it needs `BaseData`, an IO source reference, or pipeline
configuration, it belongs in the adapter or module layer.

Shared code should be promoted before one process-step module imports a private
helper from another. Mirror the source boundary in tests: `tests/geometry`,
`tests/models`, `tests/io`, and `tests/modules` respectively.

Current examples include:

- `modacor.geometry.cylinders` and `modacor.geometry.transforms` for pure
  intersections and homogeneous transforms;
- `modacor.models.attenuation` for beam-profile quadrature and attenuation
  factors;
- `modacor.io.nexus.geometry` for NeXus transformation-chain traversal; and
- `modacor.modules.helpers.scattering.detector_data` for `BaseData` preparation
  used by detector-coordinate process steps. Those helpers remain in the
  module layer because they adapt MoDaCor data containers rather than
  implement geometry.

Non-step implementation support belongs below `modacor.modules.helpers`, with
technique-specific helpers in a matching subpackage such as
`modacor.modules.helpers.scattering`. Filesystem discovery excludes every
`helpers` and `deprecated` tree. A retired step may be kept under a
`deprecated` package for source reference and direct tests, but it must not be
exported from `modacor.modules` and cannot be selected by a pipeline.

## Where modules live

- Put broadly reusable steps in `src/modacor/modules/base_modules/`.
- Put technique-specific steps in a dedicated subpackage under
  `src/modacor/modules/technique_modules/`.
- Bespoke instrument-specific steps should be put in a subfolder of `src/modacor/modules/instrument_modules/`, where the subfolders follow the following structure:
`[institute abbreviation]/[instrument abbreviation]`. For example:
`src/modacor/modules/instrument_modules/DLS/I22/`.
- Export any public step from `src/modacor/modules/__init__.py` so the curated
  registry and generated reference docs stay aligned.
- Put shared non-step adapters under `src/modacor/modules/helpers/`, not beside
  discoverable `ProcessStep` modules.

## Required class structure

Every module must:

1. Subclass `modacor.dataclasses.process_step.ProcessStep`.
2. Define a class-level `documentation = ProcessStepDescriber(...)`.
3. Implement `calculate(self) -> dict[str, DataBundle] | None`.

Optionally implement `prepare_execution()` if the step needs one-time setup or
cached derived state before `calculate()` runs.

The template in `docs/templates/correction_module_template.py` is the best
starting point for new work.

## Configuration and execution contract

`ProcessStep` already provides shared configuration keys:

- `with_processing_keys`: select which `ProcessingData` bundles the step should
  operate on.

Step-specific configuration belongs in `documentation.arguments`. Those entries
seed the instance `configuration` automatically through
`ProcessStepDescriber.initial_configuration()`.

`ProcessStep` builds one effective configuration schema from `CONFIG_KEYS` and
`documentation.arguments`. Defaults are copied from that schema when the step is
constructed, and values supplied through `ProcessStep(configuration=...)`,
pipeline YAML, graph specs, `modify_config_by_dict(...)`, or
`modify_config_by_kwargs(...)` are validated against the same schema. Unknown
keys raise `KeyError`; wrong value types raise `TypeError`.

For new module-specific settings, add entries to `documentation.arguments`:

```python
documentation = ProcessStepDescriber(
    ...,
    arguments={
        "signal_key": {
            "type": str,
            "default": "signal",
            "doc": "BaseData key to read and write within each DataBundle.",
        },
        "scale": {
            "type": (float, int),
            "default": 1.0,
            "doc": "Scalar factor applied to the signal.",
        },
    },
)
```

Use `CONFIG_KEYS` only for shared `ProcessStep`-level keys or legacy
compatibility. It supports `type`, `allow_iterable`, `allow_none`, and
`default`; when a key exists in both places, the `CONFIG_KEYS` type policy stays
authoritative and `documentation.arguments` supplies the public default,
required flag, and docs text. This keeps existing keys such as
`with_processing_keys` compatible while moving module-specific validation into
the public metadata.

The central schema checks key presence and top-level value types. Keep
step-local validation in `calculate()` or helper methods for semantic rules such
as non-empty strings, mutually exclusive options, source-reference shape, or
nested dictionary contents.

## Dependency contracts and partial reruns

Declare the `BaseData` keys a step accesses in this order:

1. Use `dependency_role` for configurable keys in `documentation.arguments`.
   A string supplies one key, a list supplies its members, and a mapping
   supplies its keys. `None` and empty values contribute no keys. Supported
   roles are `processing_read_basedata_key`, `processing_write_basedata_key`,
   and `processing_read_write_basedata_key`, plus their `_list` variants.
2. Use `fixed_data_keys` for keys hard-coded by the implementation. Each
   access is `read`, `write`, or `read_write`; keys must be non-empty strings.
   The describer strips whitespace and rejects invalid keys or access values
   when the class-level metadata is constructed.
3. For conditional or derived extras, keep an extension override: start with
   `contract = super().dependency_contract()`, add the extra source references
   or processing paths, and return a new `ProcessStepDependencies` preserving
   the base sets. `AngleToQ` adds a conditional centre read;
   `FindCenterOfMass1D` adds prefix-derived diagnostic writes.
4. Keep a full override for multi-bundle and I/O steps, where input/output
   selection, path parsing, or registration side effects cannot be expressed
   by declarations. Account for every source reference and processing read
   and write, including additions and overwrites.

For example, this step updates a hard-coded signal, reads a configurable mask,
and writes a configurable correction map:

```python
documentation = ProcessStepDescriber(
    ...,
    fixed_data_keys={"signal": "read_write"},
    arguments={
        "mask_key": {
            "type": (str, type(None)),
            "default": None,
            "doc": "Optional mask BaseData key.",
            "dependency_role": "processing_read_basedata_key",
        },
        "correction_key": {
            "type": str,
            "default": "correction",
            "doc": "Output correction map key.",
            "dependency_role": "processing_write_basedata_key",
        },
    },
)
```

The default contract expands fixed keys and role values over
`with_processing_keys`, yielding exact paths such as `sample.signal`.
Non-empty declared keys without `with_processing_keys` give `*` reads and
writes because the bundle selection is unknown. An empty declaration, such
as a mapping-valued role with `{}`, gives an empty processing contract even
when bundle selection is unset. External `ref::path` references in configuration
are collected independently in `source_refs`.

Steps without dependency hints retain conservative whole-bundle reads and
writes, or `*` when neither sources nor bundle selection describe their work.
Prefer declarations for new steps so unrelated key changes do not trigger
unnecessary partial reruns.

Keep `required_data_keys` equal to the default declared reads and `modifies`
keys equal to the default declared writes. The public-module metadata test
checks both sets; conditional or derived extras need an explicitly reasoned
exception in its table. Full overrides retain hand-written expectations.

The session-installed access trace in `tests/_contract_trace.py` checks each
step's execution against its declared contract throughout the test suite. It
records `DataBundle` reads and writes, including `update()` keys and newly
inserted bundles, and fails the executing test if a touched key is uncovered.
Re-storing the same bundle object is ignored. Keep focused tests asserting
`source_refs`, `processing_reads`, and `processing_writes` as well: the trace
only checks configurations exercised by tests. `no_contract_trace` is reserved
for test steps with deliberately inconsistent contracts; it must not hide a
real module's missing dependency.

Pipeline YAML uses YAML sequences for list-like values. If a module declares a
configuration key with `"type": tuple`, a YAML sequence such as `[1.0, 0.0,
0.0]` is accepted and stored as a Python `tuple` in `self.configuration`.

During execution the runner injects:

- `processing_data`
- `io_sources`
- `io_sinks`
- `step_id`

`calculate()` is the in-place mutation boundary. It should update
`self.processing_data` directly, either by mutating existing `BaseData` arrays,
replacing entries inside a `DataBundle`, or assigning a new/replacement
`DataBundle` to a `ProcessingData` key.

The optional return value is bookkeeping only: return a mapping of touched
`ProcessingData` keys to their current `DataBundle` values when that is useful
for tests, tracing, or compatibility. The base `execute()` method stores this
mapping in `produced_outputs`, but it does not merge returned data into
`ProcessingData`.

This contract keeps the normal execution path allocation-aware. Rollback is a
runtime/session concern: partial-service runs can use snapshots or full rerun
fallback when recovery is needed, but individual steps should not take full
pipeline copies just to execute.

`DataBundle` values must be `BaseData` instances stored under non-empty string
keys. Wrap raw arrays in `BaseData` with explicit units before inserting them
into a bundle.

For `BaseData` arithmetic, rely on the core operation layer for cheap structural
metadata protection: it preserves metadata through metadata-neutral scalar
factors and correction maps, and rejects rank or axes conflicts. Unit
compatibility and unit algebra are still Pint's responsibility. If both operands
carry array-valued weights, ordinary arithmetic keeps the left operand's weights
to preserve existing in-place correction behavior.

If your module requires stronger domain compatibility, such as identical
coordinate-axis values or a different rule for combining two weight maps, check
that explicitly before arithmetic and assign the result metadata deliberately.

For steps that operate on existing bundles, prefer
`self._normalised_processing_keys()` instead of duplicating input-selection
logic.

## Documentation metadata

`ProcessStepDescriber` is not optional bookkeeping. It drives both runtime
introspection and generated reference docs. At minimum, keep these fields
accurate:

- `calling_name`: human-facing short name
- `calling_id`: class name used in pipeline YAML
- `calling_module_path`: usually `Path(__file__)`
- `calling_version`: module version string
- `required_data_keys`
- `arguments`
- `modifies`
- `step_doc`
- `step_note` and `step_keywords` where useful

The generated pages under `docs/reference/modules/` come from that metadata via:

```bash
python scripts/generate_module_doc.py --all --output-dir docs/reference/modules --index docs/reference/modules/index.md
```

If a new public step is added but not exported from `modacor.modules`, both the
registry behavior and the generated docs become inconsistent.

## Testing expectations

Add tests close to the behavior you are changing:

- step-focused unit tests under `tests/modules/...`
- registry/discovery tests under `tests/runner/...` when export or lookup
  behavior changes
- integration coverage under `tests/integration/...` when behavior only shows up
  in a full pipeline run

The current suite already has good examples for new tests, including:

- `tests/modules/base_modules/test_append_source.py`
- `tests/modules/base_modules/test_append_sink.py`
- `tests/runner/test_process_step_registry.py`
- `tests/integration/test_pipeline_run.py`

## Maintainer checklist for a new public step

1. Add the module class and metadata.
2. Export it from `src/modacor/modules/__init__.py`.
3. Add or update tests.
4. Regenerate `docs/reference/modules/`.
5. Rebuild the docs if the step changes user-facing behavior.

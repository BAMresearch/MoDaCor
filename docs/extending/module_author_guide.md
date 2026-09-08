# Module author guide

MoDaCor processing modules are `ProcessStep` subclasses. They are instantiated
from pipeline YAML, resolved by name through `modacor.modules` and the
`ProcessStepRegistry`, and documented through their
`ProcessStepDescriber` metadata.

## Where modules live

- Put broadly reusable steps in `src/modacor/modules/base_modules/`.
- Put technique-specific steps in a dedicated subpackage under
  `src/modacor/modules/technique_modules/`.
- Bespoke instrument-specific steps should be put in a subfolder of `src/modacor/modules/instrument_modules/`, where the subfolders follow the following structure:
`[institute abbreviation]/[instrument abbreviation]`. For example:
`src/modacor/modules/instrument_modules/DLS/I22/`.
- Export any public step from `src/modacor/modules/__init__.py` so the curated
  registry and generated reference docs stay aligned.

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
- `output_processing_key`: optional output target for steps that write to a
  different `ProcessingData` bundle than they read.

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

Argument specs can also declare how configuration values contribute to the
runtime dependency contract used by partial reruns. Use `dependency_role` when a
configuration value names a `BaseData` entry in each selected
`with_processing_keys` bundle:

```python
documentation = ProcessStepDescriber(
    ...,
    arguments={
        "source_basedata_key": {
            "type": str,
            "default": "signal",
            "doc": "BaseData key to read.",
            "dependency_role": "processing_read_basedata_key",
        },
        "target_basedata_key": {
            "type": str,
            "default": "corrected",
            "doc": "BaseData key to write.",
            "dependency_role": "processing_write_basedata_key",
        },
        "basedata_to_update": {
            "type": list,
            "default": ["signal"],
            "doc": "BaseData keys to read and update.",
            "dependency_role": "processing_read_write_basedata_key_list",
        },
    },
)
```

Supported roles are:

- `processing_read_basedata_key`
- `processing_write_basedata_key`
- `processing_read_write_basedata_key`
- `processing_read_basedata_key_list`
- `processing_write_basedata_key_list`
- `processing_read_write_basedata_key_list`

When at least one argument declares a dependency role,
`ProcessStep.dependency_contract()` derives exact `sample.signal`-style reads
and writes from the current configuration. Modules with no dependency roles keep
the older conservative behavior based on whole `with_processing_keys` bundles.
Override `dependency_contract()` only for unusual cases, such as IO side
effects, source/sink registration steps, parsing dependency paths from strings,
or steps where different positions in `with_processing_keys` have different
meanings.

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

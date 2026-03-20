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
3. Implement `calculate(self) -> dict[str, DataBundle]`.

Optionally implement `prepare_execution()` if the step needs one-time setup or
cached derived state before `calculate()` runs.

The template in `docs/templates/correction_module_template.py` is the best
starting point for new work.

## Configuration and execution contract

`ProcessStep` already provides shared configuration keys:

- `with_processing_keys`: select which `ProcessingData` bundles the step should
  operate on.
- `output_processing_key`: optional output target for steps that emit a new
  bundle instead of updating in place.

Step-specific configuration belongs in `documentation.arguments`. Those entries
seed the instance `configuration` automatically through
`ProcessStepDescriber.initial_configuration()`.

During execution the runner injects:

- `processing_data`
- `io_sources`
- `io_sinks`
- `step_id`

`calculate()` should return a mapping of `ProcessingData` key to updated
`DataBundle`. The base `execute()` method merges that mapping back into the
current `ProcessingData`.

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

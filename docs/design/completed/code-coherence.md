# Module and Component Code Coherence

Status: completed on 2026-09-27.

## Purpose

This document turns the current software design contracts into an ordered
coherence programme for existing code. Its aim is not to make every module look
identical. It is to ensure that every component states the same facts in its
runtime behavior, dependency contract, public metadata, tests, and generated
documentation.

The normative authoring rules remain in
[the module author guide](../../extending/module_author_guide.md) and
[the contribution checklist](../../extending/contribution_checklist.md). This note
records deviations found in the current module set and the recommended order
for resolving them.

## Audit scope and baseline

The current audit surface covers all 36 public `ProcessStep` classes exported through
`modacor.modules`, their module-layer helpers, their generated reference pages,
and focused module/runtime dependency tests. The deliberately non-discoverable
deprecated `XSGeometry` step was not treated as part of the public surface.

The following checks were made:

- package placement and dependency direction;
- `ProcessStepDescriber` configuration and public metadata;
- exact `source_refs`, `processing_reads`, and `processing_writes` behavior;
- authoritative in-place mutation and `produced_outputs` bookkeeping;
- Pint conversion and `BaseData` unit/uncertainty propagation;
- semantic validation and use of `attrs` where a class owns validation; and
- agreement between implementation, tests, generated reference pages, and
  prose documentation.

At the audit baseline, the focused command

```text
.venv-dev/bin/python -m pytest -q tests/modules tests/server/test_execution.py
```

passed all 280 tests. The findings below are therefore primarily uncovered
contract gaps, not a list of failing tests. Each fix must add a regression test
that fails before the fix; a passing legacy suite is not sufficient evidence
that the contract is complete.

## Implementation progress

- **C1 completed on 2026-09-27:** `FindScaleFactor1D` now converts a copied
  working signal, including all named uncertainties, to the reference signal
  units before fitting. Compatible-unit and incompatible-unit regression tests
  protect the behavior without mutating the input signal.
- **C2 completed on 2026-09-27:** `IndexedAverager` now honors
  `output_processing_key` for a single input, rejects its ambiguous use with
  multiple inputs, preserves the source bundle in distinct-output mode, and
  retains auxiliary entries in in-place mode. Its dependency contract now
  declares exact input reads and output-mode-specific writes.
- **C3 completed on 2026-09-27:** `Integrate1D` now declares exact signal,
  axis, and optional mask reads plus output-mode-specific writes, including
  newly created output bundles.
- **C4 completed on 2026-09-27:** source contracts now include flat-plate
  transmission uncertainty sources and the plain source identifiers in NeXus
  detector-frame and sample-position mappings.
- **C5 completed on 2026-09-27:** `ApplyMask` converts non-`uint32` masks only
  into a local working array and no longer mutates its declared read-only mask.
- **Phases 1 and 3 completed on 2026-09-27:** executable guardrails now cover
  every exported step, generated-reference discovery, exact contract values,
  and the absence of runtime assertions. All identified wildcard dependency
  contracts were narrowed to the paths actually read or written.
- **Phase 4 completed on 2026-09-27:** default required keys, modified keys,
  calling identifiers, return annotations, and argument descriptions now agree
  with the current public implementations. The generated module reference was
  rebuilt from the 36-step public surface.
- **Phase 5 completed on 2026-09-27:** module input checks now raise explicit
  exceptions; `_EstimatorSpec` is a validated `attrs` class; and the source and
  sink registration helpers no longer use mutable function defaults.
- **Phase 6 completed on 2026-09-27:** reusable quadrature, scale-fitting,
  planar attenuation, and polarization kernels now live under `modacor.models`.
  Their `ProcessStep` adapters retain unit conversion, `BaseData` handling,
  dependency declarations, configuration, and pipeline mutation. An import
  guardrail protects the documented `models` and `geometry` boundaries.

## Contract hierarchy

Four related contracts must remain distinct:

1. `ProcessStepDescriber.required_data_keys` and `modifies` document the public
   default behavior and drive generated reference material.
2. `dependency_contract()` drives partial-rerun invalidation and must describe
   every actual external source read and `ProcessingData` read/write.
3. Mutations made directly to `self.processing_data` are the authoritative
   execution result.
4. The optional mapping returned by `calculate()` is only a record of touched
   current bundles; `execute()` stores it as `produced_outputs` and does not
   merge it into pipeline data.

A change is coherent only when all four views agree where they overlap.

## Findings requiring behavioral correction

These items can produce incorrect scientific values or incorrect partial-rerun
selection and should be addressed before metadata cleanup or refactoring.

### C1. `FindScaleFactor1D` dependent-unit handling (completed)

At audit time, the module converted the independent axes to a common unit but
fitted the raw magnitudes of the dependent signals. Physically equal signals
expressed in metres and centimetres consequently produced a dimensionless
scale of 100 instead of 1.

Implemented outcome:

- require compatible dependent units;
- convert the working signal and all its uncertainty components to the
  reference signal unit through `BaseData.to_units()` before extracting arrays;
- keep the fitted scale dimensionless after conversion and the optional
  background in the reference signal unit; and
- add compatible-unit and incompatible-unit regression tests.

### C2. `IndexedAverager` advertised and actual output disagree (completed)

At audit time, `output_processing_key` was accepted and changed the inherited
dependency contract, but calculation ignored it and replaced each selected
input bundle. The class documentation also said `pixel_index` and masks
remained present even though the replacement bundle contained only `signal`,
`Q`, and `Psi`.

Implemented outcome:

- a distinct output is supported for one input and leaves the source bundle
  intact;
- configuring one output for multiple inputs is rejected as ambiguous;
- in-place operation retains auxiliary entries while replacing `signal`, `Q`,
  and `Psi`; and
- the dependency contract, return mapping, public metadata, and tests describe
  that behavior.

### C3. `Integrate1D` does not declare new output bundles (completed)

At audit time, configuring `output_processing_keys` created those bundles but
the inherited dependency contract declared only writes to the input bundles.

Implemented outcome: the custom contract declares exact signal, axis, and
optional mask reads, and declares either `input.output_key` writes or each new
`output_processing_key.*` write according to the configured mode.

### C4. Incomplete external source tracking (completed)

At audit time:

- `FlatPlateSelfAbsorptionCorrection` read
  `transmission_uncertainties_sources` but omitted those references from its
  contract; and
- `PixelCoordinates3D` and `XSGeometryFromPixelCoordinates` missed plain source
  identifiers nested in NeXus `detector_frame` and `sample_z_override`
  dictionaries because the generic `ref::path` extractor cannot infer those
  fields.

Implemented outcome: the field-specific contracts include those source
identifiers explicitly without making the generic extractor treat arbitrary
configuration strings as source references.

### C5. `ApplyMask` mutates a read-only dependency (completed)

At audit time, the mask was declared as a processing read, but non-`uint32`
masks were converted and written back to the source `BaseData`. It is now
converted into a local working array without mutating the mask.

## Dependency-contract precision (completed)

At audit time, the following modules used whole-bundle invalidation where exact
paths were available:

- `CopyDataBundleKeys`;
- `Divide`, `Multiply`, and `Subtract`;
- `DivideDatabundles`, `MultiplyDatabundles`, and `SubtractDatabundles`;
- `IndexPixels`;
- `PoissonUncertainties`;
- `SolidAngleCorrection`;
- `UnitsLabelUpdate`;
- `AttenuatorPlateCorrection`; and
- `DetectorEfficiencyCorrection`.

These contracts now name exact source references and `BaseData` paths. The
two-bundle arithmetic steps use asymmetric contracts that write only their
first operand; configurable-key steps derive paths from their configuration;
and the material corrections name their signal, geometry, and correction-map
paths rather than invalidating whole bundles.

Focused tests assert equality for all three `ProcessStepDependencies` fields so
future wildcards and undeclared dependencies cannot pass as subsets.

## Public metadata reconciliation (completed)

At audit time, the generated module reference was incomplete or misleading in
these areas:

- `AttenuatorPlateCorrection`, `DetectorEfficiencyCorrection`,
  `FlatPlateSelfAbsorptionCorrection`, and `PolarizationCorrection` create
  correction-map entries not listed in `modifies`.
- `IndexPixels`, `Integrate1D`, `DilateMask`, `ReduceMask`, and `ThresholdMask`
  add or replace entries while declaring `modifies={}`.
- `PoissonUncertainties` describes `variances.Poisson`, although it writes
  `signal.uncertainties["Poisson"]`.
- `UnitsLabelUpdate` uses empty-string placeholders in `required_data_keys` and
  `modifies`.
- `FindScaleFactor1D`, `Integrate1D`, and `DivideDatabundles` understated their
  default required inputs.
- `BitwiseOrMasks`, `Divide`, `Multiply`, and `Subtract` have `calling_id`
  values different from the class names used by the registry and pipeline YAML.

The current descriptors now state default public behavior, while argument
documentation explains configurable keys. Empty strings are no longer used as
dynamic placeholders. If exact dynamic outputs become necessary for runtime
introspection, extend the descriptor with a deliberate machine-readable
mechanism instead of overloading `modifies`.

The reference pages were regenerated after the corrections. A discovery test
now requires the exports, discoverable `ProcessStep` implementations, generated
targets, filenames, and index entries to remain in agreement.

## Validation and typing (completed for the current public surface)

At audit time, runtime input validation relied on `assert` in `ApplyMask`,
`BitwiseOrMasks`, `CopyDataBundleKeys`, `DilateMask`, `DivideDatabundles`,
`MultiplyDatabundles`, `ReduceMask`, `SubtractDatabundles`, and
`ThresholdMask`. These checks now raise explicit `TypeError` or `ValueError`
exceptions. An AST guardrail rejects new runtime assertions in public modules.

Four internal carriers use Python's built-in `dataclasses`:

- `ReduceDimensionality._EstimatorSpec`;
- `CapillarySelfAbsorptionCorrection._ResolvedSampleMu`;
- `material_attenuation.MaterialAttenuation`; and
- `statistics.WeightedScatterEstimates`.

`_EstimatorSpec` now owns its semantic constraints as an `attrs` class with a
converter and validators. The remaining three are passive internal result
carriers; migrate them only when they become validation boundaries or when
consistency materially simplifies the code. Do not perform a mechanical
dataclass-to-attrs rewrite without a contract benefit.

The documented return annotation is now present on every public `calculate()`
method, including `PixelCoordinates3D`, `PoissonUncertainties`, and
`XSGeometryFromPixelCoordinates`.

## Units, uncertainties, and data containers

Most scientific arithmetic already follows the intended design:

- arithmetic corrections use `BaseData` operators;
- geometry modules use Pint and `BaseData` for unit conversion and uncertainty
  propagation at their adapter boundary; and
- reduction modules that require array kernels reconstruct explicit units,
  weights, axes, ranks, and named uncertainties.

The confirmed unit correctness defect is C1. `UnitsLabelUpdate` is an
intentional metadata repair tool, not a converter. Keep that limitation explicit
and never use it where `BaseData.to_units()` or a Pint quantity conversion is
required. Numeric thresholds in `ThresholdMask` are documented as magnitudes
in the source signal's current units. An explicit threshold-unit configuration
remains necessary before cross-unit thresholds can be supported.

## Package-boundary cleanup (completed for the audited candidates)

The audit identified these reusable numerical or physical kernels:

- `Integrate1D._quadrature_weights` and the fit/interpolation kernel in
  `FindScaleFactor1D` for `modacor.models`;
- the pure transmission and efficiency equations in
  `AttenuatorPlateCorrection` and `DetectorEfficiencyCorrection` for the
  attenuation models package; and
- the linear polarization-factor equation in `PolarizationCorrection` for a
  scattering model module.

They now reside in `modacor.models.integration`, `modacor.models.scaling`,
`modacor.models.attenuation.planar`, and
`modacor.models.scattering.polarization`, respectively. The `ProcessStep`
classes retain configuration interpretation, source resolution, `BaseData`
adaptation, dependency declarations, and mutation of `ProcessingData`. Pure
kernels do not import `ProcessStep`, `BaseData`, or I/O registries, and a
package-boundary test enforces that direction.

## Recommended implementation sequence

### Phase 1: add executable guardrails (completed)

1. Add regression tests for C1--C5 before changing behavior.
2. Add exact dependency-contract tests for every module touched in later
   phases.
3. Add lightweight consistency tests for public classes: non-empty class-name
   `calling_id`, no empty metadata keys, valid `calculate()` return annotation,
   and public export/reference-page alignment.

These checks should report concrete class names rather than attempting to infer
all reads and writes statically.

### Phase 2: correct values and false invalidation claims (completed)

Implement C1 through C5 in this order:

1. `FindScaleFactor1D` units (completed);
2. `IndexedAverager` output semantics (completed);
3. `Integrate1D` output dependencies (completed);
4. missing external source references (completed); and
5. `ApplyMask` mask mutation (completed).

Keep each behavioral correction in a focused change with its own tests. Do not
combine these fixes with package moves.

### Phase 3: make dependency contracts exact (completed)

1. Repair fixed-key single-bundle arithmetic steps.
2. Repair asymmetric two-bundle operations and `CopyDataBundleKeys`.
3. Repair dynamic-output modules such as `IndexPixels` and
   `UnitsLabelUpdate`.
4. Narrow the material-correction contracts.

Run the server partial-rerun tests after each group because an under-declared
contract is a correctness error, while an over-declared contract is a
performance and coherence defect.

### Phase 4: reconcile public metadata and generated documentation (completed)

Update `required_data_keys`, `modifies`, `calling_id`, argument descriptions,
and module notes. Regenerate all public module pages and ensure examples use the
class names accepted by `ProcessStepRegistry`.

### Phase 5: harden validation and class contracts (completed)

Replace runtime assertions, migrate `_EstimatorSpec` to validated `attrs`, add
missing return annotations, and remove mutable function defaults in
`AppendSource` and `AppendSink` while those files are being touched.

### Phase 6: extract reusable kernels (completed)

Move reusable numerical functions into `models` only after behavioral and
contract tests are stable. Keep module adapters thin and verify that package
imports still point in the documented direction.

### Phase 7: final verification (completed)

Final verification on 2026-09-27 produced these results:

- the full test suite passed with 935 tests, one opt-in memory test skipped,
  and three established `BaseData` numerical-domain warnings;
- repository-wide Ruff checks passed for `src`, `tests`, and `scripts`;
- generated-reference discovery covered exactly all 36 exported steps and the
  generator pruned the obsolete analyser-angle page; and
- Sphinx completed with `-E -W --keep-going` and no warnings.

## Completion criteria

This coherence programme is complete when:

- every public step has an exact dependency contract or a documented reason for
  conservative wildcard behavior;
- every nontrivial contract has an exact regression test;
- Pint conversions and `BaseData` arithmetic protect all compatible-unit and
  uncertainty-bearing operations at module boundaries;
- public metadata and generated reference pages match default runtime behavior;
- no runtime validation relies on `assert`;
- validation-owning data classes use `attrs`;
- all public `calculate()` methods satisfy the mutation, return, and annotation
  contract;
- reusable pure kernels reside below the module adapter layer; and
- the full tests, lint checks, and documentation build pass.

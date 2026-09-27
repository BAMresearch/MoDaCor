# Module and Component Code Coherence

Status: active remediation plan, based on the 2026-09-27 module audit.

## Purpose

This document turns the current software design contracts into an ordered
coherence programme for existing code. Its aim is not to make every module look
identical. It is to ensure that every component states the same facts in its
runtime behavior, dependency contract, public metadata, tests, and generated
documentation.

The normative authoring rules remain in
[the module author guide](../extending/module_author_guide.md) and
[the contribution checklist](../extending/contribution_checklist.md). This note
records deviations found in the current module set and the recommended order
for resolving them.

## Audit scope and baseline

The audit covered all 37 public `ProcessStep` classes exported through
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

### C1. `FindScaleFactor1D` dependent-unit handling

The module converts the independent axes to a common unit but fits the raw
magnitudes of the dependent signals. Physically equal signals expressed in
metres and centimetres consequently produce a dimensionless scale of 100
instead of 1.

Required outcome:

- require compatible dependent units;
- convert the working signal and all its uncertainty components to the
  reference signal unit through `BaseData.to_units()` before extracting arrays;
- keep the fitted scale dimensionless after conversion and the optional
  background in the reference signal unit; and
- add compatible-unit and incompatible-unit regression tests.

### C2. `IndexedAverager` advertised and actual output disagree

`output_processing_key` is accepted and changes the inherited dependency
contract, but calculation ignores it and replaces each selected input bundle.
The class documentation also says `pixel_index` and masks remain present even
though the replacement bundle contains only `signal`, `Q`, and `Psi`.

Required outcome:

- choose one explicit output model before implementation;
- preferably use a distinct output for a single input and leave the source
  bundle intact, or introduce an unambiguous per-input output mapping for
  multiple inputs;
- if in-place replacement remains supported, document which entries are
  deliberately discarded; and
- make the dependency contract, return mapping, public metadata, and tests
  describe that exact behavior.

### C3. `Integrate1D` does not declare new output bundles

When `output_processing_keys` is configured, calculation creates those bundles
but the inherited dependency contract declares only writes to the input
bundles.

Required outcome: implement a custom contract that declares exact signal, axis,
and optional mask reads, and declares either `input.output_key` writes or each
new `output_processing_key.*` write according to the configured mode.

### C4. Incomplete external source tracking

- `FlatPlateSelfAbsorptionCorrection` reads
  `transmission_uncertainties_sources` but omits those references from its
  contract.
- `PixelCoordinates3D` and `XSGeometryFromPixelCoordinates` miss plain source
  identifiers nested in NeXus `detector_frame` and `sample_z_override`
  dictionaries; the generic `ref::path` extractor cannot infer those fields.

Required outcome: include field-specific source identifiers explicitly. Do not
make the generic extractor treat every arbitrary configuration string as a
source reference.

### C5. `ApplyMask` mutates a read-only dependency

The mask is declared as a processing read, but non-`uint32` masks are converted
and written back to the source `BaseData`. Prefer converting into a local
working array without mutating the mask. If canonicalizing the stored mask is a
required public behavior, declare the mask read/write and document that output.

## Dependency-contract precision backlog

The following modules use whole-bundle invalidation where exact paths are
available:

- `CopyDataBundleKeys`;
- `Divide`, `Multiply`, and `Subtract`;
- `DivideDatabundles`, `MultiplyDatabundles`, and `SubtractDatabundles`;
- `IndexPixels`;
- `PoissonUncertainties`;
- `SolidAngleCorrection`;
- `UnitsLabelUpdate`;
- `AttenuatorPlateCorrection`; and
- `DetectorEfficiencyCorrection`.

For fixed-key modules, add a small explicit contract. For configurable
BaseData-key arguments, prefer `dependency_role`. Multi-bundle operations need
custom contracts because input positions have different read/write roles.
`AttenuatorPlateCorrection` and `DetectorEfficiencyCorrection` already override
the method, but should replace `bundle.*` patterns with exact signal, geometry,
and correction-map paths.

Every repaired module must receive a focused equality assertion for all three
fields of `ProcessStepDependencies`; subset assertions do not prevent accidental
wildcards or undeclared dependencies.

## Public metadata backlog

The generated module reference is currently incomplete or misleading in these
areas:

- `AttenuatorPlateCorrection`, `DetectorEfficiencyCorrection`,
  `FlatPlateSelfAbsorptionCorrection`, and `PolarizationCorrection` create
  correction-map entries not listed in `modifies`.
- `IndexPixels`, `Integrate1D`, `DilateMask`, `ReduceMask`, and `ThresholdMask`
  add or replace entries while declaring `modifies={}`.
- `PoissonUncertainties` describes `variances.Poisson`, although it writes
  `signal.uncertainties["Poisson"]`.
- `UnitsLabelUpdate` uses empty-string placeholders in `required_data_keys` and
  `modifies`.
- `FindScaleFactor1D`, `Integrate1D`, `DivideDatabundles`, and
  `XSGeometryFromAnalyserAngle` understate their default required inputs.
- `BitwiseOrMasks`, `Divide`, `Multiply`, and `Subtract` have `calling_id`
  values different from the class names used by the registry and pipeline YAML.

For configurable key names, metadata should describe the default public
behavior and the argument documentation should explain how configuration
changes it. Do not use empty strings as dynamic placeholders. If exact dynamic
outputs become necessary for runtime introspection, extend the descriptor with
a deliberate machine-readable mechanism instead of overloading `modifies`.

After correcting metadata, regenerate `docs/reference/modules/` and review the
resulting pages as part of the same change.

## Validation and typing backlog

Runtime input validation currently relies on `assert` in `ApplyMask`,
`BitwiseOrMasks`, `CopyDataBundleKeys`, `DilateMask`, `DivideDatabundles`,
`MultiplyDatabundles`, `ReduceMask`, `SubtractDatabundles`, and
`ThresholdMask`. Assertions disappear under optimized Python. Replace them with
explicit `TypeError` or `ValueError` checks, preferably during
`prepare_execution()` when validation does not depend on per-frame values.

Four internal carriers use Python's built-in `dataclasses`:

- `ReduceDimensionality._EstimatorSpec`;
- `CapillarySelfAbsorptionCorrection._ResolvedSampleMu`;
- `material_attenuation.MaterialAttenuation`; and
- `statistics.WeightedScatterEstimates`.

`_EstimatorSpec` is the clear migration candidate because its fields have
semantic constraints currently validated outside the class. Move those
constraints into an `attrs` class with converters and validators. The remaining
three are passive internal result carriers; migrate them only when they become
validation boundaries or when consistency materially simplifies the code. Do
not perform a mechanical dataclass-to-attrs rewrite without a contract benefit.

Add the documented return annotation to `PixelCoordinates3D.calculate()`,
`PoissonUncertainties.calculate()`, and
`XSGeometryFromPixelCoordinates.calculate()`.

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
required. Numeric thresholds in `ThresholdMask` are interpreted in the source
signal's current units; document that fact or add an explicit threshold-unit
configuration before cross-unit thresholds are supported.

## Package-boundary cleanup

Move pure reusable numerical or physical kernels only after their behavior is
covered by the preceding tests. Candidates include:

- `Integrate1D._quadrature_weights` and the fit/interpolation kernel in
  `FindScaleFactor1D` for `modacor.models`;
- the pure transmission and efficiency equations in
  `AttenuatorPlateCorrection` and `DetectorEfficiencyCorrection` for the
  attenuation models package; and
- the linear polarization-factor equation in `PolarizationCorrection` for a
  scattering model module.

The `ProcessStep` classes should retain configuration interpretation, source
resolution, `BaseData` adaptation, dependency declarations, and mutation of
`ProcessingData`. Pure kernels must not import `ProcessStep`, `BaseData`, or I/O
registries.

## Recommended implementation sequence

### Phase 1: add executable guardrails

1. Add regression tests for C1--C5 before changing behavior.
2. Add exact dependency-contract tests for every module touched in later
   phases.
3. Add lightweight consistency tests for public classes: non-empty class-name
   `calling_id`, no empty metadata keys, valid `calculate()` return annotation,
   and public export/reference-page alignment.

These checks should report concrete class names rather than attempting to infer
all reads and writes statically.

### Phase 2: correct values and false invalidation claims

Implement C1 through C5 in this order:

1. `FindScaleFactor1D` units;
2. `IndexedAverager` output semantics;
3. `Integrate1D` output dependencies;
4. missing external source references; and
5. `ApplyMask` mask mutation.

Keep each behavioral correction in a focused change with its own tests. Do not
combine these fixes with package moves.

### Phase 3: make dependency contracts exact

1. Repair fixed-key single-bundle arithmetic steps.
2. Repair asymmetric two-bundle operations and `CopyDataBundleKeys`.
3. Repair dynamic-output modules such as `IndexPixels` and
   `UnitsLabelUpdate`.
4. Narrow the material-correction contracts.

Run the server partial-rerun tests after each group because an under-declared
contract is a correctness error, while an over-declared contract is a
performance and coherence defect.

### Phase 4: reconcile public metadata and generated documentation

Update `required_data_keys`, `modifies`, `calling_id`, argument descriptions,
and module notes. Regenerate all public module pages and ensure examples use the
class names accepted by `ProcessStepRegistry`.

### Phase 5: harden validation and class contracts

Replace runtime assertions, migrate `_EstimatorSpec` to validated `attrs`, add
missing return annotations, and remove mutable function defaults in
`AppendSource` and `AppendSink` while those files are being touched.

### Phase 6: extract reusable kernels

Move reusable numerical functions into `models` only after behavioral and
contract tests are stable. Keep module adapters thin and verify that package
imports still point in the documented direction.

### Phase 7: final verification

Run focused tests throughout, then the complete test suite, lint checks,
generated-reference check, and warning-free documentation build. Record the
verification result and move this document to `docs/design/completed/` only
when every acceptance criterion below is met.

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

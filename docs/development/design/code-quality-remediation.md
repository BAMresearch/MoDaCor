# Code Quality Remediation

Status: active, started 2026-10-30.

## Purpose

This is a living tracker for a code-quality review of MoDaCor (and one item in
MoDaCor-examples). Each item is a self-contained work package with a scope,
approach, and acceptance criteria. Items are ordered by cost-benefit: cheap
fixes with real user impact first, and large structural refactors last.

The normative rules remain `AGENTS.md` at the repository root, the
[module author guide](../module-author-guide.md), and the
[contribution checklist](../contribution-checklist.md). If this note disagrees
with them, they win; flag the conflict in the item's review log.

## Baseline (2026-10-30, `main` at `2361d35`)

- `.venv-dev/bin/python -m pytest -q`: 1015 passed, 1 skipped (about 28 s).
- Line coverage of `src/modacor`: 82.9 %.
- `flake8 src tests`: clean.
- Package dependency direction (`modules -> models -> geometry`,
  `modules -> io`) is respected; `geometry`/`models` do not import `BaseData`
  or `ProcessStep`.

Every item must leave the full suite green and flake8 clean.

## Workflow

Roles: **implementer** (Codex/GPT, relayed by the maintainer) and **reviewer**
(Claude), with the maintainer in the middle approving each step.

1. The reviewer sets an item to `ready` and produces an implementation brief
   from this note.
2. The implementer works only on that item, as one commit (or a short series)
   whose message starts with the item ID, for example `R03: ...`.
3. The implementer fills in the item's **Implementation notes** and sets the
   status to `in review`. Do not edit other items.
4. The reviewer checks the diff against the acceptance criteria and `AGENTS.md`,
   runs the focused tests, and records the verdict in the **Review log**:
   `done`, or `changes requested` with a numbered list.
5. When all items are `done` or `deferred`, move this note to `completed/`.

Status values: `open`, `ready`, `in progress`, `in review`,
`changes requested`, `done`, `deferred`, `decision needed`.

### Rules for every item

- Read `AGENTS.md` and the item fully before you start. Do not widen scope;
  if you find a related problem, note it under Implementation notes instead of
  fixing it.
- Every bug fix comes with a regression test that fails before the fix.
- Behavior-preserving refactors must not change any existing test's
  expectations, unless the item explicitly says so.
- If a public `ProcessStep`'s metadata changes, regenerate and check its
  reference page:

  ```text
  .venv-dev/bin/python scripts/generate_module_doc.py --all --output-dir docs/reference/modules --index docs/reference/modules/index.md
  .venv-dev/bin/python scripts/generate_module_doc.py --all --output-dir docs/reference/modules --index docs/reference/modules/index.md --check
  ```

- Run the focused tests named in the item, then the full suite and
  `flake8 src tests`. Report counts only; do not paste long output.
- Bump the touched module's `__version__` (format `YYYYMMDD.N`) when its
  behavior or public metadata changes.

## Overview

| ID | Item | Kind | Cost | Benefit | Status |
| --- | --- | --- | --- | --- | --- |
| R01 | `PolarizationCorrection` fails when no factor is set | bug | XS | high | done |
| R01b | One input per setting: remove configuration aliases and the mask-key fallback | simplification | S | high | ready |
| R01c | "Did you mean …?" hint for unknown configuration keys | usability | XS | medium | open (optional) |
| R02 | `DilateMask` documents an alias that is rejected | bug | XS | medium | done |
| R03 | One source of truth for configuration defaults, with a guard test | consistency | S | high | open |
| R04 | Derive capillary sample μ through `BaseData` | contract | S | medium | open |
| R05 | Test coverage for arithmetic steps and low-coverage helpers | tests | S | medium | open |
| R06 | Repository hygiene: untracked artefacts and `.gitignore` | housekeeping | XS | low | open |
| R07 | Declarative fixed-key dependencies; retire most `dependency_contract()` overrides | design | M–L | high | decision needed |
| R08 | Audit remaining unit stripping in modules | contract | M | medium | open |
| R09 | Narrow broad `except Exception` handlers | robustness | M | medium | open |
| R10 | Conventions: `attrs` in step modules, headers, `step_reference` | consistency | S–M | low | open |
| R11 | Split `RuntimeService` into focused services | structure | L | medium | open |
| R12 | Large I/O files (`hdf_chunked_processing_sink`, `chunking`) | structure | L | low | deferred |
| E01 | MoDaCor-examples: slim I22 preprocessing helpers | examples | M | medium | open |

Sizes: XS under an hour, S a few hours, M about a day, L several days.

---

## R01 — `PolarizationCorrection` fails when no factor is set

**Status:** done

**Problem.** `polarization_factor` and the alias `polarisation_factor` are both
declared with default `None`. `_fraction_from_config` falls back with
`cfg.get("polarisation_factor", 0.5)`, but the key is always present (seeded with
`None`), so the fallback never applies and `float(None)` raises `TypeError`.
The documented behavior is that 0.5 means unpolarized; a step configured with
neither key should use 0.5 or fail with a clear error.

Reproduction:

```python
from modacor.modules import PolarizationCorrection as P
P._fraction_from_config(P(io_sources=None).configuration)  # TypeError
```

**Files.**
`src/modacor/modules/technique_modules/scattering/polarization_correction.py`
(`_fraction_from_config`, about line 123); its test module under
`tests/modules/technique_modules/scattering/`.

**Approach.** Treat `None` as "not set" for both keys. Resolve
`polarization_factor`, then `polarisation_factor`, then 0.5. Raise `ValueError`
if both are set to different values. Keep the range check.

**Acceptance criteria.**

- With neither key set, the fraction is 0.5.
- With either key set, that value is used; with both set and different,
  `ValueError` is raised.
- Regression tests for all three cases; the first fails on `main`.

**Implementation notes.**

- Treat `None` as unset for both spellings, resolve the primary spelling then
  its alias then 0.5, and reject conflicting explicit values with `ValueError`.
  The existing range check is retained.
- Added regression coverage for the default correction (including signal and
  uncertainty propagation), explicit `None`, either spelling, matching aliases,
  and conflicting aliases. Confirmed the no-factor regression failed before
  the fix with `TypeError` from `float(None)`: 1 failed, 11 deselected.
- Bumped the module version to `20261030.1`, clarified the argument docs, and
  regenerated the polarization reference page.
- Validation: polarization tests 14 passed; full suite 1021 passed, 1 skipped,
  3 warnings; `flake8 src tests` 0 violations; isort 0 errors; generated
  reference check 0 errors; Sphinx build 0 warnings, 0 errors.
- Scope is limited to R01; no related items were changed.

**Review log.**

- 2026-10-30, `2a637dc`: **done.**
  - Verified independently: with the pre-fix `src` and the new tests, 3 failed
    and 9 passed; on `2a637dc`, 12 passed. Full suite 1021 passed, 1 skipped;
    flake8 clean; reference-page `--check` clean.
  - The resolution order, the conflict error and the range check match the
    acceptance criteria. The end-to-end test also covers signal and
    uncertainty propagation at the 0.5 default.
  - Minor: the notes report "14 passed" for the polarization tests; the file
    has 12. Probably a different selection; no action.
  - Process note: the commit added this note but not its `index.md` toctree
    entry, so a strict docs build of `2a637dc` alone would warn about an
    orphan document. The entry is committed with this review.
  - Future items: commit only the item's own files; the reviewer commits
    tracker updates.

---

## R01b — One input per setting: remove configuration aliases and the mask-key fallback

**Status:** ready

**Principle (maintainer decision, 2026-10-30).** Each setting has exactly one
input. MoDaCor does not accept aliases, alternative spellings, or
case-variant fallbacks for configuration keys or `BaseData` keys. Every alias
needs a precedence rule, a conflict check and tests. The existing aliases
already disagree: `PolarizationCorrection` lets the American factor spelling
win and rejects conflicts (R01), while the British offset spellings silently
override the American ones.

**Usage check (reviewer, 2026-10-30).** Searched MoDaCor, MoDaCor-examples and
the maintainer's 2025/2026 project folders (YAML, notebooks, Python, Markdown):

- `polarisation_*` keys: no uses outside the step, its tests and its
  reference page.
- `chemical_composition`: no uses.
- `ThresholdMask.threshold`: one use, in
  `tests/modules/base_modules/test_threshold_mask.py`.
- The capillary `mask`/`Mask` fallback: no pipeline uses either capillary
  correction.

**Mask-key convention (maintainer delegated, reviewer decided).** The canonical
key is lowercase `mask`. Fixed core keys are lowercase (`signal`,
`uncertainties`, and NeXus `mask`); capitalized keys are derived physical
quantities (`Q`, `Psi`, `TwoTheta`, `Omega`, `CosAlpha`). Seven of the eight
mask-key defaults are already `mask`. Case-insensitive lookup (`.lower()`) was
rejected. `DataBundle` keys are case-sensitive, so `mask` and `Mask` can both
exist in one bundle, and a case-insensitive lookup would reintroduce two names
for one thing.

**Scope.**

1. **`PolarizationCorrection`:** remove `polarisation_factor`,
   `polarisation_angular_offset` and `polarisation_angular_offset_units`, and
   simplify `_fraction_from_config` and `_offset_from_config` to one key each
   (`None` factor → 0.5). Keep `polarisation` in `step_keywords` so search
   still finds the step. Replace R01's alias tests with tests showing that each
   British key is rejected with `KeyError`.
2. **`AttenuatorPlateCorrection` and `DetectorEfficiencyCorrection`:** remove
   the `chemical_composition` alias for `material`, including its resolution
   code and dependency handling.
3. **`ThresholdMask`:** remove the deprecated `threshold` alias for
   `upper_bound`, and update the test that uses it to `upper_bound`.
4. **Capillary corrections** (`capillary_self_absorption_correction.py` and
   `capillary_sample_container_correction.py`): remove the `mask`/`Mask`
   fallback in the mask lookup (`get_first_present` with aliases) and the
   matching special case in `dependency_contract()`. Read exactly the
   configured key.
5. **`IndexedAverager.mask_key`:** today it defaults to `"Mask"` and silently
   skips masking when that key is absent
   (`indexed_averager.py`, about line 297).
   A pipeline that names its mask `mask` (the default everywhere else) but
   omits `mask_key` gets no averager masking and no warning. Every maintained
   MoDaCor-examples pipeline sets `mask_key` explicitly to avoid this.
   Change it to:
   - default `None`, meaning no mask, with the documentation saying so;
   - a configured `mask_key` that is missing from the bundle raises
     `KeyError` naming the key and the bundle, instead of being skipped;
   - an updated dependency contract, so `None` adds no read.

   Check `FindScaleFactor1D`, `ApplyMask`, `ReduceMask`, `BitwiseOrMasks` and
   `DilateMask` for the same "skip if absent" pattern on configured keys.
   Report what you find in the notes, but change only `IndexedAverager`.

**Out of scope.** Configuration keys whose values legitimately accept several
forms, such as `XSGeometryFromPixelCoordinates.sample_z_override` taking a
value/units or NeXus mapping. They are one input with a structured value, not
aliases.

**Compatibility.** This removes public configuration keys and changes a
default, so it is a breaking change for semantic-release. The commit or the
pull-request merge must carry a `BREAKING CHANGE:` footer listing the removed
keys and the new `IndexedAverager.mask_key` default. Add a section to
`docs/processing/interface-migrations.md`, following its existing style.

**Acceptance criteria.**

- No step's arguments documentation contains "alias" (verify with the
  reviewer's alias scan in the review log), and the capillary modules no
  longer contain `"Mask"`.
- Tests:
  - each removed key is rejected with `KeyError`;
  - the capillary steps do not find a `Mask` entry when configured for
    `mask`;
  - `IndexedAverager` with `mask_key=None` ignores a present mask, with a
    configured key it applies the mask, and with a configured but missing key
    it raises `KeyError`;
  - `dependency_contract()` assertions for the changed steps.
- Reference pages regenerated (`--check` clean); module `__version__` values
  bumped; full suite and flake8 clean.
- Commit as `R01b: <summary>` with the `BREAKING CHANGE:` footer. Touch only
  R01b's files.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R01c — "Did you mean …?" hint for unknown configuration keys

**Status:** open (optional; after R01b)

**Problem.** With aliases removed, a misspelled key (for example
`polarisation_factor`) fails at construction with `KeyError` listing every
known key. That is correct but noisy for steps with many arguments.

**Approach.** In `ProcessStep`'s unknown-key check, add the closest matches from
`difflib.get_close_matches(key, known_keys, n=3, cutoff=0.6)` to the message,
for example `Unknown key 'polarisation_factor'. Did you mean
'polarization_factor'?`. Keep the existing message and exception type, so that
`KeyError` and its match patterns in tests keep working.

**Acceptance criteria.** A focused test in `tests/dataclasses/` shows the hint
for a near miss and no hint for an unrelated key; existing tests pass
unchanged.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R02 — `DilateMask` documents an alias that is rejected

**Status:** done

**Problem.** The `radius` argument documents the aliases `number_of_pixels` and
`number_of_pixels_to_dilate`, and `calculate()` reads both. Neither is
declared in the configuration schema, so `ProcessStep` rejects them with
`KeyError` at construction. That code is dead, and the documentation is wrong.

**Files.** `src/modacor/modules/base_modules/dilate_mask.py` (about lines 60–66
and 117); `tests/modules/base_modules/test_dilate_mask.py`; regenerated
reference page.

**Approach.** First check whether any pipeline YAML in this repo or in
MoDaCor-examples uses either alias:

```text
grep -rn "number_of_pixels" <repo roots>
```

- If none do, remove the alias reading and the alias mention in the docs.
  This is the preferred route.
- If some do, declare the aliases with `default: None` and resolve them
  explicitly, as in R01.

Record which route was taken and why.

**Acceptance criteria.**

- Schema, documentation, and runtime agree on which keys exist.
- A test asserts the chosen behavior: either `KeyError` for the removed alias,
  or the alias taking effect.
- The reference page is regenerated and passes `--check`.

**Implementation notes.**

- Searched both `/Users/bpauw/Code/MoDaCor` and
  `/Users/bpauw/Documents/BAM/Projects/2026/MoDaCor_examples` with
  `rg -n --hidden -g '!.git/**' 'number_of_pixels' <repo roots>` before
  editing. Found 6 matching lines in MoDaCor: 2 in `dilate_mask.py`, 1 in
  `docs/reference/modules/DilateMask.md`, and 3 in this tracker. The examples
  checkout had 0 matches; neither checkout had pipeline YAML using either
  alias.
- Chose the preferred removal route: removed the dead alias fallback and the
  alias documentation. `radius` remains the supported key; both aliases
  continue to raise `KeyError` at construction.
- Added tests for rejection of both aliases and for radius documentation that
  does not advertise them. Before the fix: 1 failed, 1 passed, 6 deselected,
  2 subtests passed. The documentation regression fails on the original code.
- Bumped the module version to `20261030.1` and regenerated the reference pages.
- Validation: DilateMask tests 8 passed, 2 subtests passed; full suite
  1023 passed, 1 skipped, 3 warnings, 2 subtests passed;
  `flake8 src tests` 0 violations; reference `--check` 0 errors;
  isort 0 errors; Sphinx build 0 warnings, 0 errors.
- Scope is limited to R02; no related items were changed.

**Review log.**

- 2026-10-30, `d7a33b6`: **done.**
  - Independent grep for `number_of_pixels` across MoDaCor, MoDaCor-examples
    and the maintainer's 2025/2026 project folders (YAML, notebooks, Python):
    the only hits are the new tests. Removal is safe; any pipeline that used
    an alias already failed with `KeyError`.
  - `DilateMask` tests: 8 passed. Full suite: 1023 passed, 1 skipped. flake8
    clean; reference-page `--check` clean.
  - The rejection test also passes on the old code, because rejection was
    already the behavior. The documentation assertion is the regression
    test, and it fails on the old code. Acceptable for a documentation-only
    defect.
  - Carried to R03: `_radius_from_config` still uses `cfg.get("radius", 1)`,
    a literal fallback that duplicates the schema default. R03 removes it.
  - Process: the commit touched only R02's files. Good.

---

## R03 — One source of truth for configuration defaults, with a guard test

**Status:** open. Do this after R01 and R02.

**Problem.** Defaults are seeded from the schema (`ProcessStepDescriber.arguments`
plus `CONFIG_KEYS`) when a step is constructed. Even so, modules contain about
124 calls of the form `configuration.get("key", fallback)` (`cfg.get` included).
The worst files are `capillary_self_absorption_correction.py` (24),
`capillary_sample_container_correction.py` (20), and `angle_to_q.py` (9).

The fallback is dead code when it equals the schema default. It is misleading
when it differs, because the seeded `None` wins; that is the R01 bug. Reads of
keys that are not declared can never succeed; that is the R02 bug. Nothing
currently prevents either from recurring.

**Files.** All exported `ProcessStep` modules under `src/modacor/modules/`
(not `deprecated/`); a new guard test, for example
`tests/modules/test_configuration_defaults.py`.

**Approach.**

1. Add a guard test that iterates over every `ProcessStep` exported by
   `modacor.modules`. For each, parse its source module with `ast` and find
   `.get("<literal>", <literal>)` and `["<literal>"]` reads on
   `self.configuration`, or on a local name assigned from it (for example
   `cfg = self.configuration`). Assert that:
   - every key read is in `effective_config_schema()`; and
   - no `.get(key, fallback)` is used where the fallback is a literal. Allow a
     fallback only where the schema default is `None` and the `None` case is
     handled explicitly, and mark it inline with a short reason comment.

   Reads from nested configuration dictionaries (for example
   `ReduceDimensionality`'s `uncertainty_estimation` mapping) are not reads of
   step keys; restrict the scan to the names above so they are not flagged.
2. Remove the redundant fallbacks: `cfg.get("key", x)` becomes `cfg["key"]`, or
   `cfg.get("key")` where `None` is a valid value. Where the fallback differs
   from the schema default, decide which one is correct, and move the correct
   value into the schema default. List each such case in the Implementation
   notes.

**Acceptance criteria.**

- The guard test exists, passes, and fails if either the R01 or the R02 defect
  is reintroduced. Demonstrate this once, locally, and note it.
- No literal fallbacks remain in exported steps, apart from commented
  exceptions.
- No behavior change other than the cases listed in the notes, which must each
  have a test.
- Reference pages are regenerated if any default changed.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R04 — Derive capillary sample μ through `BaseData`

**Status:** open

**Problem.** `CapillarySelfAbsorptionCorrection._sample_mu` (about lines
395–440) derives μ = −ln(T)/t from an absorption or transmission and a
thickness. It strips both inputs to floats, writes out the partial derivatives
by hand, and combines them with `np.hypot`. Both inputs already arrive as
`BaseData` (from `scalar_quantity_from_config_or_source`). `AGENTS.md`
requires this arithmetic to be done with `BaseData`, which supports `log`,
`__neg__`, `__rsub__`, and `__truediv__`.

**Scope boundary.** The finite-difference propagation through the attenuation
kernels (`nominal_and_finite_difference`, `combine_uncertainty_component`) is
legitimate, because `BaseData` cannot propagate through numerical quadrature.
Do not change it. Only the analytic μ derivation is in scope.

**Files.**
`src/modacor/modules/technique_modules/scattering/capillary_self_absorption_correction.py`
and its tests. See also `docs/development/design/capillary-self-absorption.md`.

**Approach.**

- Obtain the thickness and the absorption or transmission as `BaseData`. Add a
  `_scalar_basedata(...)` helper next to `_scalar_with_uncertainties`.
- Compute `transmission = 1 - absorption` where applicable, then
  `sample_mu = -transmission.log() / thickness`, all with `BaseData`.
- Convert to `1/m` and reduce to the existing `_ResolvedSampleMu`
  float-plus-uncertainty form only at the boundary where the kernel needs it,
  reusing the constant-valued uncertainty check.
- Keep all existing validation messages: thickness positive, absorption in
  [0, 1), transmission in (0, 1], and the mutually exclusive inputs.

**Acceptance criteria.**

- The manual derivative code and `np.hypot` are gone from `_sample_mu`.
- New tests compare μ and its uncertainties, for both the absorption and the
  transmission routes, against analytic values with independent thickness
  and factor uncertainties under distinct names and under a shared name. Use
  tight tolerances. If `BaseData` correlation semantics differ from the
  current behavior for a shared uncertainty name, stop and record the
  difference instead of choosing one.
- All existing capillary tests pass unchanged.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R05 — Test coverage for arithmetic steps and low-coverage helpers

**Status:** open

**Problem.** Steps that are widely used and simple to test are poorly covered:

| File | Line coverage |
| --- | --- |
| `divide_databundles.py` | 63 % |
| `multiply_databundles.py` | 61 % |
| `subtract_databundles.py` | 57 % |
| `negate.py` | 45 % |
| `solid_angle_correction.py` | about 70 % |
| `models/centroid.py` | 51 % |
| `models/integration.py` | partial |
| `modules/helpers/statistics.py` | 40 % |
| `io/buffer/codec.py` | 26 % |

**Approach.** Add focused tests only. Production changes are limited to fixing
genuine bugs that the tests expose, recorded in the notes. For each step, cover:

- units and uncertainty propagation through `BaseData`;
- the error paths (missing keys, incompatible shapes or units); and
- a `dependency_contract()` assertion of `source_refs`, `processing_reads`, and
  `processing_writes`, following `tests/modules/base_modules/test_apply_mask.py`.

**Acceptance criteria.**

- Each listed file reaches at least 90 % line coverage, or the notes explain
  which lines are unreachable.
- Overall coverage does not drop.
- Command:
  `.venv-dev/bin/python -m pytest -q --cov=src/modacor --cov-report=term-missing <touched test files>`

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R06 — Repository hygiene

**Status:** open. The maintainer owns this item; the implementer only proposes.

**Problem.** The checkout contains untracked artefacts that are easy to commit
by accident:

- draw.io `.bkp` files, `docs/Classes.drawio.pdf`, PNG/JPG images, and
  `docs/logo/` in `docs/`;
- a `docs/design/` folder alongside the tracked `docs/development/design/`;
- `gendocs_test.sh`, `uv.lock`, and `MoDaCor_and_friends.code-workspace` at the
  root;
- `tests/testdata/`, `dist/`, and stale `__pycache__` files built for Python
  3.10, which the package no longer supports (`requires-python >= 3.12`).

**Approach.** Propose `.gitignore` additions (`*.bkp`, `*.code-workspace`,
`dist/`, and so on). List, for the maintainer to decide, each untracked item:
track it, ignore it, or delete it. In particular, decide whether `uv.lock` is
tracked or ignored, and whether `docs/design/` duplicates the tracked
`docs/development/design/`. Delete nothing without the maintainer's approval.

**Acceptance criteria.** `git status` on a fresh build-and-test checkout shows
only intended changes.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R07 — Declarative fixed-key dependencies

**Status:** decision needed (maintainer)

**Problem.** 35 of the 40 exported steps override `dependency_contract()`; only
6 use `dependency_role`. The reason is structural, not neglect. A
`dependency_role` attaches to a configuration value, but most steps read and
write fixed `BaseData` keys: `"signal"` (hard-coded in 27 steps), `"Omega"`,
mask keys, and so on. The schema has no way to declare those, so each step
writes a near-identical override. `Multiply` and `SolidAngleCorrection` are
typical examples. Every override is a hand-maintained duplicate that can drift
from `calculate()`.

**Options.**

- **A. Recommended:** declare fixed keys in the schema. Add explicit
  `ProcessStepDescriber` fields, for example
  `fixed_processing_reads: list[str]` and `fixed_processing_writes: list[str]`.
  The default `dependency_contract()` expands them against
  `with_processing_keys`, combined with any `dependency_role` keys and the
  existing source-ref extraction. Keep these fields separate from `modifies`
  and `required_data_keys`, which `AGENTS.md` says must not be confused with
  the runtime contract. Steps with genuinely irregular relationships (sinks,
  cross-bundle steps, `output_processing_key` remapping) keep their overrides.
- **B.** Make `signal_key` configurable in every step and use roles. This is
  more flexible, but it changes the public configuration surface of 27 steps,
  for little demonstrated user need.
- **C.** Keep the overrides, and add a test that each contract matches what
  `calculate()` actually touches. This is the cheapest option but leaves the
  boilerplate.

**Approach (option A, once approved).**

1. **Safety net first.** Add a snapshot test that records the current
   `dependency_contract()` of every exported step for one or two representative
   configurations each. This defines the behavior to preserve.
2. Add the describer fields, the default-contract expansion, documentation in
   the module author guide (dependency section), and unit tests in
   `tests/dataclasses/`.
3. Migrate steps in small batches: base arithmetic, then masks, then scattering
   corrections. Delete each override whose contract the schema now reproduces
   exactly. The snapshot must stay identical; any intentional difference is
   listed and justified.
4. Update the `module-author-guide.md` and correction-module-template guidance.

**Acceptance criteria.**

- The snapshot test passes unchanged, apart from listed and justified
  differences.
- The number of remaining overrides is recorded, each with a one-line reason
  (target: at most 12).
- Docs and template updated; reference pages regenerated.

This item should get its own branch and pull request.

**Decision.** _(maintainer)_

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R08 — Audit remaining unit stripping in modules

**Status:** open. Do this after R03 and R04.

**Problem.** Several modules convert to raw magnitudes with
`.to(...).magnitude`, `.value`, or `np.asarray(...)`:
`capillary_sample_container_correction.py` (about line 568),
`find_center_of_mass_1d.py` (about lines 196–203),
`find_scale_factor1d.py` (about lines 294–299),
`subtract_interpolated_1d.py` (about line 197), `indexed_averager.py`,
`polarization_correction.py` (about line 139), and `plot_1d_visualization.py`.

Many of these are legitimate: converting a configuration scalar into the units
of an axis, preparing arrays at the boundary of a `models` kernel, or plotting.
Some may be reimplementing `BaseData` arithmetic.

**Approach.** Classify every occurrence in a table in the Implementation notes:

- **(a) legitimate:** a configuration scalar, a kernel boundary, or display;
- **(b) should use `BaseData`;**
- **(c) a kernel embedded in a module** that belongs in `models` or `geometry`.

Fix (b) and (c) occurrences, one commit each. Extract repeated "configuration
scalar plus units → axis units" conversions into one helper under
`modacor.modules.helpers`. The existing
`helpers/scattering/material_attenuation.scalar_in_units` is a candidate to
generalize.

**Acceptance criteria.** Every occurrence is classified; (b) and (c) are fixed
with tests showing that units and uncertainties propagate; no duplicate
conversion helpers remain.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R09 — Narrow broad `except Exception` handlers

**Status:** open

**Problem.** There are broad `except Exception` handlers in
`server/runtime_service.py` (7), `io/tiled/tiled_source.py` (3),
`server/chunked_outputs.py`, `runner/pipeline.py`, `io/hdf/hdf_processing_sink.py`,
`client/local_server.py`, `capillary_self_absorption_correction.py`, and
`helpers/scattering/material_attenuation.py` (2 each). In the server, some are
deliberate top-level boundaries that record a failed run. Others wrap a single
Pint or I/O call and turn every error, including programming errors, into a
`ValueError`.

**Approach.** Classify each handler:

- **(a) a deliberate boundary:** keep it, make sure it logs the traceback or
  keeps it in the run record, and add a one-line comment;
- **(b) narrowable:** catch the specific types, such as
  `pint.DimensionalityError`, `pint.UndefinedUnitError`, `KeyError`, `OSError`,
  or `h5py` errors. Always chain with `from exc`.

**Acceptance criteria.** A classification table in the notes; every (b) handler
narrowed, with a test that an unexpected exception type now propagates. No
change to the server's API error responses (`tests/server` unchanged).

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R10 — Conventions

**Status:** open. Low priority; a good filler task.

**Scope.**

1. **`attrs` vs `dataclasses`.** `_ResolvedSampleMu` in
   `capillary_self_absorption_correction.py` and the records in
   `modules/helpers/statistics.py` and
   `modules/helpers/scattering/material_attenuation.py` become `attrs`
   classes, following `AGENTS.md`. Plain internal records in `server/`,
   `client/`, `io/`, `geometry/`, and `models/` that do no validation may stay;
   do not churn them.
2. **File headers.** Pick one header convention, either the full
   authors/copyright/date/status block or SPDX-only, and record it in the
   contribution checklist. Apply it only to files already touched by other
   items. Do not do a mass rewrite.
3. **`step_reference`.** 13 steps cite the same general review
   (DOI 10.1088/0953-8984/25/38/383201) and 8 are empty. Where a more specific
   reference exists, add it; otherwise keep the general review, but do not
   leave the field empty. The maintainer supplies or approves the references.

**Acceptance criteria.** As in the scope above; reference pages regenerated.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R11 — Split `RuntimeService` into focused services

**Status:** open. Do this after R09.

**Problem.** `server/runtime_service.py` is 1516 lines, essentially one class.
It covers health and policy, session CRUD, source and sink management, buffer
I/O, chunked-output lifecycle, process planning, execution, failure handling,
and the auto-fallback path. This makes changes risky and review hard.

**Approach.** A behavior-preserving extraction, keeping `RuntimeService` as a
thin facade so `server/api.py` and the client are untouched:

- `ChunkedOutputService`: initialize, reopen, inspect, finalize, recover, detach;
- `BufferService`: the `put_buffer_*`, `get_buffer_*`, and `clear_buffer*`
  methods, plus plot payloads;
- `SessionResourceService`: sources, sinks, and the sample source;
- `ProcessExecutor`: `process`, dry run, reset, recover, the `_prepare`,
  `_execute`, `_finalize`, and `_handle_process_failure` methods, and
  `_run_auto_fallback`.

Move one service per commit, with no logic edits in the same commit.

**Acceptance criteria.**

- `tests/server` and `tests/client` pass unchanged.
- No module over about 600 lines.
- No public method signature changes on `RuntimeService`.

This item should get its own branch and pull request.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R12 — Large I/O files

**Status:** deferred

`io/hdf/hdf_chunked_processing_sink.py` (1431 lines) and `io/chunking.py`
(1250 lines) are large but cohesive, recent, and well covered by the chunked
design notes. Revisit them only if they become a change bottleneck, or after
R11 shows the split pattern pays off.

---

## E01 — MoDaCor-examples: slim I22 preprocessing helpers

**Status:** open. This item is in the MoDaCor-examples repository; give it its
own branch there.

**Problem.** `DLS/I22/i22_helpers.py` is 1034 lines, including a roughly
290-line `preprocess_measurement` and a `calibrated_mean_transmission`.
`AGENTS.md` asks for lean I22 preprocessing: reshape or summarize
incompatible metadata, and let MoDaCor modules do the geometry and corrections.

**Approach.** First produce an inventory, without changing code. List each
helper as (a) metadata reshaping or summarizing, which is fine; (b) numerics
that an existing MoDaCor step already does, which should be replaced; or
(c) numerics with no MoDaCor equivalent, which is a candidate for a new step in
MoDaCor. The reviewer and maintainer then decide what to migrate. Validate any
replacement against the existing I22 run outputs within stated tolerances.

Also found on 2026-10-30: `DLS/B21/pipelines/B21_frame_prefilter.yaml` uses
`module: FramewiseIndexedAverager`, which does not exist in MoDaCor, so that
pipeline cannot run. Include it in the inventory: either port it to
`IndexedAverager` or mark it as unmaintained.

**Acceptance criteria.** Inventory reviewed; migrated paths reproduce previous
outputs; any remaining DAWN-derived constants keep their provenance comments.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

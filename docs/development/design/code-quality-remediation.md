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
| R01b | One input per setting: remove configuration aliases and the mask-key fallback | simplification | S | high | done |
| R01c | "Did you mean …?" hint for unknown configuration keys | usability | XS | medium | open (optional) |
| R02 | `DilateMask` documents an alias that is rejected | bug | XS | medium | done |
| R03 | One source of truth for configuration defaults, with a guard test | consistency | S | high | done |
| R04 | Derive capillary sample μ through `BaseData` | contract | S | medium | done |
| R05 | Core-container validation and untested contracts (rescoped) | bug + tests | S | medium | done |
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

**Status:** done

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
- `chemical_composition`: no uses. `material` (the key it aliased) is used
  in the I22 pipelines; see point 2c.
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
2. **`AttenuatorPlateCorrection` and `DetectorEfficiencyCorrection`: rename
   the material input to `chemical_formula` (maintainer decision, 2026-10-30).**
   - What it is: a chemical formula in xraylib syntax (for example `Si`,
     `CdTe`, `SiO2`, `Ca5(PO4)3OH`), passed to `xraylib.CS_Total_CP` and
     combined with `density` to give μ. It is not a mass-fraction composition. (Correction, 2026-10-30: xraylib's
     `CS_Total_CP` also accepts *exact* names from its NIST compound database,
     for example `"Kapton Polyimide Film"`; informal names such as `"Silicon"`
     or `"Kapton"` are rejected.)
   - Replace `material`, `material_source`, `chemical_composition`,
     `chemical_composition_source` and the undeclared `composition` with exactly
     `chemical_formula` and `chemical_formula_source`. Today
     `material_attenuation_from_config` resolves `chemical_composition`, then
     `composition`, then `material`, so the alias overrides the documented key.
     Afterwards it reads only the one name.
   - Document it as: "Chemical formula in xraylib syntax, e.g. `SiO2`; not a
     material name." Rename the `MaterialAttenuation.material` field to
     `chemical_formula` for consistency.
   - Remove the `except Exception` retry through `xraylib.SymbolToAtomicNumber`.
     `CS_Total_CP` already parses bare element symbols, and the retry turns
     formula typos into a less clear element-lookup error. A parse failure
     should raise `ValueError` naming the formula, chained to xraylib's error.
     Confirm with a test that `"Si"` still works and that `"Sii"` raises.

   **2b. Value and `_source` are mutually exclusive.** In
   `modules/helpers/scattering/material_attenuation.py`,
   `scalar_quantity_from_config_or_source` lets `<name>_source` override
   `<name>`, while `text_from_config_or_source` lets `<name>` override
   `<name>_source`: opposite rules in one file. Under this item's principle,
   setting both is a configuration error. Both helpers raise `ValueError`
   naming the two keys when both are set and not `None`. This also applies to
   the capillary corrections, which share
   `scalar_quantity_from_config_or_source`. Also replace the broad
   `except Exception` in `text_from_config_or_source` (static metadata, then
   data): ask the source for the reference's capability explicitly, or catch
   only the specific "not static metadata" error. Describe the approach in the
   notes.

   **2c. Update existing pipelines in step.** Measurement, calibration and
   mask files are read-only and are not modified; the I22 and B21 naming is
   not ours to change. Only pipeline YAMLs change, on branch
   `quality-remediation-cycle` in MoDaCor-examples, committed as
   `R01b: rename material to chemical_formula`. The reviewer checked the
   files on 2026-10-30:

   - **MOUSE** (`BAM/MOUSE/pipelines/MOUSE_solids.yaml`,
     `MOUSE_solids_operando.yaml`) already read the formula from the
     measurement file. Rename `material_source:` to
     `chemical_formula_source: sample::/entry1/instrument/detector00/sensor_material`.
     The field is NeXus `NXdetector/sensor_material` and holds `'Si'`, a valid
     formula. Do not use a `sensor/material` path; it does not exist.
   - **I22** (`DLS/I22/pipelines/`: `I22_SAXS_solids_operando.yaml`,
     `I22_SAXS_WAXS_solids_operando.yaml`,
     `I22_WAXS_solids_operando_standard_saxs_waxs.yaml`,
     `I22_WAXS_solids_operando_usaxs_saxs_waxs.yaml` and
     `I22_WAXS_DAWN_crosscheck.yaml`). The I22 measurement, calibration and
     mask files contain no sensor-material or attenuator field, so a
     `_source` pointer is not possible. Rename `material:` to
     `chemical_formula:`, keep the literal values (`Si` for the Pilatus
     sensors, `Al` for the attenuator plate), and make sure each has a
     provenance comment nearby (detector specification, or the existing DAWN
     comment for `Al`), as `AGENTS.md` requires for DAWN-derived constants.
   - Grep the whole examples repository, including notebooks and
     `i22_helpers.py`, for any remaining `material`, `material_source` or
     `chemical_composition` configuration keys.
   - Run the examples repository's own tests (`tests/`) and record the counts.
   - The maintainer's experiment folders are the maintainer's to update; list
     the affected files in the notes. Known so far:
     `20260901_I22_experiments/processing_pipelines/I22_SAXS_solids_operando.yaml`,
     `I22_WAXS_solids_operando.yaml` and `I22_WAXS_DAWN_crosscheck.yaml`.

   Note for future `_source` use: NeXus string fields may hold a material
   name rather than a formula. For example, the MOUSE collimator `material`
   fields hold `'Germanium'` and `'Silicon'`. These are not valid
   `chemical_formula` values. With point 2's error handling they fail with a
   clear `ValueError` instead of a misleading element-lookup error.

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

- No step's configuration schema contains `material`, `chemical_composition`
  or `composition`; the helper resolves only `chemical_formula`.
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
  - setting both a value and its `_source` raises `ValueError`, for one
    scalar parameter and for `chemical_formula`;
  - `"Si"` resolves and `"Sii"` raises a clear `ValueError`;
  - `dependency_contract()` assertions for the changed steps.
- Reference pages regenerated (`--check` clean); module `__version__` values
  bumped; full suite and flake8 clean.
- Commit as `R01b: <summary>` with the `BREAKING CHANGE:` footer. Touch only
  R01b's files.

**Implementation notes.**

- Implemented only R01b. MoDaCor implementation commit:
  `a573107513411dad297ec502f0053e5bb590d2e3`
  (`R01b: enforce single configuration inputs and explicit mask selection`),
  including the `BREAKING CHANGE:` footer and interface-migration section.
  This follow-up records the completed work and moves the item to review.
- MoDaCor-examples commit on `quality-remediation-cycle`:
  `3bd19d04dd00e2136b94685fb69651b6f840abd3`
  (`R01b: rename material to chemical_formula`). Changed only the seven
  pipeline YAMLs listed in point 2c; preserved unrelated existing checkout
  edits. Both MOUSE formula sources now use
  `sample::/entry1/instrument/detector00/sensor_material`, as specified.
  I22 retains its Si/Al literals, with nearby Pilatus specification or
  provisional DAWN provenance comments, including the commented-out Al step.
  Measurement, calibration and mask files were not modified.
- Removed the three British configuration spellings (retaining the search
  keyword), the threshold alias, and the retired material/composition names.
  Attenuation resolves only `chemical_formula`; the result container uses
  the same field name and now uses attrs. Bare `Si` works; `Sii` raises a
  formula-naming `ValueError` chained to xraylib's `ValueError`, with no
  element-lookup retry.
- Both shared scalar/text readers reject non-`None` value/source pairs.
  Text resolution first requests static metadata, catching only
  `NotImplementedError`, the base `IoSource` unsupported-capability error,
  before reading data. Missing references, file errors and parse/decode errors
  propagate instead of triggering a second read. Tests exercise HDF string
  datasets/attributes, the unsupported-capability path and error propagation.
- Capillary lookup and dependencies now use only the configured mask key.
  Existing mask fixtures use lowercase `mask`; regressions verify a lone
  `Mask` is ignored and that both keys remain distinct when present together.
  `IndexedAverager.mask_key` defaults to `None`, ignores present masks unless
  configured, and names the key and input bundle when a configured mask is
  missing. Its existing dependency builder omits `None`; tests assert this
  and the configured-mask contract. All seven step versions were bumped;
  the touched helper also declares `20261030.1`.
- Configured-key audit: `FindScaleFactor1D` has no mask input and indexes its
  configured signal/axis keys directly. `ApplyMask`, `ReduceMask`,
  `BitwiseOrMasks` and `DilateMask` also index their configured inputs
  directly and raise for missing keys; none silently skip them. No changes
  were made to these five modules.
- Scanned the entire examples repository's YAML, Python and notebook code
  for retired material/material_source/chemical_composition configuration
  keys: 0 remaining matches. Scanned the maintainer's 2025/2026 experiment
  trees, decoding notebook code cells to distinguish configuration from
  prose. Affected experiment files (left for the maintainer to update;
  paths relative to `/Users/bpauw/Documents/BAM/Projects/2026/`):
  - `20260901_I22_experiments/processing_pipelines/I22_SAXS_solids_operando.yaml`
  - `20260901_I22_experiments/processing_pipelines/I22_WAXS_solids_operando.yaml`
    (including its commented-out Al step)
  - `20260901_I22_experiments/processing_pipelines/I22_WAXS_DAWN_crosscheck.yaml`
  - `20260904_MOUSE_modacor/processing_pipelines/MOUSE_solids.yaml`
  - `20261009_N008_MoDaCor/pipelines/MOUSE_solids.yaml`
  - `20261009_N008_MoDaCor/pipelines/MOUSE_solids_operando.yaml`
- Acceptance scans over exported public `ProcessStep.documentation.arguments`:
  0 descriptions containing `alias` (case-insensitive), 0 schemas containing
  `material`, `chemical_composition` or `composition`; capillary module
  source contains 0 `"Mask"` literals. Retired geometry's reference to pixel
  unit aliases describes accepted values, outside R01b's configuration-key
  scope.
- Regression selection against the original committed source: 21 failed,
  3 passed, 76 deselected. The already-undeclared `composition` rejection
  passes before the change; the new behaviors fail before it.
- Validation: touched modules' tests 100 passed; full suite 1047 passed,
  1 skipped, 3 warnings, 2 subtests passed; examples `tests/` 12 passed;
  `flake8 src tests` 0 violations; regenerated reference `--check` 0 errors;
  isort 0 errors; Sphinx 0 warnings, 0 errors. Repository pre-commit hooks
  also passed.

**Review fixes.** Addressed requested changes 1–4 only: examples commit
`382989541dd6aecb7c8d999a4203e78923c51fce` restores the MOUSE operando
`sample_static::` formula source; MoDaCor commit
`5b4242d4b896eccadad929570e51e6bd1986bd03` makes capillary masks optional
by default and missing configured masks errors (retaining empty-mask default
derivation), removes the unused helper, corrects NIST-name documentation,
bumps touched module versions, and updates migration/reference pages.
Regression checks before the fix: 8 failed, 45 deselected. Validation:
touched tests 102 passed; full suite 1063 passed, 1 skipped, 3 warnings,
2 subtests passed; examples `tests/` and `DLS/I22/tests/` 23 passed;
flake8 0 violations, reference `--check` 0 errors, isort 0 errors,
Sphinx 0 warnings/0 errors. Unrelated checkout edits were preserved.

**Review log.**

- 2026-10-30, MoDaCor `a573107` + `b0aa410`, MoDaCor-examples `3bd19d0`:
  **changes requested.**

  Verified: full suite 1047 passed, 1 skipped; flake8 clean; reference
  `--check` clean; Sphinx `-W` clean; alias scan over exported step
  arguments clean; no `"Mask"` literals in the capillary modules; examples
  tests 23 passed (`tests/` and `DLS/I22/tests/`). xraylib behaviour
  confirmed: `Si`, `Al`, `SiO2` parse; `Sii`, `si`, `Silicon`, `Germanium`
  raise `ValueError`. `NotImplementedError` is the documented `IoSource`
  unsupported-capability signal (io-source-sink guide), so the narrowed
  catch is correct. The `IndexedAverager` missing-key check, the value and
  `_source` exclusivity, and the `BREAKING CHANGE:` footer are all good.
  The audit of the other mask steps was useful.

  Requested changes:

  1. **Regression (caused by the reviewer's brief):**
     `BAM/MOUSE/pipelines/MOUSE_solids_operando.yaml` originally read the
     formula from `sample_static::`. The brief prescribed `sample::` for both
     MOUSE files, and the commit followed it. Every other detector input of
     that step (`thickness_source`, `wavelength_source`, units) reads from
     `sample_static`. Restore
     `chemical_formula_source: sample_static::/entry1/instrument/detector00/sensor_material`
     in that file only. `MOUSE_solids.yaml` correctly uses `sample::`.
  2. **Capillary missing-mask silent skip** (scope gap in point 5, which
     named only `IndexedAverager`). `_active_mask` in both capillary
     corrections returns an all-active mask when the configured `mask_key`
     (default `"mask"`) is absent. That is the same silent skip R01b removed
     from `IndexedAverager`. Apply the same rule to `mask_key` in
     `CapillarySelfAbsorptionCorrection` and to `mask_key` and
     `empty_mask_key` in `CapillarySampleContainerCorrection`: default `None`
     means no mask, and a configured key that is missing raises `KeyError`
     naming the key and bundle. Keep `empty_mask_key: None` meaning "use
     `mask_key`"; that is a documented default derivation, not an alias.
     Update the dependency contracts, tests, reference pages and the
     migration and `BREAKING CHANGE:` text.
  3. **Dead helper:** `modacor.modules.helpers.get_first_present` has no
     users left. Remove it, its `__all__` entry and any tests.
  4. **Documentation accuracy:** the `chemical_formula` and
     `chemical_formula_source` docs say "not a material name". That wording
     came from the reviewer's incorrect description. Replace it with:
     "Chemical formula in xraylib syntax, e.g. `SiO2`. Exact xraylib NIST
     compound names (e.g. `Kapton Polyimide Film`) are also accepted."
     Apply this in both steps, then regenerate the reference pages.

  Commit these as `R01b: review fixes` in MoDaCor, plus a separate examples
  commit for item 1.

- 2026-10-30, MoDaCor `5b4242d` + `5432355`, MoDaCor-examples `3829895`:
  **done.**
  - Item 1: `MOUSE_solids_operando.yaml` reads from `sample_static::` again;
    `MOUSE_solids.yaml` still uses `sample::`.
  - Item 2: the schema defaults for both capillary `mask_key` settings are
    now `None`, as is `empty_mask_key` (which falls back to `mask_key`). A
    configured key that is missing raises `KeyError` naming the key and the
    bundle. Tests cover the default with no mask, applied masks, missing
    configured and inherited keys, and the dependency contracts.
    `BREAKING CHANGE:` footer and migration text present.
  - Item 3: `get_first_present` removed; no references remain.
  - Item 4: the NIST-name wording is corrected in both steps.
  - Verified: full suite 1063 passed, 1 skipped; flake8 clean; reference
    `--check` clean; Sphinx `-W` clean; examples tests 23 passed.
  - Maintainer action after merge: rename `material` / `material_source` in
    the six experiment pipelines listed in the Implementation notes. Any
    experiment pipeline using a capillary correction must now set
    `mask_key: mask` to keep masking; none were found on 2026-10-30.


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

**Status:** done

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

   Also treat as reads the key names passed as string literals to the shared
   configuration helpers (`scalar_quantity_from_config_or_source`,
   `text_from_config_or_source`, and any other helper taking `cfg` plus a
   key name), including the derived `<name>_source`, `<name>_units` and
   `<name>_units_source` keys those helpers read. The undeclared
   `composition` key found during R01b was read through such a helper, so a
   scan limited to `self.configuration` would have missed it.

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

**Implementation notes.**

- R03a checkpoint: added `tests/modules/test_configuration_defaults.py`;
  production code, defaults, versions and reference pages were unchanged.
  The following baseline and probe results record that accepted checkpoint.
- Scans the 41 exported, non-deprecated steps. Tracks `self.configuration`,
  its local aliases (including `or {}`), and parameters receiving that mapping.
  Follows local methods and shared helpers with literal key-name arguments,
  including derived `_source`, `_units` and `_units_source` reads. Nested
  mappings and unrelated local dictionaries are excluded. Shared helpers are
  checked for declared reads, not for their reusable fallback policy.
- Baseline: **260 violations**, each explicitly listed by step, key, problem
  and source location in an exact allowlist: **252 literal fallback diagnostics
  and 8 undeclared-key diagnostics**. Four occurrences have both problems.
  New diagnostics and stale allowlist entries fail. R03b must empty this list.
- **Fallback/default differences:** none for declared keys; all 248 literal
  fallback diagnostics for declared keys equal the effective schema default.
  No changed default has been chosen in R03a.
- Four additional `{}` fallbacks have **no schema entry**, rather than a
  differing schema default, all in
  `CapillarySelfAbsorptionCorrection._scalar_with_uncertainties`:
  `sample_mu_uncertainties_sources`, `sample_radius_uncertainties_sources`,
  `wall_mu_uncertainties_sources`, `wall_thickness_uncertainties_sources`.
  Fallback: `{}`; schema: undeclared; chosen value: deferred to R03b.
- The other four undeclared helper reads are
  `sample_phase_absorption_units`, `sample_phase_absorption_units_source`,
  `sample_phase_transmission_units`, `sample_phase_transmission_units_source`
  in `CapillarySelfAbsorptionCorrection`. R03b must resolve these reads as
  part of emptying the allowlist; no scientific behavior was changed here.
- Temporary regression probes each produced **1 failed** guard test, then
  were reverted before validation and commit. Representative failing lines:

  ```text
  DilateMask: number_of_pixels: undeclared configuration key (dilate_mask.py:117)
  AttenuatorPlateCorrection: composition: undeclared configuration key (material_attenuation.py:93)
  PolarizationCorrection: polarization_factor: literal fallback 0.5; schema default None (polarization_correction.py:110)
  ```

  The R02 probe nested `cfg.get("number_of_pixels", 1)` in the radius read.
  The helper probe replaced the `chemical_formula` text-helper argument with
  `composition`; it also caught `composition_source` and both reads for
  `DetectorEfficiencyCorrection`. The R01 probe restored the misleading 0.5
  fallback against the seeded `None` default.
- Guard self-tests check all four scalar-helper key forms and both text-helper
  forms, exclusion of nested mappings, and the requirement for a `None` schema
  default plus an inline `# config-default: <reason>` marker for exceptions.
  Any such R03b exception must also retain explicit handling of `None`.
- Validation: guard **11 passed**; affected-module tests **76 passed,
  2 subtests passed**; full suite **1074 passed, 1 skipped, 3 warnings,
  2 subtests passed**; `flake8 src tests` **0 violations**; isort **0 errors**.
- Scope: only the new guard test and R03's tracker entry/overview status.

**R03b implementation notes.**

- The guard now starts at the three pipeline entry points, resolves inherited
  and overridden methods with `getattr(step, name)`, and follows reachable
  `self.`/`cls.` calls and configuration-bearing shared-helper calls. Added
  self-tests for inherited undeclared reads and exclusion of overridden parent
  methods; the existing shared-helper and fallback-policy self-tests remain.
- The reachability baseline was **281 diagnostics**: the original 260 plus
  **21 inherited diagnostics** for `CapillarySampleContainerCorrection`
  (**12 literal fallbacks and 9 undeclared reads**). No additional undeclared
  key required a decision. The four unreachable undeclared reads from the
  initial MRO checkpoint are excluded, as are its two associated fallback
  diagnostics (`input_state` and `transmission_uncertainties_sources`).
- The inherited reachable fallbacks were `capillary_axis`, `capillary_centre`,
  `capillary_centre_units`, `sample_mu_sensitivity_relative_step`, and the
  `_uncertainties_sources` reads for `empty_centre_mu`, `sample_mu`,
  `sample_radius`, `wall_mu`, `wall_thickness`, `sample_phase_absorption`,
  `sample_phase_transmission` and `sample_phase_thickness`. All 14 fallbacks
  from the MRO checkpoint, including the two unreachable ones, were removed
  at their source occurrences. Removed **232 literal fallback calls** across
  **35 production modules**; `ALLOWED_VIOLATIONS` is empty. No declared default
  changed and no fallback exception was needed.
- `_scalar()` now calls the scalar helper without uncertainty sources. The
  four original dead uncertainty-source reads, plus the inherited
  `empty_centre_mu` read, disappear rather than being declared or commented
  out. Existing uncertainty handling for derived phase-factor inputs remains.
- Declared `sample_phase_{absorption,transmission}_units` (`str`, default
  `"dimensionless"`) and their `_units_source` keys (default `None`) in the
  parent's arguments, inherited by the container. This is the only intended
  behavior change: both corrections accept percent absorption/transmission
  from configuration or an IoSource. Eight parameterized cases verify the
  conversion to dimensionless and the source dependency references.
- Both capillary `step_note`s now state the direct-parameter uncertainty
  limitation, including empty-centre mu for the container. Only these two
  modules' versions were bumped, to **20261030.3**, because their public
  metadata changed. Regenerated both reference pages.
- AppendSource's tests now use the normal constructor and configuration API
  instead of `__new__` and replacement dictionaries that bypassed schema
  defaults; this preserves their existing behavior assertions.
- Validation: guard **13 passed**; touched-module tests **326 passed,
  2 subtests passed**; full suite **1084 passed, 1 skipped, 3 warnings,
  2 subtests passed**; `flake8 src tests` **0 violations**; generated reference
  `--check` **0 errors**; isort **0 errors**; Sphinx build **0 warnings,
  0 errors**.
- Scope: R03 production fallback removals, capillary metadata/scalar adapter,
  focused tests, generated capillary reference pages, and this tracker entry.

**Review log.**

- 2026-10-30, `a0740b6` (R03a, checkpoint): **accepted; proceed to R03b
  with the directions below.**
  - Verified: guard 11 passed; full suite 1074 passed, 1 skipped; flake8
    clean. The scanner design is sound: it follows aliases, own methods and
    shared helpers, excludes nested mappings, and has self-tests and probes
    that reproduce the R01, R02 and `composition` defects.
  - ~~Known limitation, acceptable~~ **Correction (reviewer, same day):**
    only methods in the step's own `__dict__` are scanned. This is *not*
    harmless. `CapillarySampleContainerCorrection` subclasses
    `CapillarySelfAbsorptionCorrection` and inherits `_sample_mu`,
    `_scalar_with_uncertainties`, `_active_mask` and more, so those reads
    are never checked against the container step's schema. See direction 6.
  - All 248 declared-key fallbacks equal the schema default, so removing
    them changes no behavior.
  - The 8 undeclared reads are dead code, not lost features. The reviewer
    confirmed that configuring any of them raises `KeyError`:
    - the four `{sample_mu, sample_radius, wall_mu, wall_thickness}_uncertainties_sources`
      reads happen in `_scalar_with_uncertainties`, called via `_scalar()`,
      which then discards the uncertainties (`sample_mu` given directly is
      returned with `{}`, about line 405);
    - the four `sample_phase_{absorption,transmission}_units[_source]` reads
      come from the generic scalar helper for parameters that do not declare
      units.

  **R03b directions:**

  1. Remove all literal fallbacks (mechanical, no behavior change). Empty
     the allowlist.
  2. Make `_scalar()` stop reading `<name>_uncertainties_sources`: call the
     scalar helper without uncertainty sources instead of discarding them
     afterwards. No behavior change.
  3. Declare `sample_phase_absorption_units` and
     `sample_phase_transmission_units` (type `str`, default
     `"dimensionless"`) and their `_units_source` keys (default `None`), as
     every other scalar parameter of this step already has. This also lets a
     source supply a transmission in percent. Add a test with a value in
     `percent` that converts correctly.
  4. Do not add uncertainty support in R03b. Maintainer decisions,
     2026-10-30: `sample_mu_uncertainties_sources` for directly given μ
     is added in **R04**. Radius, wall μ and wall thickness uncertainty
     remain planned future work, recorded in `capillary-self-absorption.md`
     ("Uncertainties and diagnostics"). Remove the four dead reads; do not
     comment them out.
  5. State the limitation for users: add one sentence to the `step_note` of
     both capillary corrections, for example "Uncertainties of directly
     configured sample/wall μ, capillary radius and wall thickness are not yet
     propagated; see the capillary design note." Regenerate the reference
     pages. (In `CapillarySampleContainerCorrection`, name only the
     parameters it actually has.) R04 removes "sample μ" from this
     sentence when it adds that support.
  6. **Scan reachable inherited methods.** Start from `calculate`,
     `dependency_contract` and `prepare_execution`, resolving each with
     `getattr(step, name)` so overrides win. Follow `self.`/`cls.` method calls
     with the same resolution and follow shared-helper calls receiving the
     configuration. Check each read against the scanned step's schema; do not
     scan unreachable methods. Test both an inherited undeclared read and an
     overridden parent method whose undeclared read must not be reported.
     Rerun the baseline and record inherited violations. Resolve fallbacks as
     in direction 1 and the checkpoint's nine undeclared keys as in directions
     2–3; stop and report any other undeclared key before declaring it.

- 2026-10-30, R03b checkpoint (uncommitted; Codex stopped as directed): the
  inherited scan reported 13 new undeclared keys and 14 more fallbacks for
  `CapillarySampleContainerCorrection`. **Reviewer classification:**
  - **False positives (4):** `input_state`, `transmission_source`,
    `transmission_units_source` and `transmission_uncertainties_sources`.
    These are read only in the parent's `_measured_transmission()` and
    `calculate()`. The container overrides `calculate()`, never calls
    `_measured_transmission()`, and deliberately pops these keys from its
    arguments.
  - **Already covered by direction 2 (5):** the four `*_uncertainties_sources`
    reads already known, plus `empty_centre_mu_uncertainties_sources`
    (the container calls `_scalar("empty_centre_mu", ...)`). All of them
    disappear when `_scalar()` stops reading uncertainty sources.
  - **Already covered by direction 3 (4):** the absorption and transmission
    `_units` and `_units_source` keys. Declaring them in the parent's arguments
    is enough, because the container copies the parent's arguments at import.

  **Direction 6, revised: scan by reachability, not by MRO membership.**
  Start from the step's pipeline entry points (`calculate`,
  `dependency_contract`, `prepare_execution`, each resolved with
  `getattr(step, name)` so overrides win). Follow `self.<method>(...)` and
  `cls.<method>(...)` calls by resolving them the same way, plus the
  shared-helper calls the guard already follows. Do not scan methods that
  are not reachable. Keep the inherited-method self-test, and add one in
  which an overridden parent method reads an undeclared key that must
  **not** be reported. Then remove the remaining fallbacks, including the
  14 inherited ones, as in direction 1. No maintainer decision is needed;
  continue with R03b.

  **Design note for later (not R03):** the container subclasses the
  self-absorption step and pops arguments that some inherited methods still
  read. It works only because those methods are unreachable. A shared
  capillary base class (geometry, coefficients, μ resolution, beam profile)
  with two thin steps would remove that fragility. Consider it alongside R07
  or R11, or as its own item.


---

- 2026-10-30, `c6fe394` (R03b): **done.**
  - Verified independently: guard 13 passed with an empty
    `ALLOWED_VIOLATIONS`; full suite 1084 passed, 1 skipped; flake8 clean;
    reference `--check` clean; Sphinx `-W` clean. The reviewer's original
    AST drift script finds 0 fallbacks; its only remaining hits are the
    nested `ReduceDimensionality` estimator mapping, which the guard
    correctly excludes.
  - The reachability scan matches the checkpoint classification: the 4
    unreachable `_measured_transmission` reads are excluded, and the 9
    reachable undeclared reads are resolved by directions 2 and 3.
  - Versions: only the two capillary modules were bumped (metadata changed);
    pure fallback removals were not. As directed.
  - `AppendSource` tests now use the real constructor and
    `modify_config_by_dict` instead of `__new__` plus a replacement dict.
    That is correct: removing fallbacks means code relies on seeded
    defaults. The reviewer checked that no production code (runner, server,
    client, MoDaCor-examples) replaces `step.configuration` wholesale; only
    `ProcessStep.__attrs_post_init__` assigns it.
  - Consequence worth knowing: steps now rely on `ProcessStep` seeding every
    declared default. Code that builds steps while bypassing the
    constructor, as the old `AppendSource` tests did, will get `KeyError`.
    That is acceptable and arguably desirable.

## R04 — Derive capillary sample μ through `BaseData`

**Status:** done

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

**Extension (maintainer decision, 2026-10-30): uncertainty for directly
given μ.** Today a directly configured `sample_mu` is returned with no
uncertainties (`_ResolvedSampleMu(..., {})`). The `_uncertainties_sources`
read for it was dead code and was removed in R03b.

- Declare `sample_mu_uncertainties_sources` (type `dict`, default `{}`,
  "Mapping of uncertainty name to IoSources reference for `sample_mu`"), in
  the same style as `sample_phase_thickness_uncertainties_sources`.
- In the new `BaseData` route, load directly given μ with its uncertainty
  sources, so the direct and derived routes both produce a `_ResolvedSampleMu`
  with named uncertainties. The existing finite-difference propagation then
  carries them through the attenuation factors unchanged.
- This applies to `CapillarySampleContainerCorrection` automatically,
  because it inherits the method and copies the parent's arguments. Verify
  this with a test, not by assumption.
- `dependency_contract()` must include the uncertainty-source refs.
  Check whether `source_refs_from_references(cfg)` already covers dict values,
  and assert it in a test.
- Remove "sample μ" from the limitation sentence in both `step_note`s (added
  in R03b), and update the "Uncertainties and diagnostics" section of
  `capillary-self-absorption.md`: direct sample μ is now supported; wall μ
  and geometry remain out of scope.

**Acceptance criteria.**

- The manual derivative code and `np.hypot` are gone from `_sample_mu`.
- New tests compare μ and its uncertainties, for both the absorption and the
  transmission routes, against analytic values with independent thickness
  and factor uncertainties under distinct names and under a shared name. Use
  tight tolerances. If `BaseData` correlation semantics differ from the
  current behavior for a shared uncertainty name, stop and record the
  difference instead of choosing one.
- All existing capillary tests pass unchanged.
- Extension tests, for both capillary steps: direct `sample_mu` with an
  uncertainty source yields named uncertainty components on the corrected
  signal. For an equivalent μ ± σ, the factor uncertainty matches the
  derived route within tight tolerance. Without uncertainty sources, results
  are identical to before.

**Implementation notes.**

- 2026-10-30: wrote and ran the analytic comparison tests before changing
  runtime code: 4 passed. Both absorption and transmission routes match the
  analytic μ and uncertainty components at `rel=1e-14`, with distinct names
  and a shared name. `BaseData` and the previous implementation both combine
  shared-name operand contributions in quadrature; no correlation decision
  was required.
- Added `_scalar_basedata` for validated, constant-valued input quantities and
  reused the scalar uncertainty validation at the numerical boundary. The
  derivation is now `-transmission.log() / thickness`, with `1 - absorption`
  for absorbed fractions. Reduction to `_ResolvedSampleMu` converts to `1/m`.
  Existing domain validation messages are preserved. The attenuation kernels
  and their finite-difference uncertainty propagation are unchanged.
- Declared `sample_mu_uncertainties_sources`; configured and source-loaded
  direct μ now carry named components through both capillary steps. Extension
  and dependency regression tests failed before implementation: 6 failed.
  Tests compare direct and equivalent derived μ uncertainties on the resolved
  coefficient, attenuation factor and corrected signal, and verify identical
  nominal results with no uncertainty sources.
- Confirmed `source_refs_from_references(cfg)` recursively visits dictionary
  values; both dependency-contract tests assert the new uncertainty source
  refs and unchanged processing reads/writes. No dependency implementation
  change was needed.
- Updated both `step_note`s and the capillary design note. Bumped both
  capillary module versions to `20261030.4`. Regenerated the reference pages
  and checked them: 0 mismatches. Sphinx build: 0 warnings.
- Validation: capillary tests 71 passed; configuration guard 13 passed; full
  suite 1094 passed, 1 skipped, 3 warnings, 2 subtests passed; `flake8 src tests`
  0 violations. No changes to other remediation items.

**Review log.**

- 2026-10-30, `063735f`: **done.**
  - Shared-name semantics: Codex's claim was checked against the source.
    `BaseData` binary operations are documented as first-order and
    uncorrelated, with named uncertainties treated as independent sources
    (`basedata.py`, `_binary_basedata_op` docstring). That equals the
    previous `np.hypot` combination, so no maintainer decision was needed.
    The design note now states this explicitly.
  - `_sample_mu` now computes `-transmission.log() / thickness` (with
    `1 - absorption`) in `BaseData`, reducing to `_ResolvedSampleMu` only at
    the kernel boundary. The hand-written derivatives and `np.hypot` are
    gone. Validation messages are preserved, and the attenuation kernels and
    their finite-difference propagation are untouched, as scoped.
  - Extension: `sample_mu_uncertainties_sources` is declared; direct μ now
    carries named uncertainties through both capillary steps, with tests
    comparing against the equivalent derived route. The dependency
    contracts include the new source refs (already covered by the recursive
    `source_refs_from_references`, now asserted).
  - Step notes and design note updated; versions bumped to `20261030.4`.
  - Verified: capillary and guard tests 84 passed; full suite 1094 passed,
    1 skipped; flake8, reference `--check` and Sphinx `-W` clean.
  - Scientific note for the maintainer, not a defect: under the uncorrelated
    rule, two inputs that share an uncertainty *name* (for example both
    thickness and transmission tagged `"systematic"`) are still combined as
    independent. Fully correlated propagation would need a change to the
    `BaseData` uncertainty model, outside R04.


---

## R05 — Core-container validation and untested contracts (rescoped)

**Status:** done

**Rescoped 2026-10-30 (reviewer correction).** The original R05 table was
wrong: the reviewer had read the coverage report's "Missing" column as the
percentage. For example, "divide_databundles 63 %" meant "line 63 uncovered".
Re-measured, the `*_databundles` steps, `negate`, `models/integration`,
`helpers/statistics` and `io/buffer/codec` are all at 87–96 %. The real
gaps found by re-measuring (overall coverage 83.2 %) are below.

**Scope.**

1. **Bug: `ProcessingData` validation can be bypassed.** `__setitem__`
   enforces "string key, `DataBundle` value", but `ProcessingData({"a": 1})`,
   `.update({"a": 1})` and `.setdefault("a", 1)` all accept a non-`DataBundle`
   (verified). `DataBundle` already rejects these routes for its own
   contract; follow its implementation pattern. Cover the constructor,
   `update`, `setdefault`, `__setitem__` and `__repr__`. The file is at
   52 %; the lines in `__setitem__`'s error branches and `__repr__` are
   untested.
2. **Dead code: `dataclasses/validators.py`** (38 %). None of its four
   functions (`is_list_of_ints`, `check_data`, `arrays_are_equal_shape`,
   `check_data_element_and_units`) is used anywhere in `src/` or `tests/`.
   The only reference is a commented-out import in `process_step.py`.
   Delete the module and that comment. Check the docs and API reference
   pages for references and remove them.
3. **`SolidAngleCorrection.calculate()` is never executed by any test.** It
   is the only exported step in that position (the reviewer measured every
   step's `calculate()` coverage). The only reference is a dependency-contract
   test. Add `tests/modules/technique_modules/scattering/test_solid_angle_correction.py`:
   signal ÷ Omega with units and uncertainty propagation through `BaseData`,
   several processing keys, the error when `Omega` is missing, and the
   existing contract assertion kept or moved there.
4. **`IoSource`/`IoSink` base contracts** (`io_source.py` 75 %,
   `io_sink.py` 73 %, `io_sinks.py` 70 %): add focused tests that the base
   classes raise the documented unsupported-capability `NotImplementedError`
   for each optional method, and that the `IoSinks` registry's error paths
   (unknown ref, duplicate registration, malformed `ref::path`) behave as
   `docs/development/io-source-sink-guide.md` describes. If the
   implementation and the guide disagree, report it; do not pick one.

**Out of scope.** `server/runtime_service.py` (67 % of 744 statements) and
the `client/` modules (48–76 %) move to **R11 step 0** as characterization
tests before the split. `cli.py`, `debug/`, the plotting steps and the
tiled source are lower priority and are not part of this item.

**Acceptance criteria.**

- `ProcessingData` rejects invalid entries on every insertion route, with
  regression tests that fail on the current code.
- `validators.py` is gone and nothing references it.
- `solid_angle_correction.py`, `processing_data.py`, `io_source.py` and
  `io_sink.py` reach at least 90 % line coverage, or the notes explain
  which lines are unreachable.
- Overall coverage does not drop. Full suite and flake8 clean.
- Command:
  `.venv-dev/bin/python -m pytest -q --cov=src/modacor --cov-report=term-missing <touched test files>`.
  Read the **Cover** column, not "Missing".

**Implementation notes.**

- Implemented scope points 1–4 only. `ProcessingData` now follows
  `DataBundle`'s constructor/update/setdefault validation pattern, reusing
  the existing string-key/DataBundle-value errors. In-place union (`|=`)
  also routes through validated update; inherited `fromkeys` is covered.
  Empty string keys remain valid under the existing ProcessingData contract.
  Only `processing_data.py` gained a behavior version, `20261030.1`.
- Regression tests were written and run before the implementation changed:
  **15 failed, 8 passed**; after the fix: **23 passed**. They cover mapping,
  iterable and keyword insertion, direct assignment, setdefault (including
  existing keys), union, fromkeys, argument-count errors, and repr.
- Deleted the unused validators module and commented import. Searched source,
  tests, documentation, and API references for its module and helper names;
  only this item's historical scope/acceptance text refers to the removed code.
- Added solid-angle tests for multiple selected bundles, untouched bundles,
  signal/Omega units and independent uncertainties on both operands, the
  returned touched bundles, and missing Omega. Kept the existing exact
  dependency assertion in `test_fixed_key_dependency_contracts.py`.
- Added source/sink base-method tests and registry tests for routing, duplicate
  registration, unknown references, malformed separators, and registration
  types. The guide documents `ref::path` routing but does **not** specify
  registry error types, empty subpaths, or repeated separators; these tests
  characterize current behavior without changing either implementation or guide.
- **Contract conflict reported for review, not resolved:** the guide's
  Practical guidance says unsupported interface subsets should raise
  `NotImplementedError`; all 11 optional sink methods instead raise
  `UnsupportedSinkCapability`, a `RuntimeError` subclass. The sink section
  also promises clear unsupported-capability errors without naming an exception
  type. Eleven strict expected-failure tests preserve the NotImplementedError
  expectation and accept only `UnsupportedSinkCapability` as the known failure.
  No implementation or guide change selects either contract. Review must resolve
  this discrepancy before those expectations can become passing contract tests.
- Validation (counts only): touched tests **46 passed, 11 xfailed**; full suite
  **1138 passed, 1 skipped, 11 xfailed, 3 warnings, 2 subtests passed**;
  `flake8 src tests`: **0 violations**. Unchanged HEAD was measured separately
  from an archive with its own coverage file: **1094 passed, 1 skipped,
  3 warnings, 2 subtests passed**.
- Coverage measured with `--cov=src/modacor --cov-report=term-missing`, reading
  **Cover** (the repository also enables branch measurement):
  `processing_data.py`, `solid_angle_correction.py`, `io_source.py`, and
  `io_sink.py` each **100.00%**, in both focused and full runs.
  `io_sinks.py`: **80.21%** in the full suite. Overall full-suite Cover rose
  from **83.19%** to **83.64%**. No unreachable-line exemption is needed for
  the four files with the 90% acceptance threshold.

- Option A follow-up (2026-10-30): changed `UnsupportedSinkCapability` to
  inherit `NotImplementedError`, removed the strict xfail marker from the 11
  optional-method cases, and added the explicit RuntimeError subclass guard.
  The sink guide now names the exception and its base class. `chunking.py`
  has no `__version__` field, so no version field was added. Before the fix,
  the unmarked sink tests reported **11 failed, 2 passed**. Validation:
  I/O **155 passed**; server **58 passed**; full suite **1150 passed,
  1 skipped, 3 warnings, 2 subtests passed, 0 xfailed**;
  `flake8 src tests`: **0 violations**.

**Review log.**

- 2026-10-30, `f612c1a`: **points 1–3 accepted; point 4 waiting on a
  maintainer decision.**
  - Verified: full suite 1138 passed, 1 skipped, 11 xfailed; flake8,
    reference `--check` and Sphinx `-W` clean. Coverage claims are
    consistent with the Cover column.
  - Point 1: `ProcessingData` now mirrors `DataBundle`'s
    `__init__`/`update`/`setdefault` pattern line for line, plus `|=`.
    Accepted. Shared nit, not a defect: in both classes `update()`
    validates and inserts one item at a time, so a failing item leaves the
    earlier ones inserted. Make both atomic later if it matters.
  - Point 2: `validators.py` and its commented import are gone; no
    references remain.
  - Point 3: `SolidAngleCorrection.calculate()` is now exercised (units,
    uncertainties on both operands, multiple bundles, missing `Omega`).
  - Point 4: Codex correctly reported, without resolving, that the guide
    says unsupported subsets raise `NotImplementedError`, while all 11
    optional sink methods raise `UnsupportedSinkCapability(RuntimeError)`.
    The 11 strict-xfail tests encode the guide's expectation.

  **Reviewer recommendation (option A):** change the base class to
  `class UnsupportedSinkCapability(NotImplementedError)`. Because
  `NotImplementedError` subclasses `RuntimeError`, this is fully backward
  compatible. Every existing catch (`except RuntimeError` in
  `server/api.py`, `runtime_service.py` ×6 and `process_step_registry.py`)
  and every `pytest.raises(UnsupportedSinkCapability)` still matches. The
  only `except NotImplementedError` in `src/` wraps a *source* call
  (`material_attenuation.py`), so its behaviour is unchanged. Sources and
  sinks then share one documented signal, the guide stays correct, and the
  11 xfails become ordinary passing tests (remove the `xfail` markers).
  Option B, changing the guide to name `UnsupportedSinkCapability`, would
  leave sources and sinks with different contracts.


---

- 2026-10-30, `7537297` (option A, maintainer-approved): **done.**
  `UnsupportedSinkCapability` now subclasses `NotImplementedError`. The 11
  xfail markers are removed and those tests pass as normal tests. A new
  test pins `issubclass(UnsupportedSinkCapability, RuntimeError)` so the
  existing `except RuntimeError` catches stay covered, and the guide names
  the type. Verified: io and server tests 213 passed; full suite 1150
  passed, 1 skipped, 0 xfailed; flake8 and Sphinx `-W` clean.
  `chunking.py` has no `__version__`.

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

**Step 0 (added 2026-10-30): characterization tests first.**
`runtime_service.py` is at 67 % line coverage (744 statements) and the
`client/` modules at 48–76 %. Before moving any code, raise
`runtime_service.py` to at least 85 % with tests through the public
`RuntimeService` methods and `server/api.py`, focusing on the uncovered
failure, recovery and fallback paths. These tests are the safety net for the
split and must pass unchanged after every extraction commit.

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

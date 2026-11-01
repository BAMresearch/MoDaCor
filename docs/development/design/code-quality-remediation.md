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
6. Before a branch goes to a pull request, the reviewer runs the CI test
   environment (`tox -e py312`), not only `.venv-dev`. Added 2026-10-31 after
   PR #148 failed in CI: `.venv-dev` has optional extras (xraylib) and a
   different numpy build, which hid an unguarded optional import and an exact
   float comparison.

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
- **Versioning policy (maintainer decision, 2026-10-31):** until v2.0,
  incompatible interface changes are released as minor versions (see the
  contribution checklist). Do not write `type!:` or `BREAKING CHANGE:` in
  commit messages; document the change in
  `docs/processing/interface-migrations.md` instead. The `BREAKING CHANGE:`
  footers requested for R01b predate this decision. They sit in branch
  commits that are squashed away, so they must not appear in the squash
  message.

## Overview

| ID | Item | Kind | Cost | Benefit | Status |
| --- | --- | --- | --- | --- | --- |
| R01 | `PolarizationCorrection` fails when no factor is set | bug | XS | high | done |
| R01b | One input per setting: remove configuration aliases and the mask-key fallback | simplification | S | high | done |
| R01c | "Did you mean …?" hint for unknown configuration keys | usability | XS | medium | done |
| R02 | `DilateMask` documents an alias that is rejected | bug | XS | medium | done |
| R03 | One source of truth for configuration defaults, with a guard test | consistency | S | high | done |
| R04 | Derive capillary sample μ through `BaseData` | contract | S | medium | done |
| R05 | Core-container validation and untested contracts (rescoped) | bug + tests | S | medium | done |
| R06 | Repository hygiene: untracked artefacts and `.gitignore` | housekeeping | XS | low | done |
| R07 | Declarative data-key dependencies, verified by an access-trace test | design | M–L | high | done |
| R08 | Audit remaining unit stripping in modules | contract | M | medium | in progress |
| R09 | Narrow broad `except Exception` handlers | robustness | M | medium | open |
| R10 | Conventions: `attrs` in step modules, headers, `step_reference` | consistency | S–M | low | open |
| R11 | Split `RuntimeService` into focused services | structure | L | medium | open |
| R13 | CI never exercises the xraylib (attenuation) path | tests / CI | XS | medium | in progress |
| R14 | The `required` argument flag is never enforced | contract | S | medium | open |
| R15 | Server: explain source/sink and run failures, and keep logs | diagnostics | S–M | high | in progress |
| R16 | Uncertainty propagation gaps found by the R08 audit | science | M | medium | decision needed |
| R17 | BaseData uncertainty correctness: Stage 0 of the uncertainty specification | bug | M | high | in progress (own branch and PR) |
| R18 | Central unit conversion for BaseData (includes the transmission-unit bug) | contract | M | high | approved |
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

**Status:** done

**Problem.** With aliases removed, a misspelled key (for example
`polarisation_factor`) fails at construction with `KeyError` listing every
known key. That is correct but noisy for steps with many arguments.

**Approach.** In `ProcessStep`'s unknown-key check, add the closest matches from
`difflib.get_close_matches(key, known_keys, n=3, cutoff=0.6)` to the message,
for example `Unknown key 'polarisation_factor'. Did you mean
'polarization_factor'?`. Keep the existing message and exception type, so that
`KeyError` and its match patterns in tests keep working.

**Location (reviewer, 2026-10-31).** The unknown-key `KeyError` is raised in
exactly two places, `ProcessStep.normalize_config_value` and
`ProcessStep.validate_config_value` (`process_step.py`, about lines 487 and
496), with identical messages. No other route (runner, pipeline schema,
server, client, CLI) validates step keys itself, so pipeline YAML and server
requests all reach these two lines. Move the message into one private helper
used by both, so the hint cannot drift between them. Keep the exact existing
prefix `Key <key> not found in configuration. Known keys: ...` and add the
hint as a separate sentence, so existing `match=` patterns keep working.

**Acceptance criteria.** A focused test in `tests/dataclasses/` shows the hint
for a near miss and no hint for an unrelated key; existing tests pass
unchanged.

**Implementation notes.**

- Added `_unknown_config_key_error` in `process_step.py`, shared by
  `normalize_config_value` and `validate_config_value`. It preserves the full
  existing prefix and appends a separate suggestion sentence only when
  `get_close_matches(key, known_keys, n=3, cutoff=0.6)` finds matches.
- Added `tests/dataclasses/test_process_step_unknown_keys.py` covering both
  methods and construction with `PolarizationCorrection`, the British spelling
  near miss, an unrelated key without a hint, and the unchanged prefix.
- Bumped `process_step.py` to `20261031.1`.
- Validation: focused tests 5 passed; full suite 1155 passed, 1 skipped,
  2 subtests passed, 3 warnings; flake8 0 violations; touched-file isort
  0 violations. Before implementation, regression tests: 3 failed, 2 passed.

**Review log.**

- 2026-10-31, `744cc73`: **done.** One helper,
  `_unknown_config_key_error`, is used by both raise sites; the original
  prefix is unchanged and the hint is appended. Tried through real step
  construction: `polarisation_factor` suggests `polarization_factor` first;
  `raduis` suggests `radius`; `with_processing_key` suggests
  `with_processing_keys`; an unrelated key gets no hint. Full suite 1155
  passed, 1 skipped; flake8 clean. Cosmetic, optional: three suggestions read
  as "'a' or 'b' or 'c'"; "'a', 'b' or 'c'" would read better, and for steps
  with many arguments the hint sits after a long known-keys list. Neither
  needs action.


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

**Status:** done. Merged in PR #150 and released in v1.13.0.

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

**Implementation notes.** Done by the reviewer on 2026-10-31 with the
maintainer's decisions:

- `.gitignore`: `*.code-workspace`, `*.bkp`, `.$*`, `uv.lock`.
- Removed from git: `docs/Classes.drawio` and `docs/FlowGeneral.drawio`
  (unreferenced).
- Tracked: `docs/development/design/future-package-architecture.drawio`, a WIP
  schema for splitting MoDaCor into sub-packages with tighter
  responsibilities (formerly the untracked `docs/design/`).
- Moved out of the repository to
  `~/Documents/BAM/Projects/2026/MoDaCor-auxiliary/`: `logo/` (Keynote
  draft), `testdata/` (25 MB MOUSE `.nxs` used by no test), and
  `docs-images/` (old diagram exports). Items approved for deletion
  (draw.io backups, `Classes.drawio.pdf`, `gendocs_test.sh`, `build/`,
  `dist/`, and source/test folders holding only `__pycache__`) were moved to
  `MoDaCor-auxiliary/to-delete/` for the maintainer to empty.
- Verified: `git status` shows no untracked files; `tox -e manifest` OK;
  `tox -e py312` 1152 passed, 4 skipped; Sphinx `-W` clean.

**Review log.** _(reviewer)_

---

## R07 — Declarative data-key dependencies, verified against behaviour

**Status:** done (A′). Merged in PR #150 and released in v1.13.0. The recommendation
was revised on 2026-10-31 after checking the modules that derive data keys
from configuration.

**Problem.** 35 of the 41 exported steps override `dependency_contract()`; only
6 use `dependency_role`. Each override restates by hand which data keys
`calculate()` touches, and nothing checks that the two agree.

**What the contract is used for.** Only `server/execution.find_dirty_step_ids`
reads it. A step is a seed for a partial rerun when a changed source matches
its `source_refs`, or a changed key matches `processing_reads |
processing_writes`. Every step downstream of a seed in the pipeline graph is
rerun regardless of its contract. Two consequences:

- Reads and writes are combined at runtime. The distinction between them is
  documentation for now.
- Declaring too much costs an unnecessary rerun. Declaring too little gives
  stale results. A contract that covers everything the step touches matters
  more than one that is exactly tight.

### Findings (2026-10-31)

1. **The current overrides are accurate.** A prototype recorded every
   `DataBundle` access made during each step execution across the full test
   suite (1155 tests). It then compared each access with that step's contract.
   No step touched a key outside its contract. The only flags were `update()`
   writes of declared keys and re-storing the same bundle object. This covers
   only the configurations the tests exercise. R07 is about maintainability and
   keeping the contracts correct, not about a live bug.
2. **No data key depends on data content.** Every key that is not written
   literally in the code is a fixed function of the configuration:
   - **Diagnostic names built from a prefix:** `FindCenterOfMass1D` and
     `FindScaleFactor1D` (`{diagnostic_prefix}_…`).
   - **Fixed diagnostic sets:** `IndexedAverager` (`bin_count`,
     `sum_weights`, …).
   - **Keys taken from a mapping:** `CopyDataBundleKeys.key_map` and
     `UnitsLabelUpdate.update_pairs`.
   - **Output written to another bundle:** `Integrate1D.output_processing_keys`,
     `IndexedAverager` and `ConcatenateDatabundles`.
   - **Conditional keys:** `AngleToQ` reads `center_key` only when `angle_zero`
     is unset; `FindScaleFactor1D` writes `background_output_key` only when
     `fit_background` is set.

   A contract can therefore always be computed from configuration before the
   step runs. However, a purely declarative scheme cannot cover all of these
   cases without becoming a small configuration language of its own.
3. **The 35 overrides fall into six groups.**

   | Group | Steps | Declarative? |
   | --- | --- | --- |
   | (a) only configurable keys | `DilateMask`, `ReduceMask`, `Negate` | yes, with existing roles |
   | (b) fixed `signal` / `Omega` plus configurable keys | `Multiply`, `Divide`, `Subtract`, `PoissonUncertainties`, `SolidAngleCorrection`, `ReduceDimensionality`, `AttenuatorPlateCorrection`, `DetectorEfficiencyCorrection`, `FlatPlateSelfAbsorptionCorrection`, `PolarizationCorrection`, `CapillarySelfAbsorptionCorrection` | yes, with a fixed-key declaration |
   | (b′) keys given as a mapping | `UnitsLabelUpdate` | yes, if roles accept mapping keys |
   | (c) regular keys plus derived extras | `AngleToQ`, `FindCenterOfMass1D`, `PixelCoordinates3D`, `XSGeometryFromPixelCoordinates` | partly: roles for the regular keys, a short extension override for the rest |
   | (d) several bundles with different roles | `DivideDatabundles`, `MultiplyDatabundles`, `SubtractDatabundles`, `FindScaleFactor1D`, `SubtractInterpolated1D`, `CopyDataBundleKeys`, `CapillarySampleContainerCorrection`, `ConcatenateDatabundles`, `IndexedAverager`, `Integrate1D` | no; would need a per-role bundle selector |
   | (e) I/O and path-addressed | `AppendProcessingData`, `AppendSource`, `AppendSink`, `SinkProcessingData`, `Plot1DVisualization`, `Plot2DVisualization` | no |

4. **The fixed keys are already listed twice.** The describer's
   `required_data_keys` and `modifies` already name `signal`, `Omega` and the
   defaults of configurable keys. Only `scripts/generate_module_doc.py` reads
   them. A new fixed-key field would be a third list unless the docs are
   checked against it.
5. **`output_processing_key` is accepted by every step but honoured by two.**
   It is a base-schema key in `ProcessStep.CONFIG_KEYS`. Only
   `ConcatenateDatabundles` and `IndexedAverager` use it; `Integrate1D` has its
   own `output_processing_keys`. The other 39 steps accept it and silently
   ignore it. The base default contract also has a branch for
   `processing_key` / `databundle_output_key`, which belong to
   `AppendProcessingData`; no exported step reaches that branch. The
   MoDaCor-examples pipelines and the local experiment pipelines pass
   `output_processing_key` only to the two steps that honour it.

### Options

- **A′ (recommended).** Declare the regular part, extend explicitly for the
  rest, and verify every contract against behaviour.
  1. Add one describer field, `fixed_data_keys: dict[str, str]`, for example
     `{"signal": "read_write", "Omega": "read"}`. Values are `read`, `write` or
     `read_write` and are validated. The default contract expands these keys
     over `with_processing_keys`, together with the existing
     `dependency_role` keys and source-ref extraction.
  2. A `dependency_role` argument whose value is a mapping contributes the
     mapping's keys (needed for `UnitsLabelUpdate`).
  3. Group (c) steps declare their regular keys and override only to add the
     extras: `super().dependency_contract()` plus additions. `AngleToQ` and
     `PixelCoordinates3D` already follow this pattern.
  4. Groups (d) and (e) keep explicit overrides. They are genuinely irregular,
     and plain Python is clearer than a selector syntax.
  5. A permanent **access-trace test**, a robust version of the prototype
     above. While the test suite runs, every step execution records the data
     keys it touches and fails if they are not covered by its contract. This
     guards hand-written and declarative contracts alike, including the steps
     with derived keys, and addresses the drift risk that motivated R07.
  6. A docs consistency test: `required_data_keys` and `modifies` must agree
     with the declared fixed keys and the role defaults. This keeps the
     existing documentation lists without a larger describer redesign.
  7. Move `output_processing_key` into the two steps that honour it, and
     remove the dead branch from the base contract.

  **Result:** 21 of 41 steps need no override (6 today plus 15 migrated). Four
  keep a short extension override and 16 keep an explicit one. The trace test
  checks all of them. The earlier target of at most 12 overrides was too
  optimistic.
- **A (original).** Fixed-key lists only. It migrates the same 15 steps, but
  leaves the 20 remaining overrides unchecked and adds a third, unchecked list
  of keys.
- **B.** Add a configurable `signal_key` to every step and use roles only.
  This gives the most uniform mechanism, but adds about 11 public arguments
  that nobody has asked for, and invites applying corrections to keys other
  than the intensity.
- **C.** Keep all overrides and add only the trace test. This is the cheapest
  option and removes the drift risk, but keeps the boilerplate.
- **Not recommended:** a bundle-selector syntax inside roles, such as
  `with_processing_keys[0]`, to absorb group (d). It would remove about ten
  overrides, but every module author would have to learn a small language.

### Approach (A′, once approved)

One branch and one pull request, with one commit per sub-step.

1. **R07a: safety nets.**
   - A temporary contract snapshot covering every exported step with one or
     two representative configurations. It proves that the migration
     preserves contracts exactly.
   - The access-trace test, as a session-level fixture or a plugin module
     under `tests/`. It counts `update()` keys as writes, treats a newly
     inserted bundle as writes of its keys, and ignores re-storing the same
     bundle. A violation fails the test in which it happens. It must pass on
     the current code with no allow-list.
2. **R07b: mechanism.** Add the `fixed_data_keys` field with validation,
   mapping-valued roles, and the default-contract expansion. Remove the dead
   `AppendProcessingData` branch. Add unit tests in `tests/dataclasses/`.
3. **R07c: migrate** groups (a), (b) and (b′) in small batches, and reduce
   group (c) to extension overrides. The snapshot must stay identical; any
   difference is listed and justified.
4. **R07d: `output_processing_key`.** Move it from the base schema into
   `ConcatenateDatabundles` and `IndexedAverager`. `ConcatenateDatabundles`
   must reject `None` (`allow_none=False`). Add an entry to
   `docs/processing/interface-migrations.md`; this is a minor release under the
   versioning policy.
5. **R07e: docs.**
   - Add the consistency test for `required_data_keys` and `modifies`.
   - Have the reference-page generator render the declared dependencies.
   - Update the dependency section of the module author guide and the
     correction-module template.
   - Regenerate the reference pages.
   - Remove the snapshot test, which is no longer needed once the trace test
     and the focused contract tests remain.

### Acceptance criteria

- The trace test passes on the whole suite with no allow-list.
- The snapshot is identical before and after migration, apart from listed and
  justified differences.
- Every remaining override is listed with a one-line reason (expected: 4
  extensions and 16 explicit overrides).
- Steps that do not honour `output_processing_key` reject it, and the
  migration note says so.
- The module author guide, the template and the reference pages are updated.

**Decision.** 2026-10-31: option A′ approved, including R07d (moving
`output_processing_key` into the two steps that honour it). The work goes on
branch `quality-remediation-r07`, which also carries the R06 commits.

**Implementation notes.**

- **R07a (2026-10-31):** added 82 generated snapshot cases covering all 41
  exported steps, plus a session-installed access trace and 32 self-tests.
  There are no `no_contract_trace` uses or allow-lists.
- The trace exposed two whole-bundle reads. With maintainer approval,
  `AppendProcessingData` now logs the stored key without enumerating its bundle,
  and `ConcatenateDatabundles` resolves axis identities only among configured
  `data_keys`, rejecting excluded and ambiguous axes during lookup. Three
  focused tests cover excluded axes and aliases. Neither contract changed;
  no further exported-step violations were found.
- Full development suite with tracing: 1191 passed, 1 skipped in 20.57 s
  (22.94 s process wall time). For the identical 1159-test suite excluding the
  trace self-tests, process wall time was 24.20 s without tracing and 19.59 s
  with tracing (pytest: 20.51 s / 17.55 s). These single-run timings are noisy;
  they do not establish a speedup. The original pre-change baseline was
  1155 passed, 1 skipped in 17.50 s.
- `tox -e py312` passed without optional extras: 1188 passed, 4 skipped in
  21.48 s (25.00 s tox wall time). `tox -e lint` passed.
- Two details surfaced during implementation: `ConcatenateDatabundles` accepts
  `output_processing_key=None` and currently declares `None.*`, which the
  snapshot preserves; conftest must defer MoDaCor imports until fixture setup
  to avoid installed-wheel/source-doctest collection conflicts in tox.

- **R07b (2026-10-31):** added validated `fixed_data_keys`, mapping-key roles,
  and fixed-key expansion in the default contract; removed its dead
  `AppendProcessingData` branch. No exported steps changed; the 82-case
  snapshot is identical. All 35 module overrides remain pending migration.
  With the access trace active: development suite 1202 passed, 1 skipped;
  `tox -e py312` 1199 passed, 4 skipped; `tox -e lint` passed. No surprises.

- **R07b refinement (2026-10-31):** the R07c snapshot guard stopped on
  `UnitsLabelUpdate[default_keys]`: its empty `update_pairs` mapping widened
  the empty processing contract to `*`. Per maintainer decision, the default
  now distinguishes empty declarations from keys lacking a bundle selection.
  Empty declarations stay exact and empty; non-empty declarations without
  processing keys remain conservative. The generic fallback applies only to
  steps without hints. The snapshot is identical before module migration.
  With tracing: 1219 passed, 1 skipped; `tox -e py312` 1216 passed, 4 skipped;
  `tox -e lint` passed.

- **R07c (2026-10-31):** migrated groups (a), (b) and (b′) to declarations
  and reduced the regular group (c) contracts to extensions. Exactly 20
  overrides remain: four extensions and 16 explicit contracts. The 16 explicit
  override files are unchanged, as are the six pre-existing role-only steps.
  The regular-step focused contract tests passed unchanged (479 focused tests
  passed overall). One geometry test expected the old wildcard and was
  tightened to assert the exact new read/write sets, retaining its source-ref
  assertion; no tests were weakened.
- Only four snapshot entries changed: `PixelCoordinates3D` and
  `XSGeometryFromPixelCoordinates`, each in `default_keys` and `configured`.
  For each selected bundle `b` (`sample` and `background`):

  | Step | Before reads / writes | After reads | After writes |
  | --- | --- | --- | --- |
  | `PixelCoordinates3D` | `b.*` / `b.*` | `b.signal` | `b.coord_x`, `b.coord_y`, `b.coord_z` |
  | `XSGeometryFromPixelCoordinates` | `b.*` / `b.*` | `b.signal`, `b.coord_x`, `b.coord_y`, `b.coord_z` | `b.Q0`, `b.Q1`, `b.Q2`, `b.Q`, `b.Psi`, `b.TwoTheta`, `b.CosAlpha`, `b.Omega` |

  Every source-ref set is unchanged. The trace covers these exact keys across
  the full suite. `XSGeometryFromPixelCoordinates.prepare_execution()` accesses
  `signal` in its rank fallback, beyond the documented required coordinate
  keys, so that read is explicitly included. The earlier empty-mapping stop
  was resolved by the separate R07b refinement; no additional snapshot stop
  conditions occurred.
- With tracing active and no opt-outs: development suite 1219 passed,
  1 skipped; `tox -e py312` 1216 passed, 4 skipped; `tox -e lint` passed.
  Regenerated the reference pages; their check and the strict Sphinx build
  passed. The existing generator renders the new argument roles, including
  metadata inherited by `CapillarySampleContainerCorrection`; its explicit
  contract remains unchanged. Fixed-key rendering stays with R07e.
- Remaining overrides (one-line reasons):

  - `AngleToQ`: Adds the conditional center_key read when angle_zero is unset.
  - `FindCenterOfMass1D`: Adds diagnostic-prefix writes using _output_keys().
  - `PixelCoordinates3D`: Adds the NeXus detector_frame source reference.
  - `XSGeometryFromPixelCoordinates`: Adds NeXus detector_frame and sample_z_override source references.
  - `DivideDatabundles`: Separates dividend/divisor bundle reads and the dividend write.
  - `MultiplyDatabundles`: Separates multiplicand/multiplier bundle reads and the multiplicand write.
  - `SubtractDatabundles`: Separates minuend/subtrahend bundle reads and the minuend write.
  - `FindScaleFactor1D`: Separates work/reference bundles and conditional fit/diagnostic outputs.
  - `SubtractInterpolated1D`: Separates work/background bundles and remapped/subtracted outputs.
  - `CopyDataBundleKeys`: Maps source keys to target keys across distinct bundles.
  - `CapillarySampleContainerCorrection`: Separates filled/empty bundle reads and filled-bundle outputs.
  - `ConcatenateDatabundles`: Reads multiple input bundles and writes a separate output bundle.
  - `IndexedAverager`: Supports in-place and distinct-output bundles plus diagnostics.
  - `Integrate1D`: Pairs input bundles with configured output bundles.
  - `AppendProcessingData`: Loads source references into a configured bundle and BaseData key.
  - `AppendSource`: Declares source registration side effects.
  - `AppendSink`: Declares sink registration side effects.
  - `SinkProcessingData`: Parses configured processing paths for sink output.
  - `Plot1DVisualization`: Parses independent plot data/error paths across bundles.
  - `Plot2DVisualization`: Parses the configured two-dimensional plot data path.

- **Geometry follow-up (2026-10-31):** replaced the eagerly evaluated rank
  fallback in `XSGeometryFromPixelCoordinates` with `int(coord_x.rank_of_data)`.
  Removed its incidental `signal` read from the declaration and focused test.
  Exactly its two snapshot entries lost `sample.signal` and
  `background.signal` reads; every other snapshot field is unchanged.
  Full suite with tracing and no opt-outs: 1219 passed, 1 skipped.
- **R07d (2026-10-31):** removed `output_processing_key` from the shared schema
  and default contract. Its existing step-specific declarations now control
  validation: `ConcatenateDatabundles` rejects `None` and defaults to
  `concatenated`; `IndexedAverager` retains its string/None behavior. An ordinary
  step rejects the setting with the existing unknown-key hint. Removed the
  ignored setting from a deprecated geometry test and documented the YAML
  migration. The only snapshot change is the default concatenation write:
  `None.*` -> `concatenated.*`.
  Read-only search of all examples YAMLs found 11 uses in two DLS/I22 pipelines:
  six `ConcatenateDatabundles` and five `IndexedAverager`; no unsupported uses
  and no edits to the examples repository.
  Full suite with tracing and no opt-outs: 1221 passed, 1 skipped;
  `tox -e lint` passed. There are still 20 overrides. No stop conditions.

- **R07e (2026-10-31):** added default metadata equality checks for all
  25 declaration-based steps (including the four extensions); full overrides
  retain their existing hand-written expectations. The only exception is:

  | Step | Extra required keys | Extra modifies keys | Reason |
  | --- | --- | --- | --- |
  | `FindCenterOfMass1D` | none | `centroid_count`, `centroid_converged`, `centroid_window_min`, `centroid_window_max` | Its extension writes diagnostics derived from `diagnostic_prefix`. |

  Fixed three stale metadata descriptions, without changing their contracts:
  `IndexByCoordinate.required_data_keys` now includes default coordinate `Q`;
  `FindCenterOfMass1D.required_data_keys` now includes default axis `yaw`;
  `Negate` now requires and modifies default key `signal`, replacing its
  `configured data key` placeholder. No contract-versus-behavior discrepancy
  or snapshot stop condition was found.
- The reference generator now renders fixed-key access, role-argument access
  and defaults, custom-contract notices and extension notices. Regenerated all
  41 pages. Rewrote the author guide around roles, fixed keys, extensions and
  full overrides; documented empty-declaration and wildcard semantics and the
  access trace. Updated the correction template, AGENTS and the contributor
  checklist. Fixed the implicit concatenation in the fixed-key error message.
- Confirmed the snapshot was unchanged by R07e before removing the temporary
  snapshot test and JSON; the empty data directory was removed too. The access
  trace and focused tests remain, with no opt-outs. The final override count is
  still 20: four extensions and 16 explicit contracts, listed above.
- Final development suite with tracing: 1241 passed, 1 skipped (plus 2 subtests);
  `tox -e py312`: 1238 passed, 4 skipped (plus 2 subtests).
  `tox -e lint`, `tox -e docs` and `tox -e manifest` passed. The first manifest
  attempt could not resolve PyPI in the sandbox while fetching isolated build
  dependencies; retrying with network access passed. The examples audit is
  recorded in R07d above; that repository was not edited.

**Review log.**

- **R07a (2026-10-31): approved.** Commits `f2d9b37` (step fixes) and
  `77f5151` (snapshot and trace).
  - Re-run independently: full suite with the trace active, 1191 passed and
    1 skipped; `tox -e py312`, 1188 passed and 4 skipped; `tox -e lint` clean.
  - Mutation check: I removed the `Omega` read from `SolidAngleCorrection`
    and the prefix-derived diagnostic writes from `FindScaleFactor1D`. The
    trace failed the execution tests of both steps, and the snapshot test
    failed too, so both checks catch real omissions, including derived keys.
  - The step fixes do not change the results of successful runs. An axis
    outside `data_keys` is now rejected during lookup instead of after
    concatenation.
  - Known limitations of the trace, accepted:
    - A step executed in a thread that does not copy the context variables
      is skipped, not falsely failed. The server tests run through anyio,
      which copies them, so they are traced.
    - A bundle inserted with `ProcessingData.setdefault` is not attributed.
      No step does this.
  - Follow-up for R07d, noted by Codex: `ConcatenateDatabundles` accepts
    `output_processing_key=None`, and `_output_key()` then writes to a bundle
    literally named `"None"`.
- **R07b and R07c (2026-10-31): approved.** Commits `5585613` (mechanism),
  `d316651` (empty declarations stay empty) and `c66f43c` (migration).
  - Re-run independently: full suite with the trace active, 1219 passed and
    1 skipped; `tox -e py312`, 1216 passed and 4 skipped; `tox -e lint` clean.
    20 overrides remain, as planned.
  - The default contract only places `*` when keys are declared but there
    are no bundles to expand them over. The generic fallback applies only to
    steps without declarations.
  - Geometry check: removing the declared `signal` read from both geometry
    steps fails 11 tests through the trace, so both really read it.
    - `PixelCoordinates3D` needs it: the detector shape comes from
      `signal`.
    - `XSGeometryFromPixelCoordinates` reads it only by accident.
      `getattr(coord_x, "rank_of_data", ref["signal"]...)` evaluates its
      fallback eagerly, although `rank_of_data` is always a field of
      `BaseData`. Because geometry steps run early in a pipeline, this false
      read makes every change to `sample.signal` rerun the geometry and
      everything after it. It is fixed in the R07d/e brief.
  - Nit: an implicit string concatenation is left over in the
    `fixed_data_keys` error message in `process_step_describer.py`.
- **R07d and R07e (2026-10-31): approved; R07 done.** Commits `02e1606`
  (geometry signal read), `ba2f04d` (`output_processing_key`) and `a130290`
  (docs, metadata check, snapshot removal).
  - Re-run independently: full suite with the trace active, 1241 passed and
    1 skipped; `tox -e py312`, 1238 passed and 4 skipped; `tox -e lint`,
    `tox -e docs` and `tox -e manifest` clean. The MoDaCor-examples tests
    (23) pass against this branch.
  - `XSGeometryFromPixelCoordinates` now reads only `coord_x`, `coord_y` and
    `coord_z`, so a change to `sample.signal` no longer reruns the geometry.
  - `output_processing_key` exists only on `ConcatenateDatabundles` (string,
    `None` rejected) and `IndexedAverager` (string or `None`). Every other
    step rejects it with the "Did you mean" hint. The migration note is in
    `interface-migrations.md`.
  - The metadata consistency test fixed three stale `required_data_keys` /
    `modifies` entries (`FindCenterOfMass1D`, `IndexByCoordinate`,
    `Negate`). It has one reasoned exception: the prefix-derived diagnostics
    of `FindCenterOfMass1D`.
  - Final state:
    - 21 of 41 exported steps have no override;
    - 4 have extension overrides;
    - 16 have explicit overrides, each listed with a reason in the
      Implementation notes.
    - Every override is checked by the access trace.

---

## R08 — Audit remaining unit stripping in modules

**Status:** in progress on branch `quality-remediation-r08-r13-r15`. The
classification and Commit A (the scalar helper) are done. Commit B stopped at
the equivalence guard: BaseData dropped a sole `propagate_to_all` component.
That is a BaseData bug, now R17 (Stage 0). The remaining Commit B rows and the
kernel moves resume once Stage 0 (R17) and R18 have landed.

**Inventory (2026-10-31, after R07).** A pattern search (`.magnitude`,
`.m_as(`, `np.asarray(… .signal)`, `float(… .signal)`, `.to(…).magnitude`)
finds about 48 occurrences in 15 files, excluding `deprecated/`. The largest
counts are in `find_center_of_mass_1d.py` (11), `subtract_interpolated_1d.py`
(8), `integrate_1d.py` (5) and `concatenate_databundles.py` (5). The line
numbers in the Problem section below predate R04 and R07.

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

**Implementation notes.** 2026-10-31, classification only, reviewed against
runtime code at `97c07a0` after the three R15 commits. **No module/helper implementation was
changed in this part. R08 remains in progress pending scientific review and a
separate implementation brief.**

The expanded search covered every non-deprecated Python file below
`src/modacor/modules`, including `helpers`. It searched `.magnitude`, `.m_as(`,
`.to(...).magnitude`, `.to_base_units().magnitude`, `ureg.Quantity(`,
`np.asarray(... .signal)`, `float(... .signal)`, and `np.sqrt(`, then inspected
all `.signal`, `.uncertainties`, `.variances`, `.value`, `m_from`, array
arithmetic and variance/reduction code in context. There are **61 matching
lines in 18 files** in the expanded pattern search (the previous 48/15 inventory
was narrower). There are no `.m_as(` or `.to_base_units().magnitude` calls in
this tree. `.value` in the capillary modules is a numerical result-container
field, not a Pint magnitude accessor.

The table records **147 sites in 30 files: 98 (a), 16 (b), 33 (c)**.
A site is one operation or a related block; comma/range locations explicitly
cover every individual search hit, including both lines of multi-line
expressions. Counts are table rows, not distinct defects, modules, or regex
matches. Paths are relative to `src/modacor/modules/`; line numbers refer to
this audited source snapshot. Shape-only accesses and pure constructors from
IoSource/configuration that never strip an existing BaseData are not defects;
representative boundary/metadata sites are included for completeness. Ordinary
BaseData arithmetic steps (multiply/divide/subtract/negate, AngleToQ,
SolidAngleCorrection and PixelCoordinates3D physics) retain their semantics.

- **(a)** legitimate nominal/configuration, existing model/geometry boundary,
  display, structural/index/mask or dimensionless-count work. A legitimate
  extraction does not certify a downstream adapter's uncertainty restoration;
  those omissions have separate (b) rows.
- **(b)** supported arithmetic/component handling reimplemented manually, or
  measured uncertainties omitted/stale at output. This includes omissions
  needing a model Jacobian rather than a currently available BaseData method.
- **(c)** numerical reduction/statistics/sensitivity kernels in the module
  layer. BaseData currently supports elementwise arithmetic and unary
  transforms, indexed slicing and unit conversion; it has no sum/mean/grouped
  reduction, interpolation, centroid or two-argument arctan2 operation. Do not
  invent an unsupported BaseData call or replace a correlated derivative with
  independent-operand arithmetic.

The capillary design explicitly excludes wall-mu and geometry/profile
uncertainty (`capillary-self-absorption.md`, uncertainty discussion); those
boundaries are (a) under the accepted scope. Sample-mu, signal and configured
transmission uncertainty are in scope and must survive. The centroid model
already supplies both signal and axis sensitivities; extracting its contraction
is (c), not a claim that the current centroid has no uncertainty propagation.

| file:line | short snippet | class | reason | proposed fix (b/c) | risk |
| --- | --- | --- | --- | --- | --- |
| `base_modules/apply_mask.py:148` | `mask = mask_bd.signal` | (a) | Integer bit-mask boundary. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/apply_mask.py:159-163` | `src_bd.signal; copyto(src, masked_value, where=mask != 0)` | (a) | Explicit masking operation, not physical arithmetic. Existing uncertainty arrays are retained; mask defines invalid observations. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/bitwise_or_masks.py:100-107` | `target_bd.signal -> uint32` | (a) | Canonical bit-field target storage. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/bitwise_or_masks.py:111-118` | `src_bd.signal; bitwise_or(tgt, src, out=tgt)` | (a) | Boolean/bit-field merge; no physical uncertainties. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/combine_uncertainties.py:109-117` | `combine_uncertainty_keys(..., quadrature_aggregator)` | (a) | Uses the existing shared BaseData uncertainty API; no locally reimplemented variance sum. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/combine_uncertainties_max.py:94-102` | `combine_uncertainty_keys(..., maximum_aggregator)` | (a) | Existing shared component-selection API; explicit uncertainty regrouping, not physical arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:204` | `np.asarray(reference.signal)` | (a) | Nominal alignment comparison, after explicit unit selection. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:207` | `current.signal = np.asarray(..., float)` | (a) | Float storage on a copy before BaseData.to_units; uncertainty conversion stays in BaseData. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:209` | `np.allclose(current.signal, reference_values)` | (a) | Boolean alignment validation; no calculated quantity. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:235` | `current.signal = np.asarray(..., float)` | (a) | Same dtype preparation before unit conversion for concatenation. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:251` | `signals.append(np.asarray(current.signal))` | (a) | Structural concatenation; units, weights, every uncertainty component and axes are explicitly rebuilt at 262–269. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:255` | `broadcast_to(current.uncertainties[name], ...)` | (a) | Structural uncertainty concatenation with explicit require_matching/fill_zero policy; no new propagation formula. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/concatenate_databundles.py:287` | `np.asarray(output[sort_by].signal)` | (a) | Sort index only; BaseData.indexed applies the same permutation to signal, weights and uncertainties. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/dilate_mask.py:189` | `np.asarray(source_mask.signal)` | (a) | Mask morphology input; output preserves mask metadata, not measured uncertainty. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:157` | `sqrt(sum((signal_sensitivity * sigma)**2))` | (c) | Sensitivity contraction is numerical propagation; centroid sensitivities already come from models.centroid. | Use a pure sensitivity contraction in models.uncertainty, retaining named signal components. | Extraction changes neither result nor uncertainty. A policy change needs scientific review of correlated samples, clipped intensities and the selected window. |
| `base_modules/find_center_of_mass_1d.py:160` | `sqrt(sum((axis_sensitivity * sigma)**2))` | (c) | Same numerical contraction for measured axis components. | Share the models.uncertainty contraction with the signal branch. | Extraction changes neither result nor uncertainty. Axis calibration covariance and moving-window uncertainty require a scientific decision. |
| `base_modules/find_center_of_mass_1d.py:182` | `isfinite(signal.signal) & isfinite(axis.signal)` | (a) | Boolean validity selection. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:188` | `np.asarray(bundle[mask_key].signal) == 0` | (a) | Boolean mask. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:196` | `(float(half_width_cfg) * width_units).to(axis.units).magnitude` | (a) | Configuration half-width to axis units. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:199` | `(float(convergence_tolerance) * width_units).to(axis.units).magnitude` | (a) | Configuration convergence threshold to axis units. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:203` | `(float(baseline) * baseline_units).to(signal.units).magnitude` | (a) | Configured baseline to signal units; its configuration has no uncertainty input. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:206` | `argmax(np.asarray(signal.signal)[valid])` | (a) | Nominal anchor index only, not the centroid calculation. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:207` | `float(np.asarray(axis.signal)[anchor])` | (a) | Nominal initial window anchor; final center comes from the centroid kernel. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:213` | `abs(np.asarray(axis.signal) - center) <= half_width` | (a) | Boolean moving-window selection. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:214` | `argmin(abs(np.asarray(axis.signal) - center))` | (a) | Nominal nearest-index selection. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:217-218` | `intensity_centroid_1d(axis.signal, signal.signal, ...)` | (a) | Existing models.centroid input boundary; output restores axis units and both sensitivity-derived uncertainty families. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:239` | `BaseData(min(np.asarray(axis.signal)[contributors]), axis.units)` | (a) | Nominal window-bound diagnostic; not a measured center estimate. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_center_of_mass_1d.py:240` | `BaseData(max(np.asarray(axis.signal)[contributors]), axis.units)` | (a) | Nominal window-bound diagnostic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:44-46` | `sig2 = arr * arr ...; sqrt(sig2)` | (b) | Combines BaseData uncertainty components manually; a shared quadrature aggregator already exists. | Reuse dataclasses.uncertainty_tools.quadrature_aggregator, preserving the no-component fallback of 1.0. | Nominal fit and uncertainty should remain unchanged; retain current independent-component and fallback policies; no new scientific choice. |
| `base_modules/find_scale_factor1d.py:53` | `y = bd.signal.squeeze()` | (a) | Input boundary to models.scaling.DependentData1D; sigma and weights are supplied alongside y. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:62` | `np.asarray(bd.uncertainties[uncertainty_key])` | (a) | Named fit-weight component extraction at that kernel boundary. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:274-275` | `x_work_bd.signal.squeeze(); x_ref_bd.signal.squeeze()` | (a) | Unit-normalized coordinate input to models.scaling; fit axis-error omission is separately flagged below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:294` | `ureg.Quantity(fit_min, fit_units).to(x_ref.units).magnitude` | (a) | Static fit-bound conversion. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:299` | `ureg.Quantity(fit_max, fit_units).to(x_ref.units).magnitude` | (a) | Static fit-bound conversion. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/find_scale_factor1d.py:303-317` | `prepare_scale_fit_data(x_work=..., x_ref=..., dep_work=..., dep_ref=...)` | (b) | Fit receives dependent sigmas but no axis uncertainties, even if present on the BaseData axes. | Decide and document the axis-error contract; add a model-supported errors-in-variables treatment if those uncertainties must propagate. | Nominal fit and uncertainty may both change. Scientific choice: fixed abscissae versus errors-in-variables/correlated coordinate calibration; BaseData elementwise arithmetic alone is insufficient. |
| `base_modules/find_scale_factor1d.py:335-358` | `BaseData(fit_result.scale/background, uncertainties=...sigma)` | (a) | Model result boundary restores scale/background units and fitted sigma. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/index_by_coordinate.py:113` | `conversion = coordinate.units.m_from(configured_units)` | (a) | Pint conversion factor for static bin configuration; includes scalar bounds at 145–146 and edge arrays at 135. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/index_by_coordinate.py:145-146` | `float(bin_min/max) * conversion` | (a) | Repeated scalar-configuration conversion via the Pint factor; candidate for generalized scalar helper. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/index_by_coordinate.py:148` | `generate_bin_edges(coordinate.signal, ...)` | (a) | Existing models.binning boundary for nominal indexing. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/index_by_coordinate.py:174` | `assign_bin_indices(coordinate.signal, edges)` | (a) | Integer assignment, not propagation of a measured coordinate. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/indexed_averager.py:74-79` | `bincount(weights**2 * selected**2); sqrt(variance_sum)/sum_weights` | (c) | Grouped reduction and propagation are not BaseData elementwise operations. | Move pure grouped moment/uncertainty kernels to models.reduction; adapter keeps component naming, masks, units and axes. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:90-106` | `deviations = values - means[indices]; weighted scatter` | (c) | Numerical scatter accumulation wraps a second numerical helper under modules. | Move accumulation with finalize_weighted_scatter into models.statistics/reduction. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:281` | `np.asarray(value.signal).ravel()` | (c) | Entry to the embedded grouped mean kernel. | Pass numerical values to an extracted models.reduction kernel, restoring BaseData at the boundary. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:282` | `_broadcast(bundle[index_key].signal).astype(float)` | (a) | Validate integral bin indices, not scientific arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/indexed_averager.py:294` | `_broadcast(axis.signal).astype(float)` | (c) | Measured-axis co-reduction uses the same embedded grouped mean/propagation implementation. | Extract the shared value/axis grouped reducer; preserve its axis uncertainty attachment. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:298` | `_broadcast(bundle[mask_key].signal).astype(bool)` | (a) | Boolean exclusion mask. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/indexed_averager.py:337` | `weights[usable_sigma] *= 1 / sigma[usable_sigma]**2` | (c) | Inverse-variance weighting is part of a statistical kernel, not an uncertain quantity to multiply blindly. | Move weight construction into the grouped reducer; retain selected uncertainty-key and validity policies. | Extraction changes neither result nor uncertainty. Reinterpreting random/data-dependent weights or units would require scientific review. |
| `base_modules/indexed_averager.py:362-363` | `bincount(selected_weights); bincount(selected_weights**2)` | (c) | Numerical weighted-reduction moments. | Accumulate moments in the same models.reduction kernel. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:393-411` | `bincount(weights * values/axis) / sum_weights` | (c) | Grouped weighted means embedded in the module. | Use extracted grouped-reduction result for both value and measured axis; restore units and propagated components. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/indexed_averager.py:475-480` | `bin_count; positive_weight_count; sum_weights**2/sum_squared_weights` | (a) | Dimensionless diagnostics/counts; not propagation of measured quantities. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:136` | `np.asarray(reference_axis.signal)` | (a) | Nominal axis/domain validation; dropped axis uncertainty is separately identified at 237–241. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:142` | `np.asarray(signal.signal)` | (a) | Shape/finite checks; signal uncertainties remain available and are checked at 156. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:147` | `np.allclose(np.asarray(other_axis.signal), axis)` | (a) | Boolean equality after BaseData unit conversion. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:155` | `np.asarray(bundle[mask_key].signal) == 0` | (a) | Boolean exclusion mask. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:172-206` | `diff / duplicate groups / bincount(values)/counts` | (c) | Duplicate-coordinate reduction is numerical grouping embedded beside the integration adapter. | Move grouping/collapse into models.integration alongside quadrature_weights_1d; keep mask/config adaptation here. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/integrate_1d.py:217` | `broadcast_to(component, signal.signal.shape)` | (a) | Shape/slice adaptation for a named uncertainty; numerical propagation is below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/integrate_1d.py:221` | `sqrt(bincount(..., weights=values**2))/counts` | (c) | Duplicate means require an independent-observation reduction, absent from BaseData. | Move duplicate-group uncertainty reduction into the integration kernel. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/integrate_1d.py:222` | `sqrt(sum((values * quadrature_weights)**2))` | (c) | Quadrature uncertainty reduction has no BaseData reduction equivalent. | Move weighted signal/uncertainty integration to models.integration; return nominal and per-component results. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/integrate_1d.py:231` | `np.asarray(signal.signal)[common][sort_order]` | (c) | Entry to the locally implemented weighted integration kernel. | Use a models.integration result API; adapter supplies converted arrays and restores units/uncertainties. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/integrate_1d.py:237-241` | `BaseData(sum(values * quadrature_weights), units=signal.units * axis.units)` | (b) | Signal uncertainty is restored, but any measured axis uncertainty is silently absent from the result. | Add coordinate sensitivities in the integration model and attach their named contributions at the adapter boundary. | Nominal integral unchanged; uncertainty changes. Scientific choice: independent coordinates versus shared axis calibration, duplicate coordinates and quadrature sensitivity. |
| `base_modules/plot_1d_visualization.py:144` | `asarray(resolve_processing_path(...))` | (a) | Display serialization boundary, with separately inferred unit labels/error bars. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/plot_1d_visualization.py:172` | `ureg.Quantity(values, from_units).to(to_units).magnitude` | (a) | Display-only unit conversion for plot coordinates/error bars. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/plot_1d_visualization.py:351` | `asarray(basedata.uncertainties[name])` | (a) | Explicit plot error-bar array. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/plot_2d_visualization.py:48-51` | `value = value.signal; asarray(value)` | (a) | Display-only extraction; unit label is resolved separately. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/poisson_uncertainties.py:67-74` | `sqrt(max(signal, 1)); uncertainties[Poisson] = sigma` | (a) | Initial count-noise estimate, not propagation of an existing uncertain quantity; using BaseData.sqrt would compute a different object. | — | No change. Applying this after normalization/negative backgrounds needs a scientific counting-model decision; R08 should retain the current raw-count convention. |
| `base_modules/reduce_dimensionality.py:208` | `np.asarray(databundle[mask_key].signal)` | (a) | Integer bit-mask selection. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/reduce_dimensionality.py:234` | `x = np.asarray(bd.signal, dtype=float)` | (c) | Entry to local weighted/NaN reduction preparation, not merely validation. | Extract numerical selection/reduction inputs into models.reduction; adapter retains BaseData metadata. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:377-385` | `sum(w*x); mean; sum(w * deviations**2)` | (c) | Statistical moments implemented in the step. | Share pure reduction/scatter kernels with IndexedAverager. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:413` | `np.hypot(existing, estimate)` | (c) | Independent estimator-collision quadrature is part of the reduction policy. | Move numerical combination to the shared uncertainty kernel; retain declared collision policy. | Extraction changes neither result nor uncertainty. Independence between measured and scatter-derived components needs review only if policy changes. |
| `base_modules/reduce_dimensionality.py:484` | `x = bd.signal` | (c) | Raw input to embedded weighted reduction; the API has no BaseData sum/mean method. | Replace local numerical body with a models.reduction result; keep units, rank and axis adaptation. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:515-525` | `sum(w*x); signal_out = wx_sum/denom` | (c) | Weighted mean/sum implementation. | Move weighted moments and nominal reduction to models.reduction. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:537-540` | `sum(w_eff**2 * err_eff**2); sqrt(var_sum)/denom` | (c) | Weighted uncertainty reduction, not an elementwise BaseData operator. | Move per-component variance reduction into the same model. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:542` | `sigma = np.sqrt(var_sum)` | (c) | Weighted-sum uncertainty branch of that reduction. | Use the shared reduction result for sum as well as mean. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:609` | `x = bd.signal` | (c) | Raw entry to the scalar-weight optimized reduction. | Keep optimized scalar-weight path in models.reduction alongside general weights. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:613-616` | `np.mean(x); np.sum(x) * scalar_weight` | (c) | Numerical optimized mean/sum kernel. | Extract without changing scalar-weight, NaN and zero-weight behavior. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:624` | `sqrt(sum(err_arr * err_arr, axis=axis))` | (c) | Manual scalar-weight uncertainty reduction. | Use the extracted kernel for each named component. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:632-636` | `full_like(x, nan); sum(~mask)` | (a) | All-invalid sentinels and dimensionless contributor counts; policy stays with reduction adapter/kernel. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/reduce_dimensionality.py:640-642` | `sum(x_eff)/denom; sum(x_eff)*scalar_weight` | (c) | NaN-omit nominal reduction kernel. | Move to the same pure reduction API; preserve finite-count denominator. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_dimensionality.py:651` | `sqrt(sum(err_eff * err_eff, axis=axis))` | (c) | NaN-omit uncertainty reduction kernel. | Extract with mask/zero-weight behavior covered by existing tests. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `base_modules/reduce_mask.py:139` | `np.asarray(source_mask.signal)` | (a) | Boolean/integer reduction; numerical measurement propagation is inapplicable. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:143` | `isfinite(signal.signal) & isfinite(axis.signal)` | (a) | Validity selection only. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:151` | `np.asarray(mask.signal) == 0` | (a) | Boolean mask. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:174` | `background_signal.signal = np.asarray(..., float)` | (a) | Dtype preparation before BaseData.to_units preserves signal uncertainty conversion. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:175` | `background_axis.signal = np.asarray(..., float)` | (a) | Dtype preparation before axis conversion; coordinate propagation is separately flagged below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:186` | `np.asarray(background_axis.signal)[background_valid]` | (a) | Sort/uniqueness selection for the models.interpolation coefficient boundary. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:197` | `(float(max_gap) * gap_units).to(axis.units).magnitude` | (a) | Static configuration threshold, no measured uncertainty to propagate. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:201` | `np.asarray(sample_axis.signal)` | (a) | Input boundary to models.interpolation.remap_coefficients_1d; coordinate uncertainty is separately flagged below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:210-211` | `left_weight * background.signal[left] + right_weight * background.signal[right]` | (b) | Elementwise weighted addition duplicates supported BaseData multiplication/addition. | Slice with BaseData.indexed, multiply by dimensionless coefficients and add; handle identical source indices explicitly. | Neither nominal values nor uncertainties should change if existing coefficients are preserved. Nearest/exact hits have one-hot weights; preserve that behavior. A new cross-output covariance model needs scientific review. |
| `base_modules/subtract_interpolated_1d.py:216-218` | `sqrt((left_weight * sigma[left])**2 + (right_weight * sigma[right])**2)` | (b) | Manual propagation duplicates the supported arithmetic above. Current nearest coefficients are 1 and 0, so shared source indices do not currently understate sigma. | Use the same BaseData interpolation path, preserving current one-hot nearest/exact coefficients and named uncertainty components. | Neither nominal values nor uncertainties should change under the current independent-point policy. A scientific choice is needed only to model cross-output correlation from reused observations. |
| `base_modules/subtract_interpolated_1d.py:220-224` | `remapped_signal[~valid] = nan; uncertainty[~valid] = nan` | (a) | Masking invalid interpolation outputs consistently. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `base_modules/subtract_interpolated_1d.py:225-231` | `remapped = BaseData(..., uncertainties=remapped_uncertainties)` | (b) | Coordinate uncertainties from both sample/background axes are not included in remapping. | Extend models.interpolation with coordinate sensitivities, then restore those components before the existing sample_signal - remapped. | Nominal values unchanged; uncertainty changes. Scientific choice: interpolation slope, uncertain bracket membership and correlated axes. |
| `base_modules/threshold_mask.py:157` | `source = source_bd.signal` | (a) | Nominal threshold comparison creates a boolean/bit mask; configured bounds already use current signal units. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/detector_data.py:26-27` | `size(out.signal); shape(out.signal)` | (a) | Scalar shape validation only; returns BaseData with uncertainties intact. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/detector_data.py:88` | `size(bd.signal) == 1` | (a) | Scalar passthrough branch preserves uncertainties. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/detector_data.py:93` | `x = bd.signal.ravel()` | (c) | Input to embedded static-metadata weighted mean/SEM kernel. | Delegate numerical reduction to models.statistics; keep unit checking and BaseData construction here. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/scattering/detector_data.py:116` | `wsum = float(sum(w))` | (a) | Dimensionless weight validation/count, not physical quantity arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/detector_data.py:120` | `mean = float(sum(w*x)/wsum)` | (c) | Measured-scalar reduction embedded in an adapter. | Share pure weighted mean/scatter implementation with the other reducers. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/scattering/detector_data.py:123` | `n_eff = wsum**2/sum(w**2)` | (a) | Dimensionless effective count; no quantity uncertainty is stripped here. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/detector_data.py:126` | `var = sum(w*(x-mean)**2)/wsum` | (c) | Weighted population-scatter equation in an adapter. | Move with weighted-mean kernel; keep current population-variance definition. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/scattering/detector_data.py:127` | `sem = sqrt(var)/sqrt(n_eff)` | (c) | Numerical SEM definition; BaseData has no mean reduction. | Move with the statistical kernel. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/scattering/detector_data.py:129-134` | `BaseData(mean, units=bd.units, uncertainties={jitter: sem})` | (b) | Array input uncertainties are discarded and replaced by empirical jitter; scalar input does preserve them. | Propagate named measurement components through the mean and add jitter separately using a shared reduction result. | Nominal mean unchanged; uncertainty changes. Scientific choice: independence/correlation of metadata samples and whether jitter replaces or supplements measurement errors. |
| `helpers/scattering/material_attenuation.py:89` | `float(_decode_scalar(converted.signal))` | (a) | Explicit scalar boundary after BaseData unit conversion; callers supplying uncertain sample mu/transmission restore their components separately. | — | Current result unchanged. Use nominal-only extraction only at a documented kernel/config boundary; it is not a general uncertainty-preserving arithmetic helper. |
| `helpers/scattering/material_attenuation.py:145-146` | `linear_cm_inv = mass_attenuation * density; return linear_cm_inv * 100` | (b) | Hand-coded unit algebra and conversion outside a pure numerical kernel. | Attach explicit xraylib cm^2/g and density g/cm^3 units with Pint and convert the product to 1/m at the boundary. | Nominal coefficient should remain equal; uncertainty unchanged for current static lookup inputs. Any future energy/density uncertainty requires model sensitivities and scientific review. |
| `helpers/scattering/material_attenuation.py:194` | `np.asarray(cos_alpha.signal)` | (a) | Nominal/clipped model input boundary. Missing geometry uncertainty in consumers is flagged at their correction construction sites. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/scattering/photon_energy.py:25` | `np.asarray(converted.signal)` | (a) | Positive/finite validation only; h*c/E and h*c/lambda remain BaseData arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `helpers/statistics.py:49-51` | `n_eff; population_variance; ddof variance` | (c) | Reusable statistical mathematics lives under modules.helpers rather than models. | Move finalize_weighted_scatter and its numerical result container to models.statistics; module adapters may re-export temporarily. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/statistics.py:63` | `standard_deviation = np.sqrt(variance)` | (c) | Numerical scatter finalization, same misplaced kernel. | Move with finalize_weighted_scatter. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `helpers/statistics.py:66-67` | `sqrt(variance/n_eff); abs(sum_w)*SEM` | (c) | Mean/sum standard error equations in that kernel. | Move with finalize_weighted_scatter; preserve effective sample size and ddof definitions. | Extraction should preserve nominal values and uncertainties; retain the existing scientific policy. |
| `technique_modules/scattering/attenuator_plate_correction.py:226` | `positive_cos_alpha(cos_alpha_bd)` | (a) | Input boundary to the existing planar attenuation model; no new physics is implemented here. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/attenuator_plate_correction.py:229-241` | `planar_transmission(...); normalize by normal_transmission` | (a) | Existing model boundary with static material parameters; uncertainty omission is identified at factor restoration. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/attenuator_plate_correction.py:245-249` | `BaseData(divisor, dimensionless) with no uncertainties` | (b) | Drops supplied CosAlpha uncertainty before supported signal multiplication/division. | Restore model derivative contributions for incidence angle and preserve BaseData application; use Pint/BaseData for any uncertain scalar factors. | Nominal result unchanged; uncertainty changes. Scientific choice: incidence clipping and covariance of normal-incidence normalization; material lookup inputs are currently static. |
| `technique_modules/scattering/capillary_sample_container_correction.py:245` | `broadcast_to(np.asarray(mask_data.signal, bool))` | (a) | Boolean combined-mask adaptation. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:301-305` | `where(mask, original.signal, corrected.signal); restore uncertainties` | (a) | Explicit masked-value restoration; handles original and correction components consistently. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:321-325` | `filled.signal - wall_scale * empty.signal; full d(corrected)/d(mu)` | (c) | Numerical derivative for one shared sample-mu parameter, intentionally correlated across numerator/denominator; independent BaseData operators would double-count it. | Move the complete-expression derivative to models.attenuation (or models.uncertainty) as an array kernel; retain component attachment here. | Extraction changes neither result nor uncertainty. Scientific choice if correlation is changed; never substitute independent BaseData propagation for the shared-mu derivative. |
| `technique_modules/scattering/capillary_sample_container_correction.py:327-332` | `combine_uncertainty_component(corrected.uncertainties, ...)` | (a) | Attachment of a model derivative contribution; preserve names and existing quadrature policy. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:461-470` | `uncertainties_from_derivative(...); values/wall_empty_factor` | (a) | Model sensitivity-to-component adaptation with a nominal empty-wall factor (sample mu does not affect empty wall); surrounding corrected signal uses BaseData arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:490` | `wall_scale_calculation.signal` | (a) | Nominal array boundary for the shared-mu derivative; signal/empty-signal uncertainty is already propagated separately. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:499` | `factor_data(wall_scale_calculation.signal, wall_scale_uncertainties)` | (a) | Reattaches separately computed shared-mu components to the nominal wall-scale diagnostic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_sample_container_correction.py:563-571` | `BaseData(resolved_sample_mu.value, 1/m, uncertainties=...)` | (a) | Explicit model output units and named uncertainty restoration; .value is the numerical result container. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:41` | `ureg.Quantity(array, units).to(m).magnitude` | (a) | Configuration length-vector conversion for geometry/profile kernels, not a measured BaseData array. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:56` | `ureg.Quantity(rotation, units).to(radian).magnitude` | (a) | Configuration beam-profile rotation scalar. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:387` | `scalar_in_units(data, units)` | (a) | Static geometry/wall scalar boundary. Wall-mu and geometry/profile uncertainty are explicitly excluded by the accepted capillary design. | — | No change within the accepted model. Extending geometry/wall uncertainties needs scientific convergence/runtime choices; do not silently add them in R08. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:401-405` | `value, uncertainties = _scalar_components(...); BaseData(...)` | (a) | Scalar boundary explicitly restores both converted nominal and named uncertainty values. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:414-426` | `scalar_in_units(...); converted.uncertainties -> float` | (a) | Both nominal and every named component are extracted for the sample-mu numerical sensitivity boundary. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:450,454,459` | `isfinite(thickness/absorption/transmission.signal)` | (a) | Domain validation; deriving sample_mu uses BaseData.log/subtraction/division at 456–463. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:586` | `np.asarray(coordinate.signal)` | (a) | Meter-normalized geometry kernel boundary. Coordinate/profile uncertainty is explicitly out of scope in the capillary design. | — | No change under the accepted model; adding coordinate uncertainty needs a scientific finite-difference/convergence policy. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:601` | `np.asarray(mask_data.signal, dtype=bool)` | (a) | Boolean active-pixel selection and leading-frame mask reduction. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:631,677` | `scalar_in_units(measured_transmission, dimensionless)` | (a) | Nominal input for the existing kernel/correction; named transmission components are separately restored at 759–762. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:746` | `residual = attenuation / measured_value` | (b) | Raw ratio reimplements supported BaseData division with manually restored uncertainty below. | Construct attenuation BaseData with sample-mu terms and divide by converted measured_transmission BaseData; restore inactive values/components afterwards. | Nominal result unchanged; uncertainty may change with component-key fallback rules. Scientific choice only if covariance between attenuation and measured transmission is introduced. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:749-753` | `abs(attenuation_mu_derivative) * sample_mu_sigma` | (a) | Adapter restores sensitivity returned by the existing attenuation/finite-difference models; inactive component values are masked. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:754-762` | `sigma_A/T; A * sigma_T/T**2; combine_uncertainty_component` | (b) | Manual quotient uncertainty algebra duplicates BaseData division. | Use the same attenuation/transmission BaseData division as at 746, preserving named-key semantics and inactive masks. | Nominal values unchanged; uncertainty can change for matching/global component names. Preserve current independent-operand policy or obtain a scientific decision on covariance. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:766-769` | `abs(transmission_mu_derivative) * sample_mu_sigma` | (a) | Restores direct-beam kernel sensitivity into a dimensionless BaseData output. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/capillary_self_absorption_correction.py:790-798` | `BaseData(resolved_sample_mu.value, 1/m, uncertainties=...)` | (a) | Output boundary restores explicit mu units and all carried components; .value here belongs to _ResolvedSampleMu, not Pint/BaseData. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/detector_efficiency_correction.py:210` | `positive_cos_alpha(cos_alpha_bd)` | (a) | Input boundary to the existing absorption-efficiency kernel. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/detector_efficiency_correction.py:212-227` | `planar_absorption_efficiency(...); normalize` | (a) | Numerical model boundary using static material inputs; restoration omission is below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/detector_efficiency_correction.py:231-235` | `BaseData(divisor, dimensionless) with no uncertainties` | (b) | CosAlpha uncertainty does not reach the efficiency or corrected signal. | Attach angle/model sensitivities at the BaseData boundary, retaining BaseData division. | Nominal result unchanged; uncertainty changes. Scientific choice: clipping and shared normalization/calibration covariance. |
| `technique_modules/scattering/flat_plate_self_absorption_correction.py:103` | `scalar_in_units(transmission_data, dimensionless)` | (a) | Model scalar input boundary; transmission uncertainty is explicitly restored at 124–126. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/flat_plate_self_absorption_correction.py:118` | `positive_cos_alpha(cos_alpha_bd)` | (a) | Existing flat-plate kernel input boundary. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/flat_plate_self_absorption_correction.py:123-126` | `abs(derivative) * transmission_sigma` | (a) | Output sensitivity adaptation from an existing models kernel; retains transmission uncertainty and explicit dimensionless output units. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/flat_plate_self_absorption_correction.py:128-133` | `BaseData(attenuation, uncertainties=transmission_terms)` | (b) | Transmission uncertainty is restored but incidence-cosine uncertainty is not. | Extend the existing flat-plate model derivative API for CosAlpha, then attach those named components. | Nominal result unchanged; uncertainty changes. Scientific choice: clipping and transmission/incidence covariance. |
| `technique_modules/scattering/pixel_coordinates_3d.py:332,443` | `signal_bd.signal.shape; ndim(bd.signal)` | (a) | Detector shape/rank metadata only; coordinate physics uses BaseData arithmetic. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/polarization_correction.py:112` | `np.asarray(angle.signal)` | (a) | Radian-normalized input boundary to models.scattering.polarization; uncertainty omission is flagged at correction construction below. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/polarization_correction.py:128` | `(float(offset) * Unit(units)).to(radian).magnitude` | (a) | Configuration angular-offset conversion. | — | Neither nominal values nor uncertainties change; no scientific choice. |
| `technique_modules/scattering/polarization_correction.py:162-166` | `BaseData(factor, dimensionless) with no uncertainties` | (b) | Both two_theta and Psi uncertainties are dropped at the model output boundary. | Provide model sensitivities for both angles, attach named components, and retain the existing BaseData signal division (or express supported trig arithmetic in BaseData). | Nominal correction unchanged; uncertainty changes. Scientific choice: angle covariance/shared geometry, angular wrap and clipping thresholds. |
| `technique_modules/scattering/xs_geometry_from_pixel_coordinates.py:352-353` | `np.arctan2(dy.signal, dx.signal); BaseData(..., radian)` | (b) | Explicit nominal-only Psi drops coordinate uncertainties; BaseData has no two-argument arctan2 operation. | Add a geometry atan2/Jacobian kernel and restore named x/y uncertainty contributions at the adapter boundary; do not replace with arctan(dy/dx). | Uncertainty changes; nominal angle should not. Scientific choice: origin singularity, branch cut, shared x/y calibration covariance. |
| `technique_modules/scattering/xs_geometry_from_pixel_coordinates.py:361` | `cos_alpha_clipped.signal = clip(cos_alpha.signal, 0, None)` | (b) | Clips nominal cosine while leaving copied pre-clip uncertainties unchanged. | Define the clipping derivative/mask policy and update affected uncertainty components consistently; keep surrounding BaseData geometry arithmetic. | Nominal clipped cosine unchanged; uncertainty may change at clipped points. Scientific choice: hard clipping, invalid/back-facing mask and derivative at zero. |
| `technique_modules/scattering/xs_geometry_from_pixel_coordinates.py:450` | `bd.rank_of_data = min(RoD, ndim(bd.signal))` | (a) | Shape/rank metadata only; no numerical quantity is stripped. | — | Neither nominal values nor uncertainties change; no scientific choice. |

**Repeated configuration-scalar conversions and proposed helper (not implemented).**

The repeated scalar-plus-units-to-target-units sites are:

- `base_modules/find_scale_factor1d.py:294,299`: fit lower/upper bounds → reference axis units.
- `base_modules/find_center_of_mass_1d.py:196,199,203`: half width and convergence tolerance → axis units; baseline → signal units.
- `base_modules/subtract_interpolated_1d.py:197`: max gap → sample axis units.
- `technique_modules/scattering/polarization_correction.py:128`: angular offset → radians.
- `technique_modules/scattering/capillary_self_absorption_correction.py:56`: beam-profile rotation → radians.
- `base_modules/index_by_coordinate.py:113,145,146`: Pint factor followed by scalar bin limits → coordinate units (the edge array at 135 is not scalar).
- `helpers/scattering/material_attenuation.py:86–89` and its call sites
  `118,128,159,167,187`: scalar BaseData → keV, 1/m, g/cm³ or m. Capillary
  call sites `387,414,631,677` and flat-plate `103` also reuse this boundary.

Propose moving/generalizing **one** existing helper, rather than adding a
parallel scalar conversion:

```python
# src/modacor/modules/helpers/quantities.py
# re-export through modacor.modules.helpers

def scalar_in_units(
    value: BaseData | float | int,
    target_units: str | pint.Unit,
    *,
    name: str,
    units: str | pint.Unit | None = None,
) -> float:
    ...
```

For numeric configuration values, `units=None` means target units and Pint
performs the conversion. For BaseData inputs, require `units=None`, convert a
copy with `BaseData.to_units`, and retain the existing scalar-or-exactly-constant
array policy of material_attenuation's `_decode_scalar`; do not silently average
metadata. Return only a nominal float, explicitly for configuration or numerical
kernel boundaries. Callers with measured uncertainty must retain/convert the
BaseData and attach its components separately, or keep BaseData arithmetic.
Move the implementation out of `helpers/scattering/material_attenuation.py`
and update imports; a compatibility re-export may be used, with no duplicate
implementation. Preserve contextual errors using `name`.

`scalar_quantity_from_config_or_source` remains an IoSource/configuration
loader returning BaseData and enforcing mutual exclusion; it should not become
a nominal conversion helper. `require_scalar` remains a strict size-one
BaseData validator preserving uncertainties. `prepare_static_scalar` is an
actual statistical reduction (with separate (b)/(c) rows), not a conversion;
its scientific behavior must not be hidden in the proposed helper. Keep
`capillary_self_absorption_correction._length_array` and plotting's vector
conversions vector-aware, using Pint directly. `AngleToQ._angle_zero` already
converts and retains BaseData; it needs no nominal-float round trip.

Review priority: decide the missing/stale uncertainty policies in (b) before
changing results; extract (c) kernels with existing numerical/uncertainty behavior
preserved. The table deliberately records where scientific choices are needed
instead of treating a refactor as permission to choose a new estimator or
correlation model. All implementation fixes are deferred to the next brief.


**Refactor implementation progress (2026-10-31).**

- Shared scalar helper: moved and generalized to
  `modules/helpers/quantities.py:scalar_in_units`, re-exported by
  `modacor.modules.helpers`. Numeric configuration defaults to target units;
  BaseData inputs reject separate units and preserve scalar/exactly-constant
  extraction on a converted copy. Contextual errors include the parameter name.
  All listed scalar configuration sites now use it, including scalar bin bounds;
  vector bin-edge, profile-length and plot conversions retain Pint directly.
  Material text/scalar decoding is shared from quantities rather than duplicated.
  IoSource loading and statistical scalar reduction remain separate.
- Helper verification: **10 passed**; focused conversion/centroid/remapping/binning
  and scattering module checks: **171 passed** with access tracing active.
  Existing nominal and uncertainty assertions remain unchanged; direct converter
  tests compare against the pre-change Pint formulas at `rtol=1e-12` and check
  that BaseData input signals and uncertainties are not mutated.

- **STOP before Commit B: arithmetic equivalence failed.** The required
  interpolation expression `background.indexed(left) * lw +
  background.indexed(right) * rw`, with dimensionless uncertainty-free
  BaseData weights, was compared against the frozen old signal/sigma formula
  at `rtol=1e-12`. A normal `shared` component passes, including one-hot and
  identical left/right indices. A sole `propagate_to_all` component does not:
  for background values `[2, 4]`, sigma `[0.1, 0.2]`, left `[0, 0, 1]`, right
  `[0, 1, 1]`, and weights `[1, .5, 1]` / `[0, .5, 0]`, nominal values remain
  `[2, 3, 4]`. The old component is `[0.1, 0.1118033988749895, 0.2]`; the
  candidate uncertainty dictionary is empty. BaseData's binary arithmetic
  excludes the sole global component when the other operand has no uncertainty
  keys. Thus the proposed substitution changes published named uncertainties.
  The proof was run outside the source tree; no candidate arithmetic change
  was applied or committed. No access-trace opt-out was used.
- Per the brief's explicit stop rule, the six (b) rows and all 33 (c) kernel
  extractions remain unimplemented. R16 remains untouched. Resolving or
  adapting the global-component behavior needs review before resuming; this
  run does not alter BaseData's arithmetic contract to bypass the guard.

**Timings for completed helper refactor only.** The stop above prevented
Commit B and the kernel moves, so these are not timings for that unimplemented
work. An uncommitted benchmark used `.venv-dev`, Apple silicon, fresh input
containers outside the timed section, three warmups and the median of seven
`calculate()` calls. Scalar/general reductions and indexed averaging use
1000×1000 inputs; integration uses 2000 points with repeated coordinates; both
100×100 capillary steps use exact mode. Sample-only attenuation includes shared
sample-mu/transmission component names and an inactive pixel. The source before
R08 was frozen locally for the baseline.

| Step / size | Before (ms) | After helper (ms) | Change |
| --- | ---: | ---: | ---: |
| `reduction_scalar` | 2.454 | 2.269 | -7.54% |
| `reduction_weighted` | 3.660 | 3.699 | +1.06% |
| `indexed_1000x1000` | 73.837 | 77.767 | +5.32% |
| `integration_2000` | 0.128 | 0.126 | -1.79% |
| `capillary_self_100x100_exact` | 601.083 | 592.741 | -1.39% |
| `capillary_composite_100x100_exact` | 1712.910 | 1700.852 | -0.70% |

All measured changes are below the 10% slowdown limit. Across these cases,
**60 stored nominal/uncertainty arrays** were compared against the frozen
baseline with `rtol=1e-12`, `atol=0`, including NaNs, and matched. Source-code
changes after the helper commit were limited to these tracker notes.

**Validation of the safe committed changes:** full development suite with
access tracing **1286 passed, 1 skipped**, 2 subtests passed. Tox `py312`:
**1284 passed, 3 skipped**, 2 subtests passed; `lint` and `docs` passed,
including module-reference freshness and warning-as-error Sphinx. The skips
remain the RSS opt-in check and, in tox only, the two optional Tiled integration
tests. No public step metadata changed, so no reference-page regeneration was
needed. R15's follow-up server/client suite passed **109 tests**. The isolated
candidate-equivalence proof intentionally stopped with a failure; it is not a
failure in the committed implementation. The arithmetic/uncertainty stop is
the only triggered stop condition; the performance guard did not trigger.

**Resume after R17 (2026-10-31): second equivalence stop before Commit B.**
R17 commit `3ea4ec2` restores BaseData uncertainty components. The original
interpolation proof now passes for both `shared` and global-only
`propagate_to_all` backgrounds, including exact/nearest one-hot and identical
left/right selections.

The capillary residual has a further difference beyond the accepted folding
exception. A real `CapillarySelfAbsorptionCorrection` run with measured
transmission **50 percent**, uncertainty **1 percent**, explicit unit metadata,
separate named `mu` and `transmission` components, and an inactive pixel was
compared with attenuation BaseData divided by a copy of the measured
transmission converted to dimensionless units. Nominal residuals, masks,
component names and sample-mu uncertainties match at `rtol=1e-12`. The first
active transmission component is **1.1908702976776804** in the current
implementation and **0.011908702976776804** in the candidate: a factor of 100.
The current manual formula converts the nominal transmission through
`scalar_in_units` but uses the source uncertainty array without converting it
from percent. The candidate converts both nominal and sigma consistently.
This case has no global-only operand, so it cannot be explained by folding.

Per Part 2's explicit **any other difference means STOP** instruction, no
Commit B arithmetic changes, capillary reclassification/comment or kernel
moves were applied. R16 was not changed. The percent-unit finding is recorded
here for a separate scientific decision; this refactor does not correct it.
No R08 performance before/after comparison was run on this resumed attempt,
since no R08 implementation was applied; the earlier timing table remains
explicitly limited to the completed helper refactor.

**Review log.**

- **Classification (2026-10-31): approved, with scope decisions.** Commit
  `04c6172`. 147 sites: 98 (a), 16 (b), 33 (c).
  - **Maintainer decision: R08 covers refactors only.** That means the
    shared scalar helper, the six (b) rows that give identical results, and
    moving the 33 (c) kernels into `models`. The ten (b) rows that would add
    missing uncertainty terms change published uncertainties, so they move
    to R16 and are decided one by one.
  - The six identical-result rows:
    - `find_scale_factor1d.py:44-46`
    - `subtract_interpolated_1d.py:210-211, 216-218`
    - `material_attenuation.py:145-146`
    - `capillary_self_absorption_correction.py:746, 754-762`
  - **Performance check** (`.venv-dev`, Apple silicon; the maintainer asked
    for this because of the capillary ray tracing):
    - The capillary residual as BaseData division is 2.5 times slower than
      the hand-written numpy (4.2 → 10.6 ms per 1 Mpx; 10.6 → 26.4 ms for
      1679×1475 px), with identical values and uncertainties.
    - The full `CapillarySelfAbsorptionCorrection` takes about 55 µs per pixel
      in `exact` mode (0.6 s at 100×100 px, 5.0 s at 300×300 px), so the
      residual change adds about 0.01 %. The ray tracing stays in numpy and is
      classified (a).
    - `SubtractInterpolated1D` interpolation via `BaseData.indexed` and
      arithmetic gives identical results but adds a fixed ~0.25 ms per call
      (8–20 µs → 260–290 µs for 500–2000 points). That is under 0.1 % of a
      ~300 ms operando frame. **Maintainer accepted this cost.**
    - The (c) extractions move numpy code unchanged, so their speed should
      not change. The implementation brief asks for before/after timings.
  - BaseData combines same-named components in quadrature. For the
    interpolation this reproduces the current formula. For the capillary
    residual it must be checked against `combine_uncertainty_component` with
    shared component names before changing anything.
  - The proposed `modules/helpers/quantities.scalar_in_units` (moved and
    generalized, no parallel helper) is approved.

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

## R13 — CI never exercises the xraylib (attenuation) path

**Status:** in progress on branch `quality-remediation-r08-r13-r15`.
xraylib 4.3.0 publishes manylinux wheels for CPython 3.10–3.15 (x86_64 and
aarch64), so CI can install it directly. Plan: add the `attenuation` extra to
the default `[testenv]`.

**Problem.** `xraylib` is the optional `attenuation` extra and is not
installed by any tox environment. Tests that need it are now skipped with
`pytest.importorskip` (fix `c28341e`, PR #148), so in CI the
`chemical_formula` resolution through `xraylib.CS_Total_CP` is never
exercised: `Si` resolves, `Sii` is rejected, and NIST names are accepted
(R01b). Only developer environments that happen to have xraylib run it.

**Approach.** Add the `attenuation` extra to one CI test environment. Either
add it to the `py312` env's `extras`, or create a small dedicated tox env
that runs `tests/modules/technique_modules/scattering/test_material_attenuation_corrections.py`.
First confirm that xraylib installs from PyPI wheels on the CI runner (Linux,
Python 3.12–3.14); if it does not, document that and keep the skip.

**Acceptance criteria.** In at least one CI job, the xraylib tests run rather
than skip; this is visible in the job log.

**Implementation notes.** 2026-10-31: added `attenuation` to the default
`[testenv]` extras, retaining the development `pytest.importorskip("xraylib")`
guards. Verified with `tox -e py312 -- -rs`:

- Before: **1238 passed, 4 skipped**, 2 subtests passed. Skips:
  `tests/integration/test_chunked_memory.py:40` (opt-in RSS check),
  `tests/io/tiled/test_tiled_sink.py:119` (Tiled not installed),
  `tests/io/tiled/test_tiled_source_integration.py:12` (Tiled not installed),
  `tests/modules/technique_modules/scattering/test_material_attenuation_corrections.py:233`
  (xraylib not installed).
- After: **1239 passed, 3 skipped**, 2 subtests passed. Remaining skips are
  exactly the RSS opt-in check and the two Tiled tests above. The xraylib
  formula acceptance/typo rejection test now executes.
- The sandbox's initial dependency download could not resolve PyPI; rerunning
  with network access installed xraylib successfully. Verification here is on
  macOS/CPython 3.12; Linux wheels were not independently tested.

**Review log.**

- **(2026-10-31): approved.** Commit `6dd5eef`. `tox -e py312` now skips 3
  tests (the chunk-memory opt-in and two Tiled tests, which have their own
  env), down from 4: the xraylib test runs. Still to confirm: the CI log after
  the push.

---

## R14 — The `required` argument flag is never enforced

**Status:** open. Found during the R07c review.

**Problem.** Argument specs can say `"required": True`, but nothing at
runtime checks it. Only `scripts/generate_module_doc.py` reads it, to fill the
"Required" column of the reference pages. Since R03, every argument is seeded
from its schema default, so a "required" argument that has a default can
simply be left out. For example, `UnitsLabelUpdate.update_pairs` is required
but defaults to `{}`: a pipeline that omits it gets a step that silently does
nothing.

**Options.**

- **(a)** Enforce the flag: a step whose required argument was not supplied by
  the pipeline, or is still empty, fails at configuration time with a clear
  error. The `required: True` and `default` combinations would need auditing.
- **(b)** Drop the flag and rely on defaults plus per-step validation, removing
  the "Required" column from the reference pages.

**Acceptance criteria.** One meaning for "required", enforced or removed,
with tests. Every existing combination of `required: True` and `default` has
been checked.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R15 — Server: explain source/sink and run failures, and keep logs

**Status:** approved; in progress on branch `quality-remediation-r08-r13-r15`.
Raised 2026-10-31: registering
sources and sinks in a server session returned only a failure code, and no log
said what went wrong.

**Findings.**

1. **Nothing in `modacor.server` logs anything.** No server module uses
   `logging`. When a run fails, `_handle_process_failure` stores the message
   and traceback in the session's in-memory run record (readable through
   `GET …/errors/latest`). It then raises HTTP 500 with the code and message
   in the response body. The server log shows only the uvicorn access line.
   In `20260524_Pauls_modacor/modacor_output/modacor_server.log` there are 68
   lines `"POST /v1/sessions/mouse-server-batch/process" 500 Internal Server
   Error` and not a single error message or traceback.
2. **Error records disappear with the session.** The client deleted the
   session right after the failure, which discarded the in-memory error record.
3. **Registration does not check the file.** `upsert_sources`,
   `upsert_sinks`, `patch_*` and `set_sample_source` only check the shape of
   the request and the path-root policy. A missing or unreadable file (for
   example a local permission problem) surfaces only when the next run fails,
   as a generic `RUN_FAILED`, not as a problem with that source.
4. **No log at all by default.** `LocalServer` sends server output to
   `subprocess.DEVNULL` unless `log_path` is given, and `modacor serve` has
   no option to write a log file.
5. **Policy rejections are clear but easy to miss.** These return HTTP 422
   with a message such as `Runtime policy rejected source.location=…: path is
   outside allowed read roots`. A client that shows only the status code hides
   this message.

**Proposal.**

- A `modacor.server` logger:
  - Every API error is logged: 4xx at WARNING with its detail, 5xx at ERROR
    with the traceback. Each entry includes `session_id`, and `run_id` where
    there is one.
  - Source and sink registration, update and deletion are logged at INFO with
    ref, type and location.
  - Building the sources and sinks for a run is logged. A failure is logged at
    ERROR, naming the ref, type, location and exception.
- Specific error codes for run-time I/O failures: `SOURCE_UNAVAILABLE` and
  `SINK_UNAVAILABLE`, with ref, type and location in the details, instead of a
  generic `RUN_FAILED`.
- At registration, check that file-based sources are readable and that the
  parent directory of file-based sinks is writable. *Open question:* fail with
  422, or return a non-fatal `warnings` list and log a WARNING?
- Logs in a known place. *Open question:* see the options below.
  - `modacor serve` gains `--log-file` (and `--log-level`).
  - `LocalServer` writes to a default log file instead of `/dev/null`, and
    includes the path in its startup and failure messages.

**Decision.** 2026-10-31, maintainer:
- **Log location:** next to the work. `LocalServer` writes to
  `./modacor_output/modacor_server.log` in the working directory by default.
  `modacor serve` logs to the console and gains `--log-file`. The path is
  printed at startup and included in error messages.
- **Unusable file at registration:** warn, then fail at run time.
  Registration succeeds, returns a `warnings` list and logs a WARNING. If the
  file is still unusable when a run starts, the run fails with
  `SOURCE_UNAVAILABLE` or `SINK_UNAVAILABLE`. This supports live acquisition,
  where files appear only after registration.

**Implementation notes.** 2026-10-31, diagnostics implementation:

- Standard module loggers now record API 4xx at WARNING (method, path,
  session and unchanged detail), API 5xx with traceback at ERROR, and run
  failures at ERROR (session, run, code, failed step when known, traceback).
- Successful source/sink registration, patch/sample update and deletion log
  identity at INFO. Existing ApiError response bodies remain unchanged.
- Unexpected handler failures return HTTP 500 with structured `INTERNAL_ERROR`,
  `message` (`ExceptionType: message`) and request context in `details` inside
  the usual HTTP `detail` envelope, with an ERROR traceback.

- File registrations now add `warnings` (always a list) with `ref`, `kind`,
  `location`, `problem`, logging each problem at WARNING. Checks are advisory
  at registration and repeated at runtime; buffer, Tiled and custom types are
  excluded. The runtime Python client emits each warning once as RuntimeWarning,
  including when called through SessionClient.
- Source/sink construction failures carry their registration and original cause.
  Runs return `SOURCE_UNAVAILABLE` / `SINK_UNAVAILABLE`; their HTTP error
  `details` add `ref`, `type`, `location`, `exception_type` alongside session/run
  IDs. The stored latest-error details also retain the traceback. Existing
  non-I/O ApiError payloads and provisional-plan error mappings are preserved.
- File sinks open lazily, so runtime builders explicitly check destination
  access without creating files/directories. A missing parent also warns and
  fails at runtime; it must exist when processing begins. Source checks avoid
  HDFSource's historical missing-file assertion / permission-error logging bug.
- Focused server/client verification at this stage: **100 passed**, including
  real chmod checks as a non-root POSIX user, late-arriving files, all registration
  routes, excluded backends, and single Python warning emission.

- LocalRuntimeServer now resolves the default log path at startup to
  `Path.cwd() / "modacor_output" / "modacor_server.log"`, creates its parent,
  prints it and exposes `log_path`. Its managed client includes the path in
  RuntimeAPIError text without changing server response fields; startup exit
  and timeout messages also point there. Reused external servers do not claim
  a local log. The subprocess output is captured once, without `--log-file`.
- `modacor serve` uses stderr and optional `--log-file PATH`, controlled by
  `--log-level` (default info), with MessageHandler's timestamp/name/level/message
  format. Serve owns the root handlers (including modacor and uvicorn records);
  existing MessageHandler console handlers are removed and later ones delegate
  to that configuration. Tests check one console and file record, including
  repeated setup and early/late handlers. Ordinary MessageHandler use retains
  its existing standalone behavior. Operations docs explain logs and error
  record lifetime; deleting a session discards its in-memory error records.
- Full development suite with access tracing: **1274 passed, 1 skipped**,
  2 subtests passed; the sole skip is the opt-in RSS subprocess test.

- Final tox checks: `py312` **1272 passed, 3 skipped**, 2 subtests passed
  (same three skips as R13 after installation); `lint` passed; `docs` passed
  including module-reference freshness and warning-as-error Sphinx build.
  No existing server test expectations or ApiError responses needed changes;
  no stop condition was hit.

Follow-up fixes (2026-10-31): registration RuntimeWarnings are now emitted
only by SessionClient's registration/sample/patch methods, preserving generic
responses and dry-run string warnings unchanged. For a new file sink, check
the nearest existing ancestor for directory type and write/execute access;
missing intermediate directories are allowed. This supersedes the original
parent-must-exist check. Real HDF/CSV runs into new nested directories and
unwritable-ancestor failures are covered by focused tests.

**Review log.**

- **(2026-10-31): changes requested.** Commits `07aa04d`, `94b0a56` and
  `97c07a0`. The full suite (1274 passed), `tox -e py312` (1272 passed) and
  lint pass, and no existing `tests/server` expectation changed. Logging,
  `INTERNAL_ERROR`, `SOURCE_UNAVAILABLE`/`SINK_UNAVAILABLE` and the log
  locations work as decided. Two defects:
  1. **The client crashes on dry-run warnings.** `RuntimeClient.request`
     treats any `warnings` key as registration warnings (`warning['kind']`).
     The dry-run plan returns plain strings ("No previous ProcessingData
     snapshot available; …"), so `SessionClient.dry_run()` raises
     `TypeError: string indices must be integers`. Reproduced with a fake
     transport.
  2. **Sinks into new directories are rejected.** This comes from my brief,
     which said a sink's parent directory "must exist".
     `check_file_location` raises when the parent directory is missing, both
     at registration and at run time. But the HDF and CSV sinks create
     missing parents themselves (`mkdir(parents=True)`). A run writing to a
     new subdirectory such as `modacor_output/run_42/out.h5` now fails with
     `SINK_UNAVAILABLE`, although it worked before. The correct rule: the
     nearest existing ancestor must be a writable directory.

---

## R16 — Uncertainty propagation gaps found by the R08 audit

**Status:** decision needed (maintainer), row by row. Split out of R08 on
2026-10-31.

**Problem.** The R08 audit found ten places where an output carries no
uncertainty contribution from an uncertain input. Each fix changes published
uncertainties and needs a scientific choice, so none is a plain refactor.
Line numbers refer to the R08 audit snapshot.

| Site | Missing contribution | Scientific question |
| --- | --- | --- |
| `attenuator_plate_correction.py:245-249` | incidence-angle (CosAlpha) sensitivity of the plate transmission | clipping at grazing angles; covariance with the normal-incidence normalization |
| `detector_efficiency_correction.py:231-235` | angle sensitivity of the efficiency | same as above |
| `flat_plate_self_absorption_correction.py:128-133` | CosAlpha sensitivity (transmission terms are already propagated) | clipping; covariance between transmission and incidence |
| `polarization_correction.py:162-166` | TwoTheta and Psi sensitivities | angle covariance through shared geometry; angle wrap; clipping thresholds |
| `xs_geometry_from_pixel_coordinates.py:352-353` | Psi from `arctan2(dy, dx)` gets no x/y uncertainty | origin singularity, branch cut, shared x/y calibration covariance |
| `xs_geometry_from_pixel_coordinates.py:361` | CosAlpha clipping does not update uncertainties | derivative and mask policy at the clip |
| `integrate_1d.py:237-241` | measured axis uncertainty absent from the integral | independent coordinates or a shared axis calibration; duplicate coordinates |
| `find_scale_factor1d.py:303-317` | axis uncertainties are ignored by the fit | fixed abscissae or errors-in-variables |
| `subtract_interpolated_1d.py:225-231` | coordinate sensitivity of the interpolated background | interpolation slope; uncertain bracket membership; correlated axes |
| `helpers/scattering/detector_data.py:129-134` | metadata mean keeps only jitter (SEM), not the measurement components | independence of metadata samples; does jitter replace or supplement measurement errors? |

**Approach.** Decide each row: propagate (which model), document as out of
scope (like the capillary wall-μ in `capillary-self-absorption.md`), or
defer. Implement the accepted rows with model sensitivities in `models` and
attach named components at the BaseData boundary, each with an analytic or
finite-difference test.

**Implementation notes.** _(implementer)_

**Review log.** _(reviewer)_

---

## R17 — BaseData silently drops `propagate_to_all` uncertainties

**Status:** superseded by the uncertainty-propagation specification, Stage 0
(maintainer, 2026-11-01). It is implemented on its own branch,
`uncertainty-stage0` from `main`, as its own PR.
- The interim fix `3ea4ec2` (union rule for mixed mappings) is reverted on
  `quality-remediation-r08-r13-r15`. Stage 0 makes a mixed mapping raise
  `ValueError` instead (R0.1).
- The material is in `auxiliary/modacor_stage0_handoff.zip`: HANDOFF.md, the
  normative specification, `repro.py` and RFC v8. The archive is not
  committed; the specification and the RFC are added to the docs by the
  Stage 0 PR.
- Stage 0 scope (specification §4):
  - one key planner (G + E keeps G, which fixes the loss found here);
  - mixed mappings raise `ValueError`, with the producer "legacy fold" (R0.4);
  - ownership and copy semantics;
  - in-place safety and an in-place `__imul__`;
  - scalar operands without full-size zero arrays;
  - unary operations that do not mutate their input.
- Stages 1 and 2 follow later.

**Problem.** `propagate_to_all` is a special uncertainty key: when it is an
operand's only key, it acts as a global fallback. The rules live in
`_binary_basedata_op` and, for in-place operators, `_uncertainty_key_plan`.
They lose uncertainty in two cases (reproduced on 2026-10-31):

1. **The other operand has no uncertainties at all.** For a global-only
   operand `g` and an exact BaseData `e`, the results `g * e`, `g / e`,
   `g + e`, `g - e` and `e * g` all have `uncertainties == {}`. With a plain
   float (`g * 2.0`) the uncertainty is kept, which is inconsistent.
2. **Mixed operands.** If an operand carries `propagate_to_all` together with
   other keys, its `propagate_to_all` component is dropped, for example
   `mixed * exact` → only `Poisson`. The docstring says that in this case the
   key is "treated like a normal key name" (union), so the code contradicts
   its own documentation.

Unary operations (`**`, `sqrt`) and `indexed` keep the key.

**Impact.** The SAXSess example pipeline gives many of its inputs uncertainty
through `propagate_to_all`: exposure time, dark current, transmitted flux,
detector and sample positions, photon energy and thickness. Wherever such a
value meets an operand without uncertainties, its uncertainty vanishes and
published uncertainties are underestimated. The `MOUSE_solids_operando`
pipeline uses it for `detector_readout_time`.

**Decision.** 2026-10-31, maintainer: `propagate_to_all` stays a special
case and is applied as documented. No supplied uncertainty component is ever
silently discarded. The contract:

| Operands' uncertainty keys | Result |
| --- | --- |
| Both global-only | Keep `propagate_to_all` |
| Global-only + no uncertainties | Keep and propagate `propagate_to_all` |
| Global-only + another non-empty mapping | Fold the global contribution into each key of the other mapping |
| Neither global-only | Union; a `propagate_to_all` mixed in with other keys is an ordinary component |

- **Caveat, to be documented prominently.** Folding does not represent
  covariance. Combining the resulting keys in quadrature (for example with
  `CombineUncertainties`) counts the shared global contribution once per key,
  up to √k times its proper contribution across k keys. When
  `propagate_to_all` is used, keep the propagated uncertainty components
  separate rather than combining them afterwards.
- **Documentation.** Correct the user docs: rule 5 in
  `docs/data-model/uncertainty-propagation.md` currently says a mixed-in
  `propagate_to_all` "is not emitted". Then align both BaseData docstrings
  with the table above.
- **Implementation.** One key-selection implementation for ordinary and
  in-place arithmetic. Tests cover mixed keys, uncertainty-free operands,
  both operand orders, and the equivalence of an exact float and an exact
  BaseData operand.
- **Release notes.** Explain the restored uncertainties. R17 stays a
  separate scientific bug fix: R08's refactors must still preserve results,
  and their equivalence guards are re-run after R17.

**Rules after the fix** are the table above. They change results:
uncertainties that were silently lost reappear. This is a `fix:`, not an
interface change.

**Implementation notes.** 2026-10-31: implemented the maintainer table with
one `_uncertainty_key_plan` shared by ordinary and in-place operators. Removed
the duplicate ordinary key plan and its unconditional global-key deletion.
Exact BaseData and float operands now agree. In-place folding preserves an
unmodified fallback array when the output also writes the global key, avoiding
order-dependent aliasing between components. Propagation formulas, unary
operations, indexing and conversions are unchanged.

- Both arithmetic docstrings and the user uncertainty guide now state the
  four rules and prominently warn about covariance and the √k folding effect.
  The guide has a Changed note; CombineUncertainties carries the requested
  caution and its reference page was regenerated.
- Analytic contract tests cover all rows, mixed/exact/named/global cases,
  both orders, ordinary and in-place +/−/×/÷, exact-float equivalence and the
  k=2 quadrature overcount. Focused arithmetic suite: **188 passed**.
- **Changed existing test expectations: none.** All previous assertions pass
  unchanged. Full suite with access tracing: **1447 passed, 1 skipped**,
  2 subtests passed (RSS opt-in skip).
- MoDaCor-examples `tests` and `DLS/I22/tests`: **23 passed before and
  23 passed after**, with no result changes or files edited by this work.
  That checkout has no `.venv-dev`, so both runs used MoDaCor's
  `.venv-dev/bin/python`; bytecode and pytest cache writes were disabled.
  Existing unrelated modifications in that checkout were left alone.
- R17 is a separate uncertainty-restoration fix. Subsequent R08 comparisons
  use the corrected R17 baseline rather than blessing any module-level change
  in published uncertainty.

Final R17 validation: tox `py312` **1445 passed, 3 skipped**, 2 subtests
passed; `lint` and `docs` passed, including regenerated reference freshness and
warning-as-error Sphinx. Skips remain RSS opt-in and the two optional Tiled
integration tests. The resumed R08 guard found a separate percent-transmission
uncertainty conversion difference, documented in R08 above; implementation
stopped before Commit B and kernel moves. No additional source changes were
made after the R17 commit.

**Review log.**

- **(2026-10-31): approved.** Commit `3ea4ec2`.
  - Re-ran the reproduction:
    - `g*e`, `e*g` and `g*2.0` now give identical `propagate_to_all`
      components;
    - `g/e`, `g+e` and `e-g` keep it;
    - `mixed*e` keeps both `Poisson` and `propagate_to_all`;
    - the in-place `*=` and `+=` match the ordinary operators.
  - One `_uncertainty_key_plan` now serves the ordinary operators and the
    in-place paths.
  - `uncertainty-propagation.md` shows the contract table, a √k warning
    admonition and a "Changed" note. `CombineUncertainties` carries the same
    caution.
  - No existing test expectation changed. The MoDaCor-examples tests (23)
    pass before and after.

---

## R18 — Transmission uncertainties left in percent while the value is converted

**Status:** approved (maintainer, 2026-11-01). The scope is widened from the
two-step transmission fix to central unit handling.

**Decisions (2026-11-01).**

1. **Central conversion in BaseData.**
   - Add a non-mutating `in_units(units) -> BaseData` that shares one code path
     with the in-place `to_units()`. Both convert signal and uncertainties, and
     promote integer signals to float.
   - Weights are dimensionless relative weights by contract; conversion
     leaves them alone. Document this.
2. **Kernel-boundary accessors return value and uncertainties together, in
   the same units.** `magnitudes(units) -> (ndarray, {name: ndarray})` and
   `scalar_components(units) -> (float, {name: float})`, the latter with the
   scalar-or-constant-array policy.
   - `scalar_in_units` is for configuration numbers only and raises
     TypeError for BaseData.
3. **Module-level arithmetic.** It SHOULD go through BaseData, unless that
   measurably and severely affects performance; such an exception is
   commented at the site with its measurement. Measured overhead: about
   90 µs per scalar operation; ×20 on 1,000-point curves (0.12 ms); ×2.7 on
   1 Mpx arrays (9.4 ms against 3.5 ms).
4. **Safe dimensionless reduction, applied only when wrapping scalars.** This
   covers building BaseData from scalar metadata or configuration. The
   reduction:
   - folds pure-number units (%, ‰, ppm, plain scale factors) into the
     magnitude;
   - cancels same-dimension pairs (m/mm);
   - never touches `count`, angle units (`rad`, `deg`, …), `sr` or `px`.

   Pint's own `to_reduced_units()` must not be used: it reduces `count/sr`
   and `count/%` to bare numbers. Results of operations keep whatever Pint
   produces.
5. **Guard test.** An AST guard flags module-local data conversions
   (`.to(…).magnitude`, `.m_as(`, `.m_from(`, `ureg.Quantity(`) under
   `src/modacor/modules`, with a short reasoned allow-list (plot display,
   configuration scalars).
6. **New `ConvertUnits` step for output units.**
   - Configuration: `units: {basedata_key: target_units, …}`.
     `with_processing_keys` takes a list of bundles or `"*"` for every
     bundle.
   - Missing keys: with `"*"`, bundles lacking a key are skipped and logged
     at INFO. With an explicit list, a missing key is an error.
   - Conversion is in place, so axis references stay valid; uncertainties
     are converted and weights are untouched.
   - The dependency contract comes from a mapping-valued role on `units`.
7. **Retire `AngleToQ.output_units`.** AngleToQ returns Q in its natural
   units (1/m, as XSGeometryFromPixelCoordinates does), and pipelines add a
   ConvertUnits step. This is an interface change: an entry in
   `interface-migrations.md`, a minor release, and the MoDaCor-examples
   `DLS/I22/pipelines/I22_USAXS.yaml` updated after the release.
   XSGeometryFromPixelCoordinates gets no `output_units`; ConvertUnits covers
   it.
8. **The transmission bug** (below) is fixed with the central API.

**Sub-steps.**
- **R18a:** API, reduction at scalar wrapping, docs.
- **R18b:** transmission fixes.
- **R18c:** `scalar_in_units` numbers-only, the guard, and migration of the
  flagged sites.
- **R18d:** ConvertUnits, AngleToQ retirement, migration note, author-guide
  "Units" section.

Every change outside the listed intentional ones must leave results
identical.

**Problem.** Two steps convert the measured transmission's nominal value to
dimensionless, but read its uncertainties from the unconverted BaseData. A
transmission given in percent therefore enters with σ in percentage points
against a dimensionless value, which makes the transmission uncertainty
contribution 100 times too large.

- `CapillarySelfAbsorptionCorrection`, with `input_state:
  transmission_normalized`: `_measured_transmission()` returns the BaseData
  in its configured units. `measured_value` is converted with
  `scalar_in_units`, but line ~757 uses `measured_transmission.uncertainties`
  as they are. For T = 50 % ± 1 pp the active-pixel contribution is
  1.1909, where 0.0119 is correct (reproduced by Codex).
- `FlatPlateSelfAbsorptionCorrection`: line 104 converts the value, line 127
  uses `transmission_data.uncertainties` raw. Reproduced: 0.5 ± 0.01
  (dimensionless) and 50 ± 1 (percent) give identical factors, but σ is
  [0.00223, 0.00106] against [0.223, 0.106], exactly 100 times larger.

The sample-μ path (`_scalar_components`) converts a copy with `to_units`, so
it is correct. Inputs that are already dimensionless are unaffected.

**Fix.** Convert the transmission BaseData (value and uncertainties together)
to dimensionless once, and use that object for both. Test that percent and
dimensionless inputs give identical factors and uncertainties for both steps.
This changes results only for non-dimensionless transmission inputs, so it is
a `fix:`.

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

# MoDaCor RFC: staged redesign of uncertainty propagation (v8)

**Status:** Agreed design (2026-11-01), basis for the normative Stage 0/1 specification; not an implemented change
**Implementation horizon:** **Stages 0 and 1 are near-term implementation priorities.** Stage 2A–2D is the subsequent near-future architectural work, designed now but not a prerequisite for shipping the Stage 0 and Stage 1 fixes. One Stage 2 item is pulled forward: the *type* of elementary identities (§9, Decision 6), because Stage 1 component IDs seed Stage 2 source IDs.
**Target:** MoDaCor `BaseData` uncertainty model and affected correction/reduction modules
**Audited code:** [BAMresearch/MoDaCor](https://github.com/BAMresearch/MoDaCor) `main` at commit `56efd4a` (2026-10-31), chiefly [`basedata.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/dataclasses/basedata.py), `uncertainty_tools.py`, the uncertainty-producing and reducing modules, and the in-repository pipeline configurations
**Related discussion:** [Issue #70: uncertainty keys and `propagate_to_all`](https://github.com/BAMresearch/MoDaCor/issues/70)

## What changed since v7

1. **Project principle P0: no backward compatibility (new, §0, §9).** MoDaCor is in early beta. Uncertainty handling may change between releases without compatibility layers, deprecation adapters or bit-identical legacy outputs. Anyone who needs an earlier result exactly can run the release that produced it.
2. **The legacy sidecar is replaced by `σ_indep` (§7.0.1, §7.5).** v7's parallel Stage 0-compatible propagation engine is dropped.
   - The single-field default is now the independence approximation `σ_indep = sqrt(est² + ek² + T²)`.
   - `T` carries the same shared terms as `S`, combined in quadrature and without the `√k` factor.
   - It costs one extra array shared by all alternatives, and it is proven to lie inside the bracket.
   - v7's correction stands: the value an *older release* produced can lie outside the bracket. That value is no longer reproduced.
3. **ESSreduce precedent stated precisely (§5, §11.3).** ESSreduce's upper-bound broadcast multiplies the variance by the broadcast volume. That is the same Cauchy–Schwarz factor as `k`, applied at broadcast time. MoDaCor applies it when overlapping elements are combined, which keeps 2D per-pixel marginals exact.
4. **Consequences of P0 elsewhere.**
   - The Stage 2 migration no longer provides a read-only legacy `uncertainties` view.
   - "Legacy key" wording becomes a porting guide for pipeline configurations.
   - Decision 13 is reduced accordingly.
5. **Small fixes.**
   - §0 status wording and checklist B5 (`est` or `ek`).
   - Final status names deferred to the normative specification, with a proposal that avoids the word "guaranteed".
   - Fixes needed in the draft public guide recorded in §11.5.

The executable specification `stage1_bound.py` now carries `T`. The property test asserts on all 76 814 elements that `σ⁻ ≤ σ_indep ≤ σ⁺` and that `σ_indep` equals the independence-assumption engine; 16/16 tests pass.

## 0. Agreement checklist (to settle before the normative specification)

**Status: agreed on 2026-11-01.** The items below are the basis of the normative specification. Items marked *v8* changed in this revision as a direct consequence of P0 or of the σ_indep decision; all others were stable across at least two iterations.

**Principle**

| # | Proposal | § |
| --- | --- | --- |
| P0 | *v8:* No backward compatibility in uncertainty handling. No compatibility layers, deprecation adapters, legacy views or bit-identical legacy outputs; earlier results are reproduced by running the earlier release | 9 |

**Stage 0**

| # | Proposal | § |
| --- | --- | --- |
| A1 | Mixed `propagate_to_all` mappings raise `ValueError`, with no warning-only transition release | 6, 9.1 |
| A2 | Global-only meeting empty keeps the global term (F1); a single key planner is used for all paths | 6.2–6.3 |
| A3 | In-place ops are alias-safe and preflighted; a failed op leaves the target untouched (F5–F7) | 6.3 |
| A4 | Transient memory target: at most (output keys + 1) frame-sized allocations per binary op (F9) | 6.3 |

**Stage 1**

| # | Proposal | § |
| --- | --- | --- |
| B1 | The dictionary holds *alternatives* with a declared estimator target; components are folding operations, not keys | 4, 7.0 |
| B2 | S1-A's bridging rules activate only together with S1-B's `S` | 7.0, 8 |
| B3 | Shared bound `S` with rules B1–B8; bracket `[σ⁻, σ⁺]` only with `bound_status = guaranteed` | 7.0.1 |
| B4 | Overlap handled by the multiplicity class `ek` and `k`; operators declare fan-out; unknown fan-out gives `not_guaranteed` | 7.0.1 |
| B5 | Origin tracking flags reuse through `est` or `ek`, and fit-input reuse (E2/E3) | 7.0.1 |
| B6 | Input coverage `exact`/`quantified`/`unquantified` with producer-declared defaults, `unquantified` only at the IO boundary | 7.0 |
| B7 | Model completeness is one pipeline-level field, default `not_assessed`; S1-C **must precede public bracket activation** | 7.0 |
| B8 | *v8:* Sinks write named bound fields. Single-field formats default to the labelled independence approximation `σ_indep = sqrt(est² + ek² + T²)`, which costs one array and provably lies inside the bracket; no legacy sidecar. Strict mode requires an explicit policy | 7.0.1, 7.5, 9.7 |
| B9 | The elementary-identity *type* is fixed before S1-A starts | 9.6 |

**Public documentation release obligations**

| # | Proposal | § |
| --- | --- | --- |
| D1 | Update the existing public `docs/data-model/uncertainty-propagation.md` with truthful Stage 0 rules in the Stage 0 release | 11 |
| D2 | Publish the Stage 1 uncertainty guide, examples, bounds and limitations, and prior-art citations **in the same release** that activates Stage 1 publicly | 11 |
| D3 | Check docs build, toctree/navigation, local links, all numerical examples, and code-versus-doc parity; do not publish Stage 2 behavior as implemented until shipped | 8, 11 |

**Stage 2 contracts that constrain Stage 1**

| # | Proposal | § |
| --- | --- | --- |
| C1 | Non-overlap rule: every elementary contribution is held exactly once (estimate, source or lineage) | 2, 4.1 |
| C2 | Identity is per realisation, never a path (§2-IP) | 2 |
| C3 | Lineages are declared where the estimate is created | 7.2 |
| C4 | Fit covariance partition uses the unscaled covariance; no automatic χ² inflation | 7.3 |

## 1. Decision summary

| Stage | Purpose | Data-model change |
| --- | --- | --- |
| **0: correctness hotfix — near term** | Lossless, exhaustive `propagate_to_all` handling; reject mixed mappings; fix in-place aliasing and partial mutation; cut transient allocations | No new uncertainty container |
| **1: semantic contract and shared bound — near term** | Distinguish `alternatives` from `components`; route broadcast and declared-shared dependencies to `S`; track overlap from many-to-many operators with a multiplicity class; produce *conditional* first-order brackets for eligible outputs; deduplicate components; separately report input coverage and model adequacy; Monte Carlo reference harness | Semantics and coverage metadata, provenance, and one unsigned `S` array per `BaseData`; still no signed, source-aware engine |
| **2A: shared scalar sources — near future** | Signed, identified responses for thickness, time, transmission, flux, calibration, dark level and similar; correct through 2D→1D reduction | New `UncertaintyModel` |
| **2B: shared statistical lineages — near future** | Per-pixel statistical noise reused across branches (operando background, empty cell, solvent) | Extends 2A |
| **2C: fitted parameters — near future** | `FindScaleFactor1D` and similar: parameter covariance and correlation with the data the fit used | Extends 2A/2B |

**Delivery policy:** Ship Stage 0 once its correctness **and public-documentation** gates pass. Stage 1 can then ship as a deliberately *approximate*, documented compatibility layer. Neither stage may claim correlation-*correct* 1D uncertainties. Stage 1 may claim the **model-conditional** first-order bracket of §7.0.1 only when its eligibility, stochastic-input coverage and output-support rules are met; other outputs must be visibly flagged or rejected in strict mode. Build Stage 2 against the original inputs or retained full-provenance data rather than attempting to invert folded standard deviations. Stage 2A precedes 2B and 2C; Stage 2D denotes migration and release validation.

**Key design choices:**

- **Stage 2**: `uncertainties` (later `estimates`) is a mapping of **competing estimates of elementwise-independent, unshared residual statistical noise**. Examples are `Poisson`, `SEM` and `STD`. **Stage 0** retains legacy mappings, which may include folded shared terms. **Stage 1** moves broadcast and declared-shared terms out of the mapping into the bound `S` (§7.0.1); other folded terms remain `legacy_folded`.
- **Independent components** (readout noise, pixel-index quantisation when genuinely independent) are folded into alternatives by an explicit operation, with deduplication by elementary component identity. Stage 0 component keys are not supported beyond Stage 1 activation (P0).
- **Shared sources** (one uncertain parameter affecting many pixels or branches) and **shared statistical lineages** (one measured array reused in several branches) live in their own containers. They carry signed responses and stable identities; they must not also reside in the independent residual estimates.
- `propagate_to_all` survives in Stages 0–1 as the *model-agnostic estimate*, never a reliable shared-source ID. In Stage 2, typed physical sources replace the use of this wildcard for uncertain thickness, time and similar shared corrections.

## 2. Definitions and worked examples

- **Alternative estimate:** a 1σ estimate of the **same output measurand and statistical target**, under a particular statistical model, excluding explicitly separated shared contributions. Valid alternatives may be compared or selected but are never added in quadrature. The estimator must declare what it estimates (e.g. uncertainty of a mean, spread of individual observations, or an uncertainty of a coordinate).
- **Independent component:** a contribution known to be uncorrelated with everything else in the estimate. It is folded into an alternative in quadrature exactly once.
- **Shared source (scalar latent):** one uncertain parameter `p_s` that affects many elements or branches. Its signed response is `g_{s,i} = (∂y_i/∂p_s) σ_{p_s}`. Responses add linearly before squaring.
- **Shared statistical lineage (elementwise latent):** the per-element noise of one measured array (for example a background frame) that enters several branches. Responses add linearly *per element* across branches and in quadrature *across elements* during a reduction.
- **Anchor:** an immutable array shared by many responses. A response is stored as `g = Σ_j c_j · a_j`, with few anchors `a_j` and scalar (or small) coefficients `c_j` (§7.4).
- **Scope:** physical extent of sharing (pixel, exposure, sample, calibration run). Scope aids validation; it is not identity.
- **Elementary contribution:** a single latent random quantity (or independent element of a latent vector), with unique identity within a run. Two occurrences of the same elementary contribution are the *same noise*, not two quadrature contributions.
- **Unquantified input:** uncertainty information has not been supplied. This is different from a constant documented as exact for the model. An empty mapping alone is not evidence of exactness.
- **Estimator target:** the physical output quantity and statistical meaning of an uncertainty (e.g. `intensity_mean / standard_uncertainty` versus `Q_bin / descriptive_spread`). Values with different targets are **not** competing alternatives.
- **Model completeness:** whether the inputs and dependence assumptions required by the *declared* stochastic model have been supplied and validated. This is separate from whether one or more inputs happen to carry a quantified σ.

**Non-overlap rule:** For any selected alternative and any elementary contribution, exactly one representation holds its variance: independent residual (`estimate`), identified scalar/vector source, or identified statistical lineage. Its effects across multiple arithmetic paths are combined via signed response before variance is formed. A transfer between representations must remove the prior contribution; it must never merely copy it.

**Statistical assumption:** Distinct elementary source/lineage IDs are independent *only by explicit model declaration*. Correlated parameters must be expressed via shared elementary latent variables (e.g., whitening/factorization) or an explicitly supported joint covariance.

**Identity principle (referenced throughout as §2-IP):** an elementary identity names a *realisation* of an uncertain quantity: this thickness measurement, this background acquisition, this calibration run. It does not name a storage path, array name, dictionary key, shape or broadcasting pattern.
- The same path can hold several independent realisations, and different paths can reference one.
- Paths, names and broadcasting are provenance and *risk detectors*. They may trigger Stage 1 routing to the conservative bound `S` (§7.0.1), which is safe whatever the true correlation.
- They never establish independence or common identity for signed Stage 2 propagation, which needs a producer declaration.

Unless explicitly marked as a **descriptive spread**, reported uncertainty estimates are absolute 1σ standard uncertainties. A standard deviation of Q values *within a bin* is a useful estimate of the bin's Q spread, but it is not automatically the standard uncertainty of its mean Q coordinate. An intensity SEM can estimate the uncertainty of a binned mean intensity if the estimator's effective-sample-size and dependence assumptions hold. These are legitimate outputs on **different measurands**, not alternative estimators to combine with each other. An intensity STD may be retained as a scatter diagnostic (or explicitly selected conservative surrogate), but must not silently masquerade as the SEM of the mean. Expanded uncertainty is out of scope.

**Example 1 (v1, unchanged).** Take `I=100`, `t=2.00±0.10`, and alternatives `σ_P=10`, `σ_N=15`. The corrected signal is `50`. The statistical-only uncertainties are `5` and `7.5`, and the thickness response is `2.5`. The complete alternatives are `5.59` and `7.91`; taking their maximum counts thickness once. For the mean of 100 such pixels the result is `sqrt(5²/100 + 2.5²) = 2.55`. Treating thickness as independent per pixel gives `0.56`. MoDaCor's `ReduceDimensionality` reports `0.559` today, and a Monte Carlo run with one thickness per realisation gives `2.57` (§3.1, F10).

**Example 2 (operando, new).** One pixel, `K=50` frames at 1000 counts each and one background at 900 counts, subtracted from every frame:

| Quantity | MoDaCor today | First order | Monte Carlo |
| --- | --- | --- | --- |
| Time-average of 50 frames | 6.16 | 30.33 | 30.33 |
| Difference of two frames | 61.64 | 44.72 | 44.66 |

The background noise does not average down over frames, and it cancels exactly in frame differences. Today's model gets the first wrong by a factor of about 5 and the second is roughly 38% higher than the reference. The same numerical probes are available in `correlation_checks.py`; results are stochastic and subject to Monte Carlo sampling error.

## 3. Audit of the current implementation

**Method.** The audit used a fresh clone of `main` at `56efd4a`. PyPI was unreachable from the audit sandbox, so `pint` was replaced by a minimal stand-in whose conversion factors are all 1. This does not affect key handling or propagation arithmetic, and the stand-in reproduces v1's 5.59 exactly. The supplied diagnostic scripts accompany this RFC (`repro.py`, `correlation_checks.py`). They require a MoDaCor checkout with its dependencies installed (including `pint`) and use the code paths available in the audited revision. They mainly print results rather than assert them; they must be converted into version-pinned, assertion-based regression/Monte Carlo tests before being used as release gates.

### 3.1 Findings

| ID | Finding | Location | Reproduced result |
| --- | --- | --- | --- |
| **F1** | Global-only operand meeting an operand with no uncertainty loses its uncertainty. The `else` branch takes the union of *non-global* keys, which is empty. | `_binary_basedata_op` L293–306; `_uncertainty_key_plan` L401–408 | `a / t`, `a /= t`, `t * a` all give `{}` |
| **F2** | A mixed mapping silently drops the `propagate_to_all` term (treated as an ordinary key, then popped). | L303–306, L363–365 | `{Poisson:10, propagate_to_all:20} / 2` → `{Poisson: 5}` |
| **F3** | Mixed mappings arise from ordinary pipeline order: `PoissonUncertainties` adds `Poisson` next to an existing global-only key. `combine_uncertainty_keys` has no reserved-key guard either. | `poisson_uncertainties.py` L74; `uncertainty_tools.py` | Keys become `{Poisson, propagate_to_all}`; next division gives 5.00 instead of 5.59 |
| **F4** | Partial key overlap produces an incomplete alternative under the union rule. | L305 | `{Poisson:10, SEM:15} − {Poisson:7}` → SEM `15.0`; complete alternative is `16.55` |
| **F5** | In-place ops write into uncertainty arrays that other objects share. `signal_converter` does not copy arrays, so the constructor and `dict(...)` share them, and `_writable_full_shape_uncertainty` returns the caller's array for in-place update. | L88–92, L412–418, L436/517 | After `p /= 3`, an unrelated `q` sharing the array changes 3 → 1; a `dict()` "copy" changes 3 → 5 after `r -= other` |
| **F6** | Self-aliasing: in `a /= a` the divisor is a view of the signal, which is overwritten before uncertainties are propagated. | L474–482 | σ `[2.83, 4.24]` instead of `[0.707, 0.471]` (4× and 9×) |
| **F7** | Partial mutation: the signal is updated before the key plan and uncertainty broadcast are validated. | `_inplace_add_sub` L505–512; `_inplace_truediv_metadata_neutral` L478–482 | Failed `-=` raises `ValueError` with the signal already changed 10 → 9 |
| **F8** | Shared corrections in sample − background are counted twice (the integration-test pipeline reads exposure time and dark current for both branches from the same paths). | `tests/integration/test_pipeline_run.py` steps 2–3, `bg` | Dark term `√2·σ` instead of exactly 0 **if both branches reuse the same physical dark measurement**; time term `0.0101` instead of `0.0090` **if the uncertain exposure-time parameter is shared**. Small here, structurally wrong under those source-identity assumptions. |
| **F9** | Scalar operands are coerced into full-size signal and zero-uncertainty arrays per key, and the generic path materialises every broadcast term. | `_binary_op` L607–638; L326–359 | `frame * 2.0` with a 2048² frame and 3 keys peaks at **352 MiB** for a 128 MiB object |
| **F10** | Reductions treat folded shared terms as independent per element. | `ReduceDimensionality`, `Integrate1D._integrated_uncertainty`, `IndexedAverager._propagate_uncertainties` | 0.559 vs 2.55 (Monte Carlo 2.57) |
| **F11** | `FindScaleFactor1D` computes the full parameter covariance but exports only the diagonal. The covariance is the unscaled one multiplied by the reduced χ² (`residual_variance`). With `uncertainty_key=None` it combines *all* keys in quadrature (component semantics). | `scaling.py` L231–242; `find_scale_factor1d.py` L38–46 | Monte Carlo corr(scale, background) = −0.78, discarded |

Other observations, outside this RFC's scope but worth fixing:

- There is no `__imul__`, so `a *= b` silently falls back to out-of-place.
- The unary `log`/`exp`/trigonometric helpers call `to_dimensionless()`/`to_units()` on their *input*, which mutates the caller's object.
- `to_units` scales stored uncertainty arrays in place, which inherits F5.
- Unary propagation uses `|f'(x)|`. That is correct for unsigned estimates but must keep its sign for Stage 2 responses.

### 3.2 Point 1: the codebase uses both semantics

| Place | Semantics implied |
| --- | --- |
| `BaseData` docstring ("Keys are treated as *independent sources*"); `docs/data-model/uncertainty-propagation.md` ("Named-component rules") | Components |
| `CombineUncertainties` docstring example `stat_total: [poisson, readout]`, `geometry: [pixel_index_slow, pixel_index_fast]` | Components |
| Geometry modules: `pixel_index_fast/slow`, `pixel_pitch_jitter`, `detector_position_jitter`, `sample_position_jitter`, `wavelength_jitter` on coordinate and Q data | Components (and the scalar ones are shared sources) |
| Correction-factor inputs in tests and docs: `sample_thickness_SEM`, `transmission_SEM`, `sample_absorption_SEM` | Components (in fact shared scalar sources) |
| `FindScaleFactor1D._combined_sigma` | Components (quadrature of all keys) |
| `PoissonUncertainties` → `Poisson`; scatter estimators in `ReduceDimensionality` / `IndexedAverager` → `SEM`, `STD`; `CombineUncertaintiesMax` | Alternatives |

A typical signal therefore ends up as `{Poisson, sample_thickness_SEM, transmission_SEM, …}`, a mixture of one alternative and several shared sources. Neither `max` nor quadrature is right for that mapping.

Two consequences are less obvious:

- **Estimates created mid-pipeline are incomplete.** A `SEM` created from scatter at the reduction step contains no upstream shared contributions, because a common scale does not create scatter. `Poisson` contains them, but under-reduced (F10). After azimuthal averaging, `max(Poisson, SEM)` can therefore omit thickness, time and transmission entirely.
- **Precedent for Stage 2 exists.** `CapillarySampleContainerCorrection` already computes a *signed total derivative* with respect to μ through two paths and then emits `|derivative|·σ` per source name (see `test_composite_propagates_derived_mu_uncertainty_as_one_correlated_source`). Stage 2 generalises this module-local technique and keeps the sign and identity beyond the module boundary.

### 3.3 Pipeline configurations

The **audited repository snapshot** contains no stand-alone production pipeline YAML. Pipelines appear inline in tests and docs; this does not establish the absence of external deployments.

- `propagate_to_all` is configured only as the *sole* key of an uncertainty source: the exposure-time divisor and dark-current subtrahend in `test_pipeline_run.py`, and the `FindScaleFactor1D` output defaults.
- No in-repo configuration constructs a mixed mapping directly. Mixed mappings arise only through module ordering (F3) or `CombineUncertainties` with `drop_source_keys: false`.
- A hard `ValueError` for mixed mappings (Decision 1) therefore breaks no in-repo configuration. External pipelines (MOUSE and others) should be dry-run once against the Stage 0 release.

## 4. Data model: alternatives, components and shared terms

### 4.1 Containers (Stage 2 target)

```python
@define
class UncertaintyModel:
    # Alternative estimates of residual elementwise-independent noise only.
    # Excludes contributions already allocated to sources or lineages.
    # Unsigned 1σ, broadcastable to the signal; MAY hold a single reserved ANY entry.
    estimates: dict[str, Array]
    # Scalar-latent shared sources. Signed responses keyed by stable SourceId.
    sources: dict[SourceId, Response]
    # Elementwise-latent shared statistics. Signed responses per lineage, per alternative.
    # Each lineage's variance is TRANSFERRED out of independent estimates.
    lineages: dict[LineageId, dict[str, Response]]
    # Immutable, reference-counted arrays referenced by Responses (copy-on-write).
    anchors: AnchorStore

@define(frozen=True)
class Response:
    terms: tuple[tuple[AnchorRef, Coefficient], ...]   # g = Σ c_j · a_j
    # AnchorRef ∈ {SIGNAL (this object's signal), ONE (broadcast 1), StoredAnchor(id)}
```

The final *marginal*, per-element 1σ for a selected alternative `k` is:

`σ_k = sqrt( est_k² + Σ_ℓ (r_{ℓ,k})² + Σ_s g_s² )`

This formula is valid only when (i) the terms have **disjoint elementary-noise support**, (ii) distinct source/lineage elementary latents are independent after any required whitening, and (iii) each response is normalized to one standard deviation. It is **not** a license to sum an existing Poisson variance again after representing that same measurement as a lineage.

For example, if a background measurement has `Poisson = 30`, promotion to `lineages[background_id]["Poisson"]` must set its corresponding independent residual contribution to zero (or subtract that contribution from an independently known variance budget). Retaining `est["Poisson"] = 30` *and* `lineages[background_id]["Poisson"] = 30` would incorrectly produce `sqrt(30² + 30²)`. Arbitrary total σ arrays cannot be split by subtraction unless their constituent variance budget is actually known; otherwise the conversion must be rejected or the original measurement re-ingested.

**Preservation invariant:** an elementary contribution occurs once in the model, even when its signed response has multiple algebraic paths. The resulting `Cov_k = diag(est_k²) + Σ_ℓ R_{ℓ,k}R_{ℓ,k}ᵀ + Σ_s g_sg_sᵀ` is schematic: a lineage `R` has one column per independent element, not one fully correlated scalar column. It should normally be implemented as operators, never materialized as a dense covariance.

Selection (for example the per-point maximum over `k`) happens only when assembling outputs, never during the pipeline. Because **source** responses are shared across alternatives, the per-point maximum of complete totals is compatible with selecting the maximal residual-plus-lineage variance and adding common source variance once. Lineage terms **may be alternative-dependent** and must be included before comparing candidates; a maximum over `est_k` alone is not generally valid.

### 4.2 Components are operations, not keys

- `AddIndependentComponent(σ_c, into=all|[k…])` folds `σ_c` in quadrature into named alternatives only where its independence and absence from the selected alternative are established. Record a stable **ComponentId** in provenance **per destination estimate**, not merely a display name. Repeating the same ComponentId for the same alternative is an error by default; explicit replacement must subtract/rebuild from recorded components rather than add twice. Readout noise and pixel-index quantisation belong here only when independent at the relevant reduction level.
- Ingestion of configured `*_uncertainties_sources` must ultimately declare each elementary contribution as `component`, `source` or `lineage`, with its physical scope and independence assumptions documented (§2-IP).
  - In Stage 1, broadcasting routes a contribution to the conservative bound `S` (§7.0.1).
  - In Stage 2, an explicit declaration or validated producer metadata is required; ambiguous inputs fail validation.
  - A full-shaped reused background is still a shared lineage. A per-exposure scalar jitter is not necessarily common between exposures.
- **SourceId identifies an elementary uncertain quantity (§2-IP).**
  - A clock-calibration error uses one ID across the relevant exposures. Independent per-exposure timing jitter uses distinct IDs.
  - A thickness measurement uses one ID even when referenced in two branches.
  - Cancellation in F8 is valid only when the correction is demonstrably the *same uncertain quantity*.
  - The integration test's background branch reads the *sample's* exposure time, which is likely a test shortcut rather than a physical claim.
- `CombineUncertainties` (quadrature) is retained to *construct* an alternative from **proven independent components** at ingestion. Applied to two alternatives, it is an error. `CombineUncertaintiesMax` becomes the output-assembly *selection* step. Reusing any component identity within a resulting estimate must be blocked.

### 4.3 Binary rules for `estimates`

| Left | Right | Result |
| --- | --- | --- |
| ∅ | ∅ | ∅ |
| `K` | ∅ | `K` algebraically; flag unquantified contribution if the empty side is a measured input (and mirrored) |
| `ANY` | ∅ | `ANY`, with the same exact/unquantified caveat (fixes F1; mirrored) |
| `ANY` | `K` | `K`, the `ANY` term contributing to each (mirrored) |
| `ANY` | `ANY` | `ANY` |
| `K` | `K` (identical sets) | pairwise |
| `K` | `{k'}` (one estimate) | `K`; `k'` fills every alternative, logged (mirrored). Example: a background with only `Poisson` meets a frame-averaged sample with `{Poisson, SEM}`. |
| `K₁` | `K₂` (≥2 each, sets differ) | `ValueError`, unless the operation is configured with an explicit `fill` mapping |
| mixed `ANY` + named | any | `ValueError` before any mutation |

`sources` combine by union of *elementary physical* IDs, linearly and with sign. `lineages` combine by union of their *elementary measurement* IDs. Where a lineage appears on both sides, its per-alternative responses follow the table above and then add linearly; its variance must **not** also be in an independent residual estimate. Different IDs are treated as independent only if the input model asserts independence (or supplies an equivalent decorrelated factorization).

The `K`-versus-single-estimate fallback is a **documented model-selection policy**, not evidence that statistical methods are interchangeable. Record `filled_from` provenance and allow strict callers to reject it. Undefined or multiply conflicting estimate mappings fail instead of silently creating incomplete totals.

## 5. Prior art and parallels

| Prior art | What it does | Parallel in this RFC | Lesson |
| --- | --- | --- | --- |
| **GUM**: JCGM 100:2008 (law of propagation with covariance), Suppl. 1 JCGM 101:2008 (Monte Carlo), Suppl. 2 JCGM 102:2011 (multivariate outputs) | First-order propagation including correlation terms; Monte Carlo as reference | §7 propagation laws; §8 Monte Carlo acceptance tests | Monte Carlo checks computations **under the declared stochastic input model**; it is not evidence that the model matches experimental reality |
| **`uncertainties`** (E. Lebigot) | Each value carries derivatives with respect to every independent variable it depends on, so correlations are exact to first order | Same mathematics as `sources` | Performance comes from layout. It stores a derivative *map per element* in Python objects, so a 4 M-pixel mean carries a 4 M-entry map. This RFC stores the transpose: one vectorised array (or anchor term) *per source*, with elementwise latents left implicit in `estimates`. That gives the same correctness for the correlations that matter at numpy speed. |
| **GUM Tree Calculator, GTC** ([MSL New Zealand](https://gtc.readthedocs.io/), B. D. Hall) | "Uncertain numbers" track sensitivities to elementary inputs; *intermediate* results support component budgets; *ensembles* declare correlated inputs | `SourceId` ≈ elementary input; diagnostic budgets (§7.5); Cholesky-whitened parameter sets ≈ ensembles | Stable identity of elementary inputs is what makes budgets and cancellation possible |
| **METAS UncLib** ([METAS](https://www.metas.admin.ch/en/metas-unclib); Zeier et al., *Metrologia* 2012) | Linear propagation with sensitivities to base inputs, including vector and matrix operations | Same, array-native | Array support is feasible if sensitivities are stored per base input, not per element |
| **Scipp / Mantid**: [Heybrock, Vaytet, Wynen, *J. Neutron Res.*](https://content.iospress.com/articles/journal-of-neutron-research/jnr220049) | Shows that broadcasting a lower-dimensional normalisation term across pixels, then summing, histogramming or fitting, systematically suppresses its uncertainty. Scipp ≥ 23.01 raises an exception when an operand with variances is broadcast. | Exactly F10. `propagate_to_all` on a scalar is the broadcast trigger. | Scipp's trigger (broadcast of an uncertain operand) is a useful non-name-based **risk detector**, not proof of physical correlation. MoDaCor's 2D-first pipeline must broadcast, so Stage 1 does not raise. Instead it routes such contributions to the conservative bound `S` (§7.0.1). Stage 2 uses declared source identity. |
| **ESSreduce** (`ess.reduce.uncertainty`, Scipp ecosystem) | `broadcast_with_upper_bound_variances` scales the variance by the broadcast volume, an upper bound for later reductions | Same factor as the multiplicity `k` (B6b/B6c), applied at broadcast rather than at combination | Precedent for conservative upper bounds; MoDaCor's later application keeps per-pixel marginals exact |
| **GUM fully-correlated limit**: JCGM 100:2008 §5.2.2, Note 1 | With all correlation coefficients +1, the combined uncertainty is the *linear* sum of sensitivity-weighted uncertainties | `S` applies that limit to the shared part only, with absolute sensitivities, which makes it an upper bound for any sign pattern | The Stage 1 bound is a familiar GUM construction, not an ad-hoc heuristic |
| **HEP systematics**: HistFactory (Cranmer et al., CERN-OPEN-2012-016); Barlow–Beeston | Each systematic is a signed ±1σ variation, fully correlated across bins and independent across sources. Normalisation (`OverallSys`), shape (`HistoSys`) and per-bin statistical (`StatError`/`ShapeSys`) terms are kept apart. | `OverallSys` ≈ source anchored on `SIGNAL` (scalar relative response); `HistoSys` ≈ source with its own anchor; `StatError` ≈ `estimates` and `lineages` | Mature, widely understood vocabulary for explaining the design to users |
| **Forward-mode automatic differentiation** (JVPs) | Push a tangent per input direction through the computation | Each source is one tangent; reductions are linear in tangents | Implementation template for arithmetic and unary ops (sign retained) |
| **Diagonal-plus-low-rank (factor) covariance** | `Cov = D + G Gᵀ` | `D` from `estimates`, `G` from `sources`; anchors give `G = A C` | No dense covariance needed; memory scales with rank |
| **Peelle's Pertinent Puzzle**; D'Agostini, *NIM A* 346 (1994) 306 | Least-squares fits using a covariance matrix with multiplicative normalisation terms can be biased | Downstream fits (e.g. McSAS) and §7.3 | Export multiplicative sources as nuisance parameters (a relative scale with its σ), not baked into a covariance matrix |
| **pyFAI** error models (`poisson`, `azimuthal`) | Alternative estimators for integrated intensities: counting statistics vs. scatter within a bin | `estimates` as alternatives | The alternatives semantics is established practice in azimuthal integration |
| **Pauw et al., modular SAXS correction sequence** ([arXiv:1706.06769](https://arxiv.org/pdf/1706.06769), *J. Appl. Cryst.* 2017) and "Everything SAXS" ([arXiv:1306.0637](https://arxiv.org/pdf/1306.0637)) | Note that for dispersions the uncertainties of several operations are added twice, and suggest postponing corrections | F8; reordering | That mitigation conflicts with the 2D-first universal pipeline, which is why Stage 2 carries sources instead |

## 6. Stage 0: correctness hotfix (no data-model change)

### 6.1 Invariants

1. Each entry is an absolute, nonnegative 1σ in signal units, broadcastable to the signal. NaN handling follows the documented arithmetic policy.
2. `propagate_to_all` is the sole key or absent. Mixed mappings are invalid.
3. No operation silently discards a nonzero uncertainty. An empty mapping on a user-supplied/measured `BaseData` means **no quantified contribution supplied**, not a claim of exactness. Stage 0 arithmetic necessarily treats absent terms as zero *in the computation*, and its documentation says so. Durable incomplete-coverage labels need metadata and therefore belong to Stage 1 (S1-C). A Python numeric literal or explicit Pint constant passed to arithmetic is **treated as exact by API contract**. Do not infer that an arbitrary `BaseData(uncertainties={})` is exact.
4. In-place and out-of-place forms of the same operation give identical values, units, keys and errors.
5. Invalid input fails before any mutation of signal, units or uncertainty arrays.
6. **(new)** No operation mutates an array it does not exclusively own. A `BaseData` owns its uncertainty arrays after construction.
7. **(new)** Self-aliased operands (`a op= a`, or operands sharing arrays) give the same result as with operand snapshots. This is a **computational alias-safety guarantee**, not a statement that independent-operand uncertainty formulas model the physical correlation in `a - a` or `a / a` correctly.

### 6.2 Key-resolution table

This is the v1 table, with F1 fixed and one row added:

| Left keys | Right keys | Output |
| --- | --- | --- |
| empty | empty | empty; unknown uncertainty is still not proven zero |
| named | empty | left keys (mirrored), potentially incomplete for unquantified measured inputs |
| named | named, identical sets | pairwise, independent-operand formulas |
| named | named, sets differ | **union, as today**, plus a `DEBUG` log and a provenance note. *Not* an error in Stage 0, because geometry modules rely on union (component) semantics; Stage 1 resolves this. |
| global-only | named | named keys, global term folded into each (mirrored) |
| global-only | global-only | global-only |
| global-only | empty | **global-only** (F1 fix; mirrored), potentially incomplete for unquantified measured inputs |
| mixed on either side | any | `ValueError`, no mutation |

### 6.3 Implementation requirements

- **One planner.** A single validated key planner replaces the duplicated logic in `_binary_basedata_op` and `_uncertainty_key_plan`, and is used by both in-place fast paths.
- **Validate at every entry point.** Validation runs at construction, arithmetic and combination entry points (`uncertainties` is a mutable `dict`). Mixed mappings are blocked in `PoissonUncertainties` and other uncertainty-producing modules; `combine_uncertainty_keys` rejects `propagate_to_all` as a component source. For Stage 0 only, a producer that meets a global-only mapping can create the newly named estimate by applying first-order quadrature to the original global and new σ, then **atomically** replacing the global-only mapping with the named result. This is the declared **legacy approximation**, not proof that the contributions are independent. It logs the transition and must never silently drop the global term. If no named estimator is created, the wildcard may remain the sole key on the working `signal`. Stage 1 replaces this approximate producer default with the role-dependent transition table in §7.0.
- **Do not conflate unknown with exact.** Exact arithmetic literals and explicit exact Pint quantities add zero contribution by documented contract. An empty `BaseData` mapping is unquantified. Stage 0 documents this; S1-C adds the metadata. Do not add a fabricated `0` uncertainty array to resolve the distinction.
- **Ownership.**
  - `signal_converter`/`dict_signal_converter` copy array inputs, or `__attrs_post_init__` does.
  - `_writable_full_shape_uncertainty` never returns an array the target does not own.
  - `to_units` rebinds rather than scaling arrays in place, unless ownership is guaranteed.
- **Preflight, then mutate.** Shapes, units, the key plan and uncertainty broadcasts are all checked before the first write.
- **Self-aliasing.** If `other` shares memory with `target` (`np.shares_memory` on signal or any uncertainty array), snapshot `other` first or fall back to the out-of-place path.
- **Transient memory (F9).**
  - Coerce a real or `Quantity` arithmetic literal to a 0-d *exact constant* operand with no uncertainty keys (known exact **by this arithmetic API's contract**), not full-size zero arrays. Preserve exactness metadata at the Stage 1 boundary.
  - Skip terms whose σ is absent.
  - Compute each output key with in-place ufuncs into a single fresh array.
  - Keep broadcast inputs as views.
  - Target: peak allocation ≤ (number of output keys + 1) frame-sized arrays for any binary op.
- Add `__imul__` with the same guarantees, or document that `*=` is out-of-place.

### 6.4 Acceptance criteria

- Every row of §6.2 is tested for `+ − × ÷` where supported, in both operand orders and in-place/out-of-place variants, including global-only uncertainties already on the working `signal`. Verify results with both exact literals and an explicitly unquantified `BaseData`.
- F1–F9 from `repro.py` become regression tests with the corrected expectations: F1 retains the global term; F2/F3 raise or fold deterministically; F5 leaves other objects untouched; F6 matches out-of-place; F7 leaves the signal unchanged; F9 peak ≤ 4 × 32 MiB for the stated case.
- Arithmetic on `a - a` and `a / a` is alias-safe and matches snapshot-based legacy calculations; explicitly test and document that the physical cancellation of the same uncertain variable is **not** fixed in Stage 0.
- No in-repo pipeline YAML requires changes (§3.3); external MOUSE pipeline configurations should undergo a pre-release dry-run, since in-memory module ordering can produce mixed mappings.

## 7. Stage 1 → Stage 2: semantics, sources, lineages, fits

### 7.0 Stage 1: semantic contract, shared bound and coverage

Stage 1 ships in three packages (§8). None of them produces signed, correlation-correct propagation; that is Stage 2. Before S1-A starts, the project fixes the *type* of elementary identities (Decision 6). S1-A's ComponentIds and S1-B's origin IDs are then the same objects that Stage 2 SourceIds and LineageIds will use.

**S1-A: semantics and bridging (prepare; activate together with S1-B).**

- **Semantics attribute.** `BaseData.uncertainty_semantics ∈ {"alternatives", "components"}`, default `components` for objects not created by a declaring producer (corrections, geometry, arbitrary arrays). It survives copy, slicing, unary ops and serialisation.
- **Producer transition from `propagate_to_all` (normative):** A producer introducing a named estimator into a `BaseData` whose only key is `propagate_to_all` must resolve that entry's *physical/statistical role before any mutation*. Stage 0's legacy folding is not inherited as a scientifically justified default.
  - `independent_component` with a distinct proven ComponentId: fold once in quadrature into each relevant new estimator; record membership and remove the reserved entry.
  - `alternative` of the same measurand/statistical target: preserve it as a separate named candidate or select explicitly; **never** fold as an independent source. A producer must supply a non-colliding estimator name and a target.
  - `shared_correction`: transfer the *identified input contribution* to `S` with the proper propagated sensitivity, removing its representation from the estimator dictionary. This requires source-side declarations/provenance, not merely a scalar shape or the reserved key's name.
  - `unknown` / historically folded `propagate_to_all`: keep the output marked `legacy_folded` and `bound_status=not_guaranteed`, with no silent promotion to a defensible bound; strict mode raises before mutation.
  - The implementation must test the case **where the reserved key is already on the working `signal`**, including producer insertion order, not only binary arithmetic. No transition may temporarily publish a mixed `propagate_to_all` mapping.
- **Producers of alternatives:** `PoissonUncertainties`, scatter estimators and explicit configuration.
  - When a `components` object becomes `alternatives`, existing *non-shared* components are folded into each new estimate once. They are recorded with stable ComponentIds and per-estimate membership, which prevents later duplicate folds.
  - Shared contributions go to `S` instead (S1-B). This transition is not enabled until S1-B is available; S1-A alone must not introduce silent shared-to-independent folding.
  - If the decomposition is unknown, the estimate is marked `legacy_folded`.
- **Bridging rule:** `alternatives ∘ components` → `alternatives`.
  - The components operand's *declared independent, non-broadcast* keys are combined in quadrature and folded once into each alternative.
  - Its broadcast or declared-shared keys go to `S`.
  - This resolves mixtures such as `{Poisson, sample_thickness_SEM, …}` without pretending that the shared ones are independent pixel noise.
- **Other combinations:**
  - `components ∘ components` uses today's union rules.
  - `alternatives ∘ alternatives` uses §4.3.
  - The same ComponentId on both operands is rejected in strict mode and flagged in permissive mode.
- **Combination modules:** `CombineUncertainties` on `alternatives` raises; `CombineUncertaintiesMax` on `components` raises.

**S1-B: shared bound, origin tracking and transform-safety contract.** These are specified in §7.0.1. Enable S1-A's new bridging rules **atomically** with S1-B's `S` representation, operator support and statuses. Before that joint activation, either retain clearly labeled Stage 0 behavior or hold the release. Reductions report a conditional first-order bracket only for eligible outputs; ineligible outputs receive a visible flag or a strict-mode failure.

**S1-C: input coverage and model completeness (public-release prerequisite).**

- **Input states.** Coverage metadata distinguishes `exact`, `quantified` and `unquantified`. `unquantified` propagates to dependents; arithmetic never converts it to `exact`.
- **Separate model status, declared once per pipeline.** `model_completeness ∈ {declared_complete, incomplete, not_assessed}` is a field of the pipeline configuration. Default is `not_assessed`. It is written to every output, and it is not propagated per `BaseData`.
  - Completeness is a property of the stochastic model the pipeline implements, not of individual arrays. Having `Poisson` uncertainty alone does not show that detector calibration or readout were modelled.
  - `declared_complete` is an assertion by whoever maintains the pipeline, with a free-text reference to where the model is documented. It is never inferred from a nonempty dictionary.
  - v5 proposed propagating this per object and checking it against a requirement manifest per stochastic model. That adds per-operation bookkeeping without adding information, and a manifest that nobody owns will drift. A pipeline that later needs machine-checked completeness can add a manifest then.
- **Conditional guarantees.** A valid mathematical bracket covers the *modeled* random inputs. The output must not imply physical completeness when model status is `incomplete` or `not_assessed`. Monte Carlo agreement with the same model validates implementation, not experimental adequacy.
- **Defaults by producer, not by constructor.**
  - Data loaded through an IO source is `unquantified` unless an uncertainty source is configured, in which case it is `quantified`.
  - Module-constructed values declare their coverage. Masks, pixel indices, bin edges and fit diagnostics are `exact`.
  - Deterministic factors computed from geometry inherit the coverage of their geometric inputs.
  - Arithmetic literals are `exact` by API contract.

  v3's default of `unquantified` for every `BaseData(uncertainties={})` would flag nearly every output, because masks, indices and diagnostics are built that way in the modules. The flag would then carry no information.
- **Output and activation.** Outputs depending on `unquantified` measured inputs carry a visible input-coverage flag. Model-completeness status is reported independently, including when all known inputs are quantified. **A Stage 1 sink must not publish `bound_status=guaranteed` until S1-C is active and the coverage and model statuses are present in the same output.** Before that gate, implementations may use `guaranteed` internally *for the mathematical model only*, but externally must use `model_only_unassessed` (or suppress the public bracket) and cannot claim physical-model completeness. The three packages may be implemented independently behind a feature flag; the public Stage 1 activation gate requires S1-A + S1-B + S1-C, end-to-end verified.

**Common to all three:**

- **No retroactive conversion.** Stage 1 `legacy_folded` estimates and `S` cannot be upgraded losslessly to Stage 2 sources. Stage 2 runs are recomputed from raw inputs or source-explicit saved corrections.
- **Monte Carlo harness.** It is moved here from Stage 2. `correlation_checks.py` and `stage1_bound.py` are the seeds: they already assert first-order versus Monte Carlo agreement and the Stage 1 bracket. They should be converted into pytest modules pinned to the MoDaCor revision and run with real `pint`.
- **Documentation.** Until Stage 2A, **eligible** 1D outputs of 2D-first pipelines report a *model-conditional first-order bracket*. Its lower end is independent residual statistics alone and its upper end bounds the uncertain shared response in the declared stochastic model; it may be very loose because signs and identities are not yet known. Unsupported dependency patterns, uncertain weighting/bin membership or incomplete model coverage must not be described as a guaranteed physical uncertainty interval.

**Estimator target contract.** Every producer of a selectable alternative declares its output measurand and statistic; `CombineUncertaintiesMax` requires compatible targets. `ReduceDimensionality` and `IndexedAverager` distinguish uncertainty of a reduced **intensity mean** (e.g. Poisson propagation or empirical SEM) from descriptive **Q-bin width** (STD of Q samples). The latter belongs on the Q output as spread metadata/diagnostic, not in the intensity alternative set. Existing legacy keys may be accepted with explicit configuration during migration, but a target mismatch cannot be silently reconciled.

### 7.0.1 The Stage 1 shared bound `S`

Each `BaseData` carries, next to its estimates, one unsigned array `S`. It is broadcastable to the signal, in signal units, and common to all alternatives. `S` holds an upper bound on the 2-norm, over all shared latents, of the signed responses that Stage 2 would track individually.

The residual estimates keep only contributions whose latents are elementwise-independent and unshared. The two parts are disjoint by construction, consistent with the v3 non-overlap rule.

**Propagation rules.** Here `f_A` and `f_B` are the partial derivatives of the operation, and `w` are the fixed reduction coefficients.

| ID | Situation | Residual estimates `est_k` | Shared bound `S` |
| --- | --- | --- | --- |
| B1 | Ingestion of a size-1 `BaseData` (scalar parameter) | none | `S = max_k(quadrature of components)`; a single latent may be reused anywhere |
| B2 | Operand whose *signal* broadcasts in a binary op (fewer elements than the output; full or partial broadcast) | its estimates leave `est` and `ek` | `S_eff = sqrt(max_k (est_k² + ek_k²) + S²)` |
| B2′ | Producer-declared shared contribution: thickness/transmission/μ-derived factors and geometry jitters (response magnitudes), or a declared reused measurement (its estimates) | not added; a declared measurement's estimates leave `est` | added, then propagated via B3–B6 |
| B3 | Binary op `R = f(A, B)` | `est` and `ek` each `hypot(|f_A| x_A,k, |f_B| x_B,k)`, with §4.3 key rules; `k_R = max(k_A, k_B)` over operands with nonzero `ek` | `|f_A| S_A + |f_B| S_B` |
| B4 | Unary `y = f(x)` | `|f'| est_k`, `|f'| ek_k`; `k` unchanged | `|f'| S` |
| B5 | Weighted mean / sum / integration, fixed coefficients, each input element in one output bin | `est`: `sqrt(Σ w_i² est²) / |Σ w_i|`; `ek` is consumed (B6c) | `Σ |w_i| S_i / |Σ w_i|` (or `Σ |w_i| S_i` for sums), then B6c |
| B6a | Pointwise transform, unique indexing, permutation, unit conversion, or a linear map with **fan-out 1** (each input element reaches one output element) | `est` by the first-order law (exact marginal); `ek` consumed (B6c) if the map combines elements, otherwise kept with its `k` | `S` with absolute coefficients |
| B6b | Linear map with **fan-out `f > 1`** (interpolation, smoothing, overlapping bins, resampling) or **repeated indexing** (max repeat `r`) | `est` moves into the multiplicity class: `ek ← quadrature marginal` (still exact per element), `k ← f` (or `k·r`); a pre-existing `ek` is consumed first (B6c), or for indexing merged with `k ← k·r` | `S` with absolute coefficients |
| B6c | Consumption: any operation that combines elements of `ek` (reduction, linear map) | `ek ← 0` | `S ← hypot(S_lin, √k · sqrt(Σ_i c_ji² ek_i²))` |
| B6d | Operator whose fan-out or support is unknown | — | `bound_status = not_guaranteed`; strict mode rejects |
| B7 | A module that is not `S`-aware receives `S ≠ 0` or `ek ≠ 0` | `sqrt(est_k² + ek_k² + S²)` (today's behaviour) | set to 0; mark `legacy_folded` and `bound_status = legacy` (no bracket claimed downstream); warn once |
| B8 | Output assembly | lower `σ⁻_k = hypot(est_k, ek_k)`; single-field default `σ_indep,k = sqrt(est_k² + ek_k² + T²)` | upper `σ⁺_k = sqrt(est_k² + ek_k² + S²)` |

**Multiplicity rule for B5–B6 (v6, replacing v5's promotion).** v5 found that many-to-many maps break v4's B6. Interpolation, smoothing and repeated indexing make different output elements depend on the same residual latents, so a later reduction that treats them as independent underestimates.

v5 fixed this by moving every affected residual into `S`, which treats it as fully correlated over the whole array. That is valid but loose by about the square root of the number of reduced elements. v6 keeps the residual and records how far it has spread:

- **Two residual classes.** Each `BaseData` carries the residual in two classes: `est`, whose latents reach exactly one element, and `ek`, whose latents reach at most `k` elements. `k` is one integer per object (per alternative if alternatives differ). Per element, both are exact marginals.
- **Where `k` comes from.** A fixed linear map `y_j = Σ_i c_ji x_i` with fan-out `f` (the largest number of outputs any input reaches) puts the quadrature marginal of `est` into `ek` with `k = f`. Repeated indexing multiplies `k` by the largest repeat count.
- **Elementwise operations** keep `k`. Binary ops take the larger `k`, which is safe because operand latents are disjoint under the origin rules.
- **Consumption bound.** When an operation combines elements of `ek` (a reduction or another map), Cauchy–Schwarz gives, for every output `j`:

  `Var_ek(j) = Σ_l (Σ_i c_ji J_il)² ≤ Σ_l |N_l| Σ_i c_ji² J_il² ≤ k · Σ_i c_ji² ek_i²`

  where `N_l` is the set of elements latent `l` reaches (`|N_l| ≤ k`). So `√k · quadrature` bounds that output's residual norm. It moves to `S` (B6c), combined with `hypot` because its latents are disjoint from those already in `S`.
- **Lower end.** It stays `hypot(est, ek)` while `ek` is unconsumed, and `est` afterwards. Only the overlapping part loses its lower-bound contribution.
- **Unknown fan-out** (opaque operators, data-dependent supports) is B6d: no bracket.

The counterexample still holds and is a mandatory regression gate. With two independent unit-σ inputs, interpolate as `y1 = 0.75 x1 + 0.25 x2` and `y2 = 0.25 x1 + 0.75 x2`, then average.
- The true first-order value is `sqrt(0.5) = 0.707`; v4's B6 gave 0.559.
- With `k = 2`, the upper end is `√2 · sqrt(1.25)/2 = 0.791`, the same as v5 in this two-point case.
- Repeated indexing `x[[0, 0]]` followed by a mean gives exactly 1.0, which equals the truth.
- The two rules differ as soon as more than a handful of elements are reduced (results below).

Simple slicing with unique indices and concatenation of disjoint origins are B6a. Nonlinear or data-dependent transforms need a declared first-order Jacobian and support map, otherwise B6d applies.

**Origin tracking (cheap, reliable for the cases it covers).**

- Each ingested array-valued measurement gets an origin ID per realisation (§2-IP).
- A `BaseData` carries the set of origins whose noise it depends on: through `est`, through `S`, or through a fit. Fitted parameters carry the origins of every fit input, even though their own uncertainty sits in `S` (B1).
- A binary op whose operands share an origin through `est` or `ek` on *at least one* side, or through a fit on either side, sets `bound_status = not_guaranteed` on the result and its descendants. Reductions report that status next to the bracket.
- Sharing through `S` on *both* sides is safe, because the bound already assumes full correlation there.
- An origin in `est`/`ek` on one side and in `S` on the other is not safe: the residual part is combined in quadrature with a term it is actually correlated with.

**Guarantee (conditional on a complete declared stochastic input model and certified operator support).** At first order, for outputs with `bound_status = guaranteed`:

`σ⁻_k ≤ σ_true,k ≤ σ⁺_k`

*Proof sketch.* Partition the latents into those carried in `est` (set R) and those carried in `S` (set H).

- Through B3–B6a, the part of R carried in `est` is reproduced exactly, because none of its latents reaches two elements or two recombined paths. Latents that do reach several elements are carried in `ek`, with `k` bounding how many. Their marginals stay exact, and every combination of their elements is bounded by Cauchy–Schwarz (B6c). Operators must declare their fan-out; object-level origin tracking alone cannot see interpolation-induced overlap.
- The H-part's norm is bounded by `S`:
  - the triangle inequality covers B3;
  - absolute homogeneity covers B4;
  - Minkowski's inequality over latents covers B5.
- Hence `σ_true² = est² + ek² + ‖g_H‖² ≤ est² + ek² + S²`, and `σ_true ≥ hypot(est, ek)`. The `ek` term is an exact marginal until it is consumed into `S` by B6c.
- **Independence approximation `σ_indep` (v8).** `T` is propagated alongside `S` by the same rules, with three differences: every linear sum of `S` becomes a quadrature sum, B2 and B6c add their terms in quadrature, and B6c has no `√k`. Then `σ_indep,k = sqrt(est_k² + ek_k² + T²)`.
  - Quadrature never exceeds the linear sum and `√k ≥ 1`, so by the same induction `T ≤ S`, and therefore `σ⁻_k ≤ σ_indep,k ≤ σ⁺_k`.
  - `σ_indep` is what an engine assuming independence everywhere would report under *Stage 1's* key rules. It is MoDaCor's natural single-field number, and its position inside the bracket is guaranteed.
  - It costs one array per `BaseData`, shared by all alternatives.
- **No claim is made for values produced by earlier releases** (v7 correction, retained). Earlier releases used different key rules. Example: the historical partial-overlap rule gives `SEM = 15` for `{Poisson: 10, SEM: 15} − {Poisson: 7}`, whereas Stage 1's completed SEM candidate gives `hypot(15, 7) = 16.55`, so the old value lies below `σ⁻ = σ⁺ = 16.55`. Under P0 those values are not reproduced; they remain available by running the earlier release.

**Exclusions** (the bound may fail, and the result is flagged or documented):

- **E1. Undeclared reuse of a full-shape measurement held in separate objects.** Example: each operando frame stored as its own `BaseData`, with the same background subtracted from each. Origin tracking catches reuse only through shared descendants. Mitigation: declare the measurement shared (B2′), or represent the series as one stacked `BaseData` so that the background broadcasts (B2).
- **E2. The same full-shape value reaching an output by two paths** (`a − a`, recombination with an input). Detected by origins; result `not_guaranteed`.
- **E3. Fitted parameters applied to data that entered the fit.** Detected by origins. Both directions of failure occur (table below). Stage 2C is the remedy.
- **E4. Data-dependent weights, masks or bin assignments** (as §7.1).
- **E5. Nonlinearity beyond first order.**
- **E6. Transform-induced cross-element correlation without a declared fan-out.** Interpolation, smoothing, repeated indexing, rebinning with overlapping supports and similar one-to-many or many-to-many maps defeat the bracket if their residual is kept in `est`. With a declared fan-out the multiplicity rule handles them. With unknown support the result is `not_guaranteed` (B6d). A correct per-output marginal σ is not evidence of residual independence.
- **E7. Uncovered or misspecified stochastic input model.** Even an algebraically valid bound is not a guarantee of physical uncertainty when required sources or independence assumptions are missing; report the separate model-completeness status (§7.0).

**Results** (`stage1_bound.py`, `correlation_checks.py` section C). "Reference" is exact first-order propagation, which agrees with Monte Carlo within the asserted tolerance.

| Case | Lower | MoDaCor today | Reference | Upper | Bracket |
| --- | --- | --- | --- | --- | --- |
| Example 1: thickness, mean of 100 px | 0.500 | 0.559 | 2.55 | 2.55 | holds, exact |
| F8: shared time and dark (integration-test magnitudes) | 0.3317 | 0.3318 | 0.3318 | 0.3319 | holds |
| Same structure, visible magnitudes (t ±5 %, dark ±20 %) | 4.36 | 8.14 | 4.39 | 12.3 | holds, loose (cancellation) |
| Example 2 time-average, background broadcast (B2) | 4.47 | 6.16 | 30.33 | 30.33 | holds, exact |
| Example 2 frame difference | 44.72 | 61.64 | 44.72 | 74.83 | holds, lower exact |
| Operando stack 10 × 20 px, background broadcast, pixel then time mean | 2.24 | 3.08 | 7.07 | 30.08 | holds, loose (spatial) |
| Operando, separate frame objects, undeclared (E1) | 13.78 | 13.78 | 31.62 | 13.78 | **fails** |
| Same, background declared shared (B2′) | 10.00 | 13.78 | 31.62 | 31.62 | holds, exact |
| FindScaleFactor1D, full-curve fit, mean over q (E3)¹ | 69.84 | 70.69 | 63.13 | 79.76 | **fails low**, flagged |
| FindScaleFactor1D, window fit, mean over q (E3)¹ | 11.91 | 13.34 | 21.42 | 22.94 | inside, but not guaranteed; flagged |
| 2-point overlapping interpolation then mean, v4 B6 (E6) | 0.559 | 0.559 | 0.707 | 0.559 | **fails high**, v4 rule invalid |
| Same, v6 multiplicity rule (k = 2) | 0 | 0.559 | 0.707 | 0.791 | holds |
| Repeated indexing `x[[0, 0]]` then mean (k = 2) | 0 | 0.707 | 1.000 | 1.000 | holds, exact |
| `SubtractInterpolated1D`-like: 200-point background onto 1000-point sample grid (k = 11), subtract, mean over a q window | 16.18 | 18.69 | 30.29 | 34.96 | holds; v5 promotion would give upper 103.6 |

¹ v6 correction: the fitted-parameter σ used for "MoDaCor today" and for the bound is now the median that MoDaCor reports over 400 noisy realisations (scale 0.0093, background 2.98; the true spread is 0.0097 and 2.89). v4 and v5 took it from a fit to the noise-free nominal curve. That fit's reduced χ² is 1e-27, so the reported scale σ was 3e-16. The full-curve row therefore contained no scale term at all.

**Interpolation followed by a mean** (unit-σ inputs; `stage1_bound.py`):

| Grid | k | Truth | MoDaCor today | v5 promotion upper | v6 multiplicity upper |
| --- | --- | --- | --- | --- | --- |
| 100 → 100 points (shifted) | 3 | 0.100 | 0.087 | 0.869 | 0.151 |
| 100 → 400 points | 9 | 0.100 | 0.041 | 0.812 | 0.123 |
| 60 → 600 points | 21 | 0.130 | 0.033 | 0.811 | 0.153 |

Today underestimates by up to a factor of 4 on upsampling. v5's promotion is 6–9× loose, while the multiplicity rule is within 1.2–1.5×.

**Random pipelines.**
- The v4 generator (mixed shapes, partial broadcasts, all four binary ops, `sqrt`/`log`/square, axis reductions) did not produce overlapping linear operators. That is why it missed the v5 counterexample.
- The v6 generator adds every B6 operator class: linear interpolation onto shifted grids, nearest-neighbour remapping (repeated indexing), moving-average smoothing, dense signed mixing, and disjoint pair rebinning.
- Across 76 814 checked elements (1 200 pipelines, three seeds), the bracket held everywhere. The upper/reference ratio had median 1.00, 95th percentile 1.86 and maximum 30.
- With multiplicity tracking switched off, which is the v4 rule, the same generator finds violations. A test asserts that, so the property test is now known to be sensitive to this class of error.

**Reading the table:**

- The bound is exact for a single same-sign shared source, which is the dominant 2D-first case.
- It is loose under cancellation (sample − background with a shared time) and under spatial reduction of per-pixel shared noise.
- It is loose by at most about `√k` on interpolated data, rather than by the square root of the reduced length.
- It is *never* low **under the tested eligibility assumptions**. The Stage 1 implementation must enforce fan-out declarations and origin status before making that claim.
- `σ_indep` lies inside the bracket by construction but can be far from the truth in either direction (the "MoDaCor today" column is `σ_indep` for these single-alternative cases). Values from earlier releases are not required to lie inside it.

**Memory.**

- `S` is one array per `BaseData`, shared by all alternatives, and so is `T` (v8). The multiplicity class `ek` adds one array per alternative, and only on data that has passed through a many-to-many operator. In current MoDaCor pipelines that means 1D data after `SubtractInterpolated1D` or `FindScaleFactor1D`'s interpolation. Pixel-splitting 2D regridding would add it in 2D; benchmark such pipelines before adopting one.
- While every shared contribution is multiplicative from scalars, `S = ρ·|signal|` with a scalar `ρ`. The relative form survives multiplication and division exactly (`ρ_R = ρ_A + ρ_B`).
- It is materialised at the first addition or subtraction, typically background subtraction.
- On an Eiger2 4M frame that is +34 MiB (102 → 136 MiB), the same incremental cost as the Stage 2 anchor-factored estimate in §7.4.

**Module work.**

- `S`-aware handling is needed in arithmetic, unary ops, `indexed`/`copy`/`to_units`, `ReduceDimensionality`, `Integrate1D`, `IndexedAverager`, `ConcatenateDatabundles`, `SubtractInterpolated1D`, sinks and plots. **Every** operator that can mix or duplicate elements declares its fan-out (B6a/B6b). In MoDaCor today these are `SubtractInterpolated1D` (linear: two-point stencil; nearest: repeated indexing), the interpolation in `FindScaleFactor1D`'s `prepare_scale_fit_data`, and `indexed` with advanced indices. Undeclared operators produce `not_guaranteed` (B6d).
- B2′ routing is needed in the capillary and flat-plate corrections, the geometry modules (scalar jitters) and `FindScaleFactor1D`, whose outputs also need origin inheritance.
- Everything else falls back through B7, which is today's behaviour with a warning.

**Fit weighting.** `FindScaleFactor1D` and other fits select an estimator compatible with the fit's intended noise target. `σ⁺` is not automatically a valid pointwise weight. A shared normalization uncertainty generally needs a nuisance-parameter/covariance treatment rather than being treated as independent point noise. Until Stage 2C, fitted results relying on data-shared noise carry the required `not_guaranteed` status.

### 7.1 Stage 2A: shared scalar sources

**Propagation (signed, linear):**

- `g_R = f_A g_A + f_B g_B`. In particular:
  - addition: `g_A + g_B`
  - subtraction: `g_A − g_B`
  - multiplication: `B g_A + A g_B`
  - division: `g_A/B − A g_B/B²`
- Unary: `g_y = f'(x) g_x`, with the sign kept.
- For a shared ID appearing on both sides, all algebraic paths to that **same elementary quantity** must combine before squaring. Two independent physical fluctuations must never share an ID merely because they have the same variable name or metadata path.
- **Identity contract:** producers assign/validate elementary `SourceId` and physical scope, with identity derived from the actual measurement/calibration realization, not its storage path. Declared correlated input parameters require common elementary factors or validated joint whitening; simply assigning different IDs and quadraturing is not sufficient.

**Reductions.** With **fixed, exact** coefficients `w_i` (weights, masks, integration coefficients), the response reduces as `g_y = Σ w_i g_i / Σ w_i` (or `Σ w_i g_i` for sums and integrals). The coefficients must be *identical* to those applied to the signal, including NaN/mask selection. If weights, selection masks or bin assignments depend on uncertain measurements, their derivatives/discontinuities are **not** captured by this formula; either model those dependencies explicitly or warn/reject for strict uncertainty outputs.

**Why the 2D-first pipeline is then correct.** For a scalar multiplicative source, the relative response is constant across pixels and survives azimuthal averaging unchanged. The 1D output then carries the full thickness, time, transmission and calibration uncertainty instead of `1/√N_bin` of it.

**Outputs:**

- **2D:** marginal per-pixel σ for the selected alternative (correct marginally), plus the source responses exported separately, so that downstream users who reduce or fit the 2D data can treat them as nuisance parameters.
- **1D:** the same, reduced correctly.

### 7.2 Stage 2B: shared statistical lineages (operando, empty cell, solvent)

A measured array `b` reused in several branches gets an elementary `LineageId` (including the identity of that measurement realization) at ingestion. Its selected independent per-element noise terms are **transferred**, not copied, into signed lineage responses `r_{b,k} = −σ_{b,k}` per alternative `k` (for subtraction). The corresponding variance is removed from the independent residual estimate.

**Promotion happens where the estimate is created.** A lineage is declared at the step that creates the measurement's statistical estimate, for example in `PoissonUncertainties` on the raw background frame, before anything has been folded into it. At that point the estimate *is* the lineage, so the transfer is exact and no variance budget has to be reconstructed. Promotion anywhere later in the pipeline is rejected.

**Stage 1 stand-in.** Until 2B ships, a reused measurement can be declared shared (§7.0.1, rule B2′), or the series can be held as one stacked `BaseData` so that the measurement broadcasts (B2). Either way the bound covers it conservatively. It is exact for time averages of one pixel, but loose by up to `√N_bin` under azimuthal averaging.

**No double-counting invariant:** For each selected estimator, an elementary per-pixel contribution appears in exactly one place: residual `estimates` *or* an explicit lineage. If two branches refer to the same measurement, they reference the same lineage; if they are independent acquisitions, they have distinct lineages even if file paths or array values match. Check this at ingestion, transformation and output assembly.

- **Across branches:** responses add linearly *per element*. In Example 2, frame differences cancel `b` exactly.
- **Within a reduction:** squares add, because the latents are independent per element: `V_{b,k}(q) = Σ w_i² r_{b,k,i}² / (Σw_i)²`.
- **Branch combination after reduction.** If both branches carry `b` with responses `c_A·base` and `c_B·base` (anchor-factored, §7.4) and were reduced with the *same* weight signature (same geometry, mask and binning, checked by hash), then `Cov = c_A c_B V_base(q)`. That is exact and needs one reduced array per lineage.
  - Otherwise the operation raises with an actionable message: "combine before reduction" or "use the Monte Carlo path".
- **Limitation and safety policy.** A scatter-derived alternative (`SEM` over pixels in a bin) may already include underlying lineage noise but cannot separate it without a statistical model. Mark it `absorbed_unresolved`; **do not** add the same lineage again as an independent term. A strict cross-branch operation or reduction that needs its correlation must reject such an alternative and request a resolvable `Poisson`-style model or Monte Carlo. A permissive diagnostic plot may display it with an explicit incomplete-correlation flag, but may not claim a correlation-correct total. Operando frame-to-frame statistics therefore default to a lineage-resolvable statistical alternative (or Monte Carlo).

**Operando acceptance target:** Example 2 numbers (30.33 and 44.72) within Monte Carlo tolerance.

### 7.3 Stage 2C: fitted parameters (`FindScaleFactor1D`)

Measured with MoDaCor's own `fit_scale_factor_1d` (`correlation_checks.py`). "MoDaCor today" uses the median σ that MoDaCor reports over 400 noisy realisations, a v6 correction (§7.0.1, footnote 1). v4/v5 used a fit to the noise-free nominal curve, which reports a scale σ of 3e-16.

| Case | Quantity | MoDaCor today | First order (full Jacobian) | Monte Carlo |
| --- | --- | --- | --- | --- |
| Scale + background fitted over the full working curve and applied to it | mean over q | 70.69 | 63.13 | 62.61 |
| | per point (low, mid, high q) | 6075 / 28.08 / 10.23 | 5944 / 27.48 / 9.50 | 5961 / 27.72 / 9.43 |
| | corr(scale, background) | not exported | — | −0.78 |
| Scale only, fitted in an overlap window (merging), applied to the whole curve | mean over q | 13.34 | 21.42 | 20.98 |

The error has either sign:
- The full-curve case overestimates by about 12 % for the mean and up to 8 % per point. The fit absorbs part of the curve's own noise, and the scale/background anticorrelation is discarded.
- The window case underestimates by about 38 %. A well-determined scale is treated as independent per point and then reduced.
- These conclusions match v4/v5 (12 % and 42 % there); the magnitudes changed by at most four percentage points.

**Proposal and variance partition (required before implementation):**

1. **Export the complete fitted-parameter covariance**, including scale/background cross terms, but never convert the **entire** covariance into independent Cholesky sources if the same input noise is already represented by lineages. For the parameter vector `p`, let the linearized dependence on all identified working/reference inputs be `δp = H δz + η`, with `η` independent of those inputs. Then:

   `Σ_p = H Σ_z Hᵀ + Σ_η`

   The `H Σ_z Hᵀ` part stays in the identified input lineages/sources. Only a separately justified **residual** covariance `Σ_η` may become new independent elementary parameter sources. Naively adding full `chol(Σ_p)` and the influence responses **double-counts** the input-driven parameter uncertainty.
2. **Partition only with a consistent covariance model.** `Σ_z` must include the covariance implied by reused inputs (and relevant parameter/source correlations), and `H` must be computed under the *same model and fit weighting* that produced `Σ_p`. For independent input sets, sum their `H_j Σ_j H_jᵀ` parts. Do not subtract an arbitrary influence budget from an optimizer covariance computed under incompatible assumptions.
3. **Use the unscaled covariance, and keep χ² inflation separate.** `fit_scale_factor_1d` returns `residual_variance · pinv(JᵀJ)`, the unscaled covariance multiplied by the reduced χ² (F11).
   - For a linear fixed-weight fit, the input-driven part `HΣ_zHᵀ` equals the unscaled covariance. Partitioning the *scaled* covariance would therefore give `Σ_η ≈ (χ²_red − 1)·HΣ_zHᵀ`.
   - That is negative whenever χ²_red < 1, which happens by chance in about half of fits whose noise model is right. In 400 simulated fits of the §7.3 full-curve case with MoDaCor's `fit_scale_factor_1d`, 51 % had χ²_red < 1. The PSD check in item 4 would fire routinely.
   - The partition therefore uses the unscaled covariance.
    - χ² inflation is **off by default**. A high χ² is a goodness-of-fit finding, not evidence that the discrepancy is an independent random source with covariance proportional to `HΣ_zHᵀ`. Additional covariance may be introduced only by a separately documented stochastic model (including its physical source/independence assumptions and covariance), not by automatic conversion of `max(χ²_red − 1, 0)` into an elementary source.
4. **Validate residual covariance.** Before factorization, symmetrize and check `Σ_η = Σ_p − H Σ_z Hᵀ` for positive semidefiniteness within a documented tolerance. A significant negative eigenvalue is a model/decomposition error and must raise. A singular valid residual may use a rank-revealing eigenfactorization rather than an ordinary Cholesky. If a reliable decomposition cannot be obtained, retain a documented unsupported-fit status or fall back to Monte Carlo; do not invent a residual source.
5. **Input dependence.** For *linear fixed-weight least squares* with correct Jacobian and noise model, an influence operator of the form `H = (Jᵀ W J)⁻¹ Jᵀ W` is appropriate (with the exact design/Jacobian and sign conventions of the implementation). It is **not automatically available as the correct data influence matrix** just because an optimizer returns a parameter Jacobian. In MoDaCor, include **both working and reference curves** and their shared lineages; full sensitivity may require differentiating the complete fit procedure, including interpolation and any weights that depend on data.
6. **Implementation order.** First implement and test the linear/fixed-weight, analytically tractable case. Treat Huber and other robust losses, interpolated/reference-dependent weights, bounds and data-dependent selections as separately validated first-order approximations, with Monte Carlo comparison. If they fail the specified tolerance, do not silently use the linear result.
7. **Fit alternative selection.** `_combined_sigma` must choose an explicitly configured statistically meaningful alternative; never add competing alternatives in quadrature. A numerical default is permitted only if the pipeline specifies one unambiguously and records it in provenance.
8. **Downstream fitting.** Export multiplicative sources as identifiable nuisance parameters or a covariance-aware representation, not as an unlabelled covariance matrix only (Peelle's puzzle, §5). The final corrected-curve response includes the direct dependency of the original working curve and its indirect dependency through the fitted parameters; this is essential for cancellations and covariance signs.

**Until Stage 2C:** fitted outputs inherit the origins of the fit inputs, and their use on those inputs marks results `not_guaranteed` (§7.0.1, E3). The table above shows that both directions of failure occur.

**Stage 2C acceptance principle:** the joint output covariance (or sampled means/selected points) must agree with a physically consistent reference simulation. Reproducing the optimizer's parameter diagonal is not sufficient. Keep the existing noiseless nominal-curve probes as useful stress tests, **and add representative noisy nominal data realizations**. A residual-χ²-scaled covariance evaluated on a perfectly fitted noise-free nominal curve may be nearly zero despite nonzero parameter dispersion over repeated noisy measurements; the benchmark must not confuse these distinct quantities.

### 7.4 Memory: anchor-factored responses

**Storage.** Each response is `g = Σ_j c_j·a_j` over a small set of anchors:

- `SIGNAL`: the object's own signal, which costs nothing.
- `ONE`: a broadcast 1, which costs nothing.
- Stored immutable arrays, referenced copy-on-write.

**Rules:**

- **Multiplication and division keep `SIGNAL` anchors, for any operand.**
  - For `R = A·B`, a term `c·SIGNAL_A` contributes `B·c·y_A = c·y_R`, whatever `B` is, pixel-varying or not. Likewise `c·SIGNAL_B` contributes `c·y_R`.
  - For `R = A/B`, the contributions are `c·y_R` and `−c·y_R` respectively.
  - So relative coefficients add for multiplication and subtract for division, exactly.
  - Only `ONE` and stored anchors need a new (shared, cached) anchor when multiplied by an array.
  - Points where `B = 0` are undefined for the signal itself and follow the NaN policy.
- **Exact multiplicative factors** such as solid angle or polarisation leave `SIGNAL` anchors unchanged. For stored anchors, the scaled anchor is computed once and shared by every source that references it.
- **Addition and subtraction** may re-express `SIGNAL_A` as `SIGNAL_R − SIGNAL_B` (for `R = A + B`) *when the required anchor and sign relations are valid*. Reuse existing immutable background/dark anchors when possible, but do not assume such sharing exists in every pipeline.
- **Pixel-varying derivatives** (self-absorption via μ, efficiency) introduce one stored anchor per derivative array. All sources that enter through that derivative share it, as the capillary module already does internally.
- **Reductions** reduce each anchor once. `SIGNAL` maps to the result's `SIGNAL` and `ONE` stays `ONE`. Scalar sources cost nothing to reduce.
- **Copy-on-write.** In-place operations on an object whose signal is referenced as an anchor copy first, using a version counter.
- **Optional precision.** `float32` storage is an option for anchors and estimates; arithmetic stays in `float64`.

**Budget.** For an Eiger2 4M frame (4.47 M pixels, 34 MiB per `float64` array), assume 2 alternatives, 6 scalar sources (time, transmission, thickness, flux, calibration, dark level), 2 μ-derived sources sharing one derivative, and one background lineage:

| Representation | Per frame | 100-frame operando series |
| --- | --- | --- |
| Today (signal + 2 estimates, everything folded) | 102 MiB | 10.0 GiB |
| Dense responses (one array per source and lineage) | 443 MiB | 43.3 GiB |
| Anchor-factored (signal + 2 estimates + 1 μ anchor; background and lineage bases shared) | 136 MiB | 13.4 GiB |

**Performance hypothesis, not an invariant:** The table is an illustrative budget for a favorable set of source shapes and shared anchors. Arbitrary chains of additions, nonuniform corrections, branching and reductions can create more distinct anchors; Stage 2 memory can exceed the table. Validate peak *additional* allocations, resident memory and full pipeline time on realistic Eiger2 series (including ownership/copy-on-write). Fixing transient allocations (F9) matters as much as the representation. Stage 0's allocation target likewise refers to incremental peak allocation, not total resident bytes.

### 7.5 Diagnostics and output assembly

For each output the pipeline should report:

- the selected alternative, its **measurand/statistical target**, and the selection policy; descriptive Q-bin spread is reported on the Q coordinate output and never silently selected as an intensity standard uncertainty;
- the independent residual statistical-only σ, explicitly excluding lineages and sources;
- per-source and per-lineage magnitudes, with proof-of-nonoverlap/ownership identities and any unresolved absorbed terms;
- the total shared contribution;
- the final σ;
- provenance and **input coverage status** (`exact`/`quantified`/`unquantified`), **declared-model completeness** (`declared_complete`/`incomplete`/`not_assessed`), including Stage 1 `legacy_folded` markers and whether covariance/lineage modeling is complete for this selected alternative;
- in Stage 1: `σ_indep` with its algorithm version, as the single-field default;
- in Stage 1: distinctly named `statistical_uncertainty`, `uncertainty_lower_bound`, `uncertainty_upper_bound`, `shared_uncertainty_bound`, `uncertainty_bound_status` (§7.0.1). These are bounds conditional on the declared model, **not** interchangeable 1σ estimates; sinks must not silently label `σ⁺` as an ordinary standard uncertainty.

**Stage 1 sink policy.**
- **Named fields.** Emit the named fields above. The upper bound is never written into a generic uncertainty field *as if* it were a 1σ estimate.
- **Single-field formats.** For formats with one mandatory uncertainty field (NXcanSAS `Idev`, three-column ASCII), the policy is configurable: `independence_approximation` (default), `upper_bound_as_conservative_proxy` or `strict_requires_stage2`. The chosen policy and its algorithm version (e.g. `independence_approximation_v1`) are recorded in the file's provenance, or in a consistently named side file if the format has no metadata slot; strict export fails if neither is possible.
- **Default: `independence_approximation` = `σ_indep`** (§7.0.1). It is computed from `est`, `ek` and `T`; no parallel engine exists. It lies inside the bracket, and its label says it assumes independence. The bounds and `bound_status` are written alongside wherever the format allows. It is not bit-identical to pre-Stage-1 releases (P0).
- **Strict mode** requires an explicit choice. Formats that cannot store the bracket warn in permissive mode and fail in strict mode, rather than silently discarding it.

2D and 1D outputs carry the same source IDs, so they remain mutually consistent.

### 7.6 Migration

1. Replace `BaseData.uncertainties` with `UncertaintyModel`. No read-only legacy view is kept (P0); modules and sinks move to the new API in the same release.
2. Before switching, inventory every writer. The audit found direct writes in `PoissonUncertainties`, `combine_uncertainty_keys`, `capillary_sample_container_correction` (L305), the scatter estimators and several constructors (§3.1). Each moves to an explicit API: `add_estimate`, `add_component`, `add_source`.
3. Porting guide for pipeline configurations (not a runtime compatibility layer):
   - geometry jitters → sources (scalar) or components (pixel index);
   - `*_SEM` on correction factors → sources;
   - `Poisson`/`SEM`/`STD` → estimates;
   - `propagate_to_all` → `ANY` as legacy **estimate** behavior only. Migration to a physical source requires an explicit source declaration and elementary identity, **not** merely evidence that the operand was a scalar or came from a particular path.
4. Version the serialised schema. Unknown or missing source identities are an explicit error, not a silent independence assumption. Serialized estimates, components, lineages and sources must retain unique elementary IDs and independent-versus-shared membership, so round-tripping cannot duplicate noise.
5. **No lossless upgrade of folded variance.** A Stage 1 `legacy_folded` standard deviation cannot generally be split into original elementary sources later. Stage 2 requires replay from original inputs and typed correction definitions, or source-explicit retained intermediate data. Treat such a conversion as migration requiring reprocessing, not a cast.
6. There is one engine per release; no mixed-engine runs exist.

## 8. Work packages and release gates

| Package | Scope | Gate |
| --- | --- | --- |
| **S0-A — now** | Single planner, F1/F2/F3 fixes, reserved-key guards; exact literal versus unknown measurement documented | §6.2 matrix; F1–F3 regressions; `signal` global-only case retained safely |
| **S0-B — now** | Ownership, self-aliasing, preflight, transient memory | F5–F7, F9 regressions; no partial mutation; additional peak allocations benchmarked |
| **S0-DOC — with Stage 0 release** | Update online `docs/data-model/uncertainty-propagation.md` with the **actual Stage 0** reserved-key table, global-only signal/producer cases, independence assumptions and limitations; cross-link tests/modules | Sphinx build and links pass; docs correspond to Stage 0, not unimplemented Stage 1 or 2; example output verified against Stage 0 tests |
| **S1-0 — before S1-A** | Elementary identity *type* and issuance API (Decision 6), shared by ComponentIds, origin IDs and later SourceIds/LineageIds | Type and serialisation reviewed; no Stage 1 identity is reimplemented in Stage 2 |
| **S1-A — soon (prepare)** | `uncertainty_semantics`, estimator target/statistic, bridging rule, §4.3 alternatives rules, ComponentId duplicate guard, combination-module checks | Prepared and tested behind an activation gate; §3.2 mixtures resolved **only on joint S1-A+B activation**; identical components never folded twice, incompatible targets not selected or combined |
| **S1-B — soon (joint activation with S1-A)** | Shared bound `S`; multiplicity class `ek` with `k` and operator fan-out declarations (B5/B6); origin tracking and `bound_status`; `S`-aware modules and B7 fallback; `T` and `σ_indep` (single-field default); Monte Carlo harness; MOUSE audit | `stage1_bound.py` cases and property test reproduced against **actual MoDaCor** objects, including interpolation, smoothing and repeated indexing; the v4-rule regression detected; E2/E3/B6d flagged; B7 warning and fallback; multi-alternative behaviour; origin and `k` survive transforms and serialisation; `σ⁻ ≤ σ_indep ≤ σ⁺` and `σ_indep` equals the independence-assumption engine on the property test; partial-overlap example shows an earlier-release value outside the bracket (documentation case, not a gate on `σ_indep`); every in-repo module is `S`-aware, so B7 is reached only by external modules; resident/peak-memory overhead measured (`S`, `T`, `ek`); activation gate ensures S1-A never folds shared terms without `S` |
| **S1-C — before public Stage 1 activation** | Input coverage metadata with producer-declared defaults; pipeline-level `model_completeness` field (§7.0) | Exact construction, unquantified IO, quantified inputs with `not_assessed` model status; coverage flags survive copy, transforms and sinks; public `guaranteed` bound is blocked until this gate passes |
| **S1-DOC — with public Stage 1 activation** | Publish the expanded online uncertainty guide (§11): Stage 1 named-rule and producer-transition tables, `S`/`ek`/`k`, guarantee and exclusions, two worked examples + interpolation, sink meanings, and accurately attributed prior art | Sphinx docs build and link checks; code-vs-doc numerical examples asserted; toctree entry and module cross-links verified; review by a maintainer familiar with uncertainty propagation; no Stage 2 behavior described as shipped |
| **S2-A — near future** | `UncertaintyModel`, elementary SourceIds, anchors, signed arithmetic/unary ops, fixed-weight reductions | Example 1 = 2.55; F8 dark cancels *only with identical physical source ID*; distinct exposure jitters remain independent |
| **S2-B — near future** | Lineage transfer, no-overlap validation, reduction weight-signature checks | Example 2 = 30.33 / 44.72; duplicate lineage/estimate raises; independent acquisitions do not cancel; absorbed scatter case handled explicitly |
| **S2-C — near future** | Fit covariance partition, linear/fixed-weight influence rows, robust-fit MC checks | Decomposition residual PSD; no double-count of fit-input noise; §7.3 cases within defined MC tolerance |
| **S2-D — near future** | Migration/reprocessing, serialization, diagnostics, memory benchmarks, **public documentation update** from bounds to identified signed-source/lineage propagation | 2D and 1D pipelines validated from original inputs; source IDs survive round-trip; measured peak memory reported; public guide updated and reviewed with shipped Stage 2 test cases |

## 9. Decisions and defaults for project review

**Defaults adopted for near-term work (change only by explicit project decision):**

0. **(v8) P0, no backward compatibility:** uncertainty handling may change between releases without compatibility layers, deprecation adapters, legacy views or bit-identical outputs. Release notes state the change; earlier results are reproduced with the earlier release.

1. **Mixed `propagate_to_all`:** a hard `ValueError` after Stage 0 producer entry-point fixes. An isolated wildcard can legitimately remain in the working `signal` until a named estimate is introduced. No direct mixed in-repo pipeline configuration was found; dry-run external MOUSE configurations.
2. **Stage 1 semantics default:** `components` for pre-existing/legacy constructed `BaseData`, with explicit `alternatives` producer declarations and estimator target/statistic metadata. Stage 1’s changed bridging rules are activated **only after** the `S` engine is available. No silent quadrature or `max` between mismatched estimator targets.
3. **Exactness/coverage and model status:**
   - Arithmetic literals and intentionally exact `Quantity` values are exact by contract.
   - IO-loaded data without a configured uncertainty is `unquantified`, and coverage propagates.
   - Module-constructed values declare their coverage (§7.0, S1-C).
   - Model completeness is a **separate, pipeline-level** declaration, `not_assessed` by default, written to every output and not propagated per object.
4. **Stage 1 reductions and transforms:**
   - Report `[σ⁻, σ⁺]` only for outputs whose operators declared their fan-out (B6a/B6b) and whose `bound_status = guaranteed`.
   - Unknown fan-out (B6d), reuse (E2) or fit dependency (E3) yields `not_guaranteed`, and strict-mode failure.
   - Never describe Stage 1 as correlation-correcting.
5. **Duplicate component folding:** the same ComponentId added twice to the same alternative raises by default. If there is no reliable component identity/budget, prohibit claims of lossless decomposition.

6. **Elementary identity type (needed before S1-A):** define the ID type, issuance per realisation (§2-IP), persistence and collision checks. Stage 1 uses it for ComponentIds and origins; Stage 2 adds independence/whitening policy and scope validation on top.
7. **Stage 1 output policy:** **adopt explicitly named bound fields**; `σ⁺` is never presented as an ordinary 1σ estimate.
   - Single-field formats use a configurable policy, default `independence_approximation` (`σ_indep`, §7.0.1), labelled with its algorithm version and provably inside the bracket. There is no legacy sidecar (P0).
   - `upper_bound_as_conservative_proxy` is allowed only with an eligible, model-conditional bound and a visible conservative-proxy label.
   - Strict mode requires an explicit choice; unknown coverage/status blocks public claims of guaranteed physical uncertainty.
   - Enable Stage 1's public `guaranteed` status only once S1-C's coverage reporting is installed and sinks can emit the status and model limitations.
   - **Status names** are fixed in the normative specification. Proposal: `valid_for_declared_model` / `not_guaranteed` / `legacy_folded`. Public wording should not say "guaranteed", which reads as a claim about the physics; this also replaces the interim `model_only_unassessed`.
8. **χ² inflation policy for fits (§7.3):** off by default. No automatic residual-covariance source from χ² alone; any additional variance requires an explicit physical stochastic model and validated independence assumptions.
9. **Overlap handling (B6):** the multiplicity rule (`ek` with integer `k`, consumed through the `√k` Cauchy–Schwarz bound), with v5's promotion to `S` as the fallback when the fan-out is declared but tracking is impractical. Unknown fan-out is B6d.

**Stage 2 details to finalize before implementation (not blockers for Stages 0–1):**
10. **Lineage eligibility:** require non-overlap with residual estimates; decide whether unresolved scatter-derived alternatives are blocked in strict mode (recommended) or handled by a separate Monte Carlo estimator.
11. **Fit support:** implement fixed-weight linear fit covariance partition first; robust losses and parameter-dependent weighting require explicit validation or MC fallback.
12. **Memory:** decide if `float32` anchors/estimates are acceptable only after numerical error and actual resident/additional-memory benchmarks.
13. **Downstream outputs:** whether sinks expose only the selected alternative or a clearly named set of complete alternatives; they must never silently mix model-specific and shared terms. No legacy view exists (P0).

## 10. Non-goals and safeguards

- No universal correctness claim for first-order propagation. Monte Carlo is the validation path.
- No inference of independence or correlation from key names, metadata paths, broadcasting or array dimensionality (§2-IP). Their only operational use is routing to the conservative Stage 1 bound, which is safe under any correlation.
- No dense pixel-by-pixel covariance matrices.
- No silent conversion of a shared source or lineage into independently reducible noise, or vice versa; representation transfers are variance-conserving and **non-overlapping**.
- No quadrature combination of alternatives, or of a result with its own inputs.

- No claim that `a - a` or `a / a` physically cancels in Stages 0–1; only computational alias-safety is guaranteed then.
- No claimed guaranteed bracket after a transformation that mixes or duplicates statistical latents across output elements **unless** its fan-out is declared and tracked (B6b/B6c), or a validated support-aware model preserves the dependence.
- No inference that a dispersion STD of coordinate values is the standard uncertainty of the coordinate mean, nor that an intensity STD is automatically an SEM.
- Fixed-weight/mask first-order reduction rules do not account for uncertainty in the weights, masks or bin membership unless the relevant derivative is explicitly modeled.
- No uncontrolled double-counting of fit input noise via both a lineage response and complete parameter covariance.

**Guiding principle:** Every elementary uncertain quantity has one identity and contributes **exactly once**. Preserve information until it can be combined under a defensible physical model. Stage 0 prevents silent loss; Stage 1 distinguishes alternative estimators by target and provides *conditional, explicitly labeled* bounds where operator and origin contracts make this valid; Stage 2 preserves source and lineage identity so both 2D and 1D outputs can correctly represent correlations in their declared physical model.

## 11. Required online documentation and scientific-method provenance

**Rationale and scope.** The online documentation is part of the **definition of done**, not an optional polishing task. Uncertainty propagation changes scientific interpretation, and its assumptions need to be inspectable by users and future maintainers. The public guide must give executable rules, examples and their justification. It must clearly distinguish standard GUM/linear-algebra mathematics from MoDaCor's engineering choices and approximations; it must **not** suggest that a novel `S`/`ek` implementation is itself an externally standardized method.

### 11.1 Site integration: use the established Sphinx/MyST hierarchy

- The repository already contains **`docs/data-model/uncertainty-propagation.md`**, linked in the `toctree` of `docs/data-model/index.md`. Keep this established public entry page and published URL, and **expand it into the canonical uncertainty-propagation guide**. Do not create a second divergent top-level explanation.
- If the guide becomes too long, let that page serve as an overview and add child pages beneath `docs/data-model/uncertainty/` (e.g. `rules.md`, `examples.md`, `scientific-basis.md`), included in a nested MyST/Sphinx `toctree`. Preserve stable links to the existing page and reference/module documentation.
- The documentation uses `myst_parser`, Sphinx, and `sphinxcontrib.mermaid` (verified in `docs/conf.py`), so explanatory Mermaid flow diagrams are supported without introducing a rendering dependency. Diagrams are strongly recommended for producer transitions and the split between independent estimates / shared bounds, but **must be supplementary to text tables** for accessibility and unambiguous exactness.
- Link from the main Data Model landing page, `BaseData` API docs, `PoissonUncertainties`, `CombineUncertainties`, `CombineUncertaintiesMax`, `ReduceDimensionality`, `IndexedAverager`, `Integrate1D`, and any interpolation or correction module documented as Stage 1-aware. Add changelog/release-note links for breaking Stage 0 behavior and Stage 1 status/field changes.

### 11.2 Required public content (minimum acceptance checklist)

| Guide section | Required normative/explanatory content |
| --- | --- |
| Terminology and targets | 1σ uncertainty versus variance; physical measurand and estimator target; **STD of Q positions as spread** versus **SEM of mean intensity**; alternative estimates versus independent components; exact/unquantified inputs. |
| Current shipped stage | Prominent version-specific statement: Stage 0 legacy approximation, Stage 1 conditional bracket, or Stage 2 source-aware propagation. Never describe a future stage as implemented. |
| Stage 0 compatibility | Exact binary and producer key tables; `propagate_to_all` as sole key (including on the working `signal`), empty operand, wildcard-to-Poisson transition, mixed mapping error, alias-safety versus physical correlation. |
| Stage 1 numerical model | `uncertainty_semantics`; declared producer-transition table (`independent_component`, `alternative`, `shared_correction`, `unknown`); per-estimator targets; `est`, `S`, `ek` and `k`; the B1–B8 propagation table; first-order **model-conditional** bounds and the exclusions. |
| Correlation and repeated measurements | Why common scalar normalization is not reduced by `sqrt(N)`; why reused background, `a-a`, fitting against one's own input, and interpolation of shared input noise require special handling. |
| Transparent outputs | Meaning and units of `statistical_uncertainty`, `uncertainty_lower_bound`, `uncertainty_upper_bound`, `uncertainty_bound_status`, the single-field `independence_approximation` (`σ_indep`) and its algorithm version; why an upper bound is **not** an ordinary 1σ estimate. |
| Known restrictions | Input coverage versus pipeline-declared completeness; fixed weight/mask assumptions; nonlinearity/Monte Carlo; unknown origin/fan-out; transition with unresolved wildcard; B7 module fallback; Stage 1 results not upgraded losslessly to Stage 2. |
| Source-aware future | Clear **planned / not yet implemented** subsection explaining Stage 2 SourceIds, lineage IDs, signed derivatives, cancellation and fit covariance partition; promote to a versioned live guide only when released. |
| Scientific foundations | References, exact applicability, and differences from MoDaCor policy (below). Link test-backed derivations rather than claiming all rules are proven by a single external package. |

**Minimum checked examples** (include dimensions, units, assumptions, and actual output/status):

1. **Thickness correction followed by 100-pixel mean:** 100 counts, `σ_P=10`, `t=2±0.1`, resulting statistical σ=0.5 and shared thickness=2.5; show legacy 0.559 and model-conditional Stage 1 upper 2.55, and explain why the true shared scalar does not average down.
2. **Operando repeated-background example:** 50 frames, one reused 900-count background; distinguish the 30.33 time-average uncertainty from the 44.72 frame-difference uncertainty; distinguish Stage 1 conservative bound from Stage 2 exact response/cancellation.
3. **Interpolation then mean:** two independent unit-σ measurements interpolated into two overlapping outputs; true first-order σ=0.707 versus naive 0.559; Stage 1 multiplicity-bound upper 0.791. Include repeated advanced indexing, and explain eligibility/fan-out.
4. **Estimator target:** descriptive STD of Q values in a bin does not quantify uncertainty of mean Q or mean intensity; show intensity Poisson versus SEM as potentially valid competing mean-intensity estimates *only under their respective assumptions*.
5. **Wildcard producer transition and sink semantics:** `propagate_to_all` already on `signal`, followed by `PoissonUncertainties`; distinguish valid declared Stage 1 roles and Stage 0 legacy fold; display `bound_status` for unresolved role. Explain `σ_indep`, why it lies inside the bracket, and that earlier releases' values are reproduced only by running those releases (P0).

### 11.3 Explicit justification and accurately attributed prior art

The guide should link the original source for each established idea, **say what it establishes**, and differentiate that from MoDaCor-specific engineering decisions:

- **GUM / JCGM 100:2008, §5.1–5.2; NIST TN 1297, Appendix A:** first-order sensitivity coefficients and covariance-inclusive propagation. This justifies signed responses, covariance terms and the special independent case; it does **not** mandate MoDaCor's wildcard key, `S`, or multiplicity metadata. Primary sources: [BIPM GUM JCGM 100](https://www.bipm.org/en/committees/jc/jcgm/publications), [NIST TN 1297 Appendix A](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).
- **JCGM 101:2008 and JCGM 102:2011:** propagation of probability distributions with Monte Carlo and covariance-aware multivariate measurands. MoDaCor's Monte Carlo harness validates the declared stochastic model, **not** completeness of the physical model. [JCGM publications](https://www.bipm.org/en/committees/jc/jcgm/publications).
- **Scipp's documented broadcast-variance restriction and the neutron-scattering analysis:** broadcasting uncertain values generates dependencies that a diagonal variance cannot track; Scipp can raise rather than silently understate. MoDaCor Stage 1 instead retains an explicit conservative bound because its 2D-first correction pipeline must broadcast. [Scipp computation guide](https://scipp.github.io/user-guide/computation.html), [Heybrock et al., Journal of Neutron Research](https://content.iospress.com/articles/journal-of-neutron-research/jnr220049).
- **ESSreduce:** `broadcast_with_upper_bound_variances` multiplies the variance by the volume of the new dimensions when an uncertain value is broadcast. That is the same Cauchy–Schwarz factor as MoDaCor's `k`, applied at the moment of broadcast. It is established practice for exposing a conservative bound instead of pretending broadcast data are independent. MoDaCor's difference is *when* the factor is applied: at the later combination (B6c), not at the broadcast. Per-element marginals therefore stay exact, which matters because MoDaCor publishes corrected 2D data as well as reduced 1D data. [ESSreduce uncertainty module](https://scipp.github.io/essreduce/generated/modules/ess.reduce.uncertainty.html).
- **pyFAI:** explicit `poisson` and `azimuthal` error models for azimuthal integration demonstrate the need to distinguish estimator assumptions and intended output targets. They do **not** imply that arbitrary STD and SEM tracks are interchangeable. [pyFAI uncertainty/error-model tutorial](https://pyfai.readthedocs.io/en/stable/usage/tutorial/Variance/uncertainties.html).
- **`uncertainties`, GTC, and other sensitivity-tracking libraries:** precedent for preserving elementary input identities and differentiating with respect to them, so one common source can cancel or reinforce through different paths. MoDaCor Stage 2 uses the same first-order mathematics but vectorized/factored array storage. [Python `uncertainties` guide](https://pythonhosted.org/uncertainties/), [GTC documentation](https://gtc.readthedocs.io/).
- **MoDaCor's own measured verification:** link the version-pinned `repro.py`, `correlation_checks.py`, `stage1_bound.py`, and pytest regression cases, with explicit provenance and tolerances; distinguish standalone mathematical prototypes from tests of the real library. Cite equations and test IDs alongside examples.

### 11.4 Documentation gates and ownership

- **S0-DOC:** land the initial correction to the existing uncertainty guide together with Stage 0, not before. Replace the now-inaccurate statement that mixed `propagate_to_all` is silently ignored with the shipped `ValueError` contract. Show limitations clearly. Add release-note migration guidance and link to the updated module reference.
- **S1-DOC (mandatory to activate Stage 1 publicly):** publish the complete user guide with the explicit Stage 1 rules, `S`/`ek`/`k`, origin/coverage contract, `σ_indep` single-field default, examples, warnings/statuses, and literature sources. Maintain it from the normative Stage 0/1 specification, not from a stale historical RFC. **Public documentation sign-off is a release gate**, along with coverage S1-C.
- **S2-DOC:** when Stage 2 ships, update the same guide to teach signed SourceId and LineageId models and fitting; keep a versioned historical note for Stage 0/1 outputs. Do not mix Stage 2 future claims into Stage 1 active API documentation.
- **Review and automation:** one scientific/methodology reviewer and one MoDaCor module maintainer approve the guide; build the Sphinx site in CI, check that the `toctree` includes the section and no internal links break, run the exact worked examples as tests against the matching released code, and verify the docs specify the same operator/producer tables as the implementation. Diagrams must have prose/table equivalents. Publish docs for the corresponding version/commit and record the testing revision.

### 11.5 Status of the draft public guide (`uncertainty-propagation-stage1-DRAFT.md`)

The draft is a useful start. It is to be regenerated from the normative specification, not maintained in parallel. Before S1-DOC it needs:

- **The broadcast rule (B2).** An uncertain operand that broadcasts goes to `S`. This is the most consequential Stage 1 rule, and the draft's propagation table omits it. The binary and unary rows also omit `ek`.
- **Example B's representation caveat.**
  - Frames held as one stacked array: the background broadcasts, the time-average bound is exact (30.33), and the frame-difference upper end is 74.83.
  - Frames held as separate objects: the background must be declared shared, or the bound fails (13.78 against a true 31.62).
- **Two missing examples.** §11.2 requires five; the draft lacks example 4 (estimator target: Q-bin STD versus intensity SEM) and example 5 (`propagate_to_all` producer transition and sink semantics).
- **Key table.** "Either order" on the global-only rows of the Stage 0 key table.
- **Stage 0 version.** A Stage 0-only version for S0-DOC, since the draft mixes Stage 0 and Stage 1.
- **Single-field default.** Its legacy paragraph is replaced by the `σ_indep` single-field default (§7.5).
- **Links.** The ESSreduce text states the same-factor relationship (§11.3), and all links are checked at S1-DOC. The pyFAI tutorial and the ESSreduce function were verified on 2026-11-01; the Scipp computation-guide URL was not reachable from the review environment.

## 12. Future normative specification and historical record

Keep this versioned RFC as the **design-review and evidence record** (audit findings, counterexamples, empirical results, rationale, alternative decisions and change history). Once the Stage 0/1 API and release gates are agreed—**preferably before Stage 0/1 code is merged**—derive a shorter, clean *normative implementation specification* with only: scope and terminology; data contracts/invariants; operator tables and status rules; public APIs and migration; acceptance tests; and explicitly deferred Stage 2 interfaces. Maintain a separate brief `DECISIONS.md` or GitHub issue/ADR links for unresolved choices and rationale. Do **not** delete the RFC history or its numerical experiments; keep v4–v8 and scripts in version control. **v8:** the §0 checklist (including D1–D3) was agreed on 2026-11-01; the normative specification is written next, and further changes are made there rather than in this RFC.

The clean document should not become a second independent source of truth: acceptance criteria and behavior are maintained in the normative spec after approval, while the RFC remains traceable evidence. Stage 2 can receive its own implementation-specific RFC when scheduled, using the settled identity and non-overlap contracts as constraints.

## Appendix A: reproduction

**Scripts**

| Script | Content | Self-checking |
| --- | --- | --- |
| `repro.py` | Audit findings F1–F10 | No; prints current behaviour of `56efd4a` |
| `correlation_checks.py` | Example 2 (operando), §7.3 fit cases, and section C applying the Stage 1 bound to both. **v6:** fitted-parameter σ taken as the median reported over 400 noisy fits, with the degenerate nominal-fit σ printed for comparison | Yes, for first-order versus Monte Carlo (4 standard errors of the sample standard deviation plus 0.5 %) and for the bracket in the eligible cases. Values under "MoDaCor" document the audited revision and are not asserted. |
| `stage1_bound.py` | Standalone mathematical prototype of §7.0.1: exact first order, the independence-assumption engine, and the Stage 1 bound. v6 added the multiplicity class `ek`/`k`, fixed linear maps with fan-out, an interpolation helper, v5's promotion for comparison, a `SubtractInterpolated1D`-like case, and a property test covering all B6 operator classes. **v8:** the `T` array and `σ_indep`, asserted inside the bracket and equal to the independence engine on every checked element | Yes; 16 tests under pytest, or run as a script that prints the §7.0.1 tables. Two optional MoDaCor cross-checks (the "today" engine and `remap_coefficients_1d`) are skipped when MoDaCor is not importable. It does not implement `BaseData` origin/status, multiple alternatives or B7 module integration. |

**Execution.** The v6 scripts were run against the same clone of `56efd4a` and stand-in as v4: 16 of 16 tests passed with both cross-checks active, and `correlation_checks.py` passed all assertions. The v5 bundle's own runs skipped the MoDaCor cross-checks. **v4 audit execution:** the three v4 scripts were run against a clone of `56efd4a` in a sandbox where PyPI was unreachable. `pint` was therefore replaced by a minimal stand-in whose unit conversion factors are all 1. This affects neither key handling nor propagation arithmetic: the stand-in reproduces the RFC's 5.59, and the "today" engine matches MoDaCor to 1e-12. Monte Carlo seeds are fixed (`default_rng(1)`, `default_rng(7)`; property-test seeds 2026, 7 and 11).

**Before using these as release gates:**
- Re-run with real `pint`.
- Pin the revision.
- Move the assertions into the project's pytest suite.

**Test additions still to write in MoDaCor itself (not established by the standalone prototype):**
- Same stochastic inputs interpolated to overlapping output points and subsequently averaged; overlapping rebinning/smoothing; unique versus duplicate advanced indexing; fan-out declaration, multiplicity tracking and consumption (B6a–B6c); operator without support metadata must fail strict mode.
- Multiple competing estimators with compatible/incompatible output targets (including Q-bin STD versus intensity SEM), and `S` being a common bound across alternatives.
- Actual origin tracking and `bound_status` through `indexed`, interpolation, arithmetic, reductions, copy, and serialization; B7 fallback and strict-mode failure.
- Simultaneous activation of S1-A and S1-B; no intermediate release that silently folds a shared component into independent noise.
- Model coverage/completeness states, including `quantified` Poisson with an explicitly unmodeled required calibration source; downstream sink behavior and labeled bound outputs.
- Fitted-parameter benchmarks on noisy nominal data realizations, not only the perfect-fit nominal stress case. (`correlation_checks.py` does this for the reported σ in v6; the MoDaCor test suite should too.)
- `ANY + empty` in all operand orders and in-place paths.
- Empty measured input versus exact literal.
- Producer transitions with `ANY`: independently declared component versus alternative versus physical shared correction versus unresolved role, including wildcard already in the working `signal`; no transient mixed mapping.
- Earlier-release values need not fall inside the Stage 1 bracket: document partial-overlap `{Poisson:10, SEM:15} - {Poisson:7}`, with Stage 1 completed `SEM = 16.55` versus the earlier `SEM = 15`.
- `σ_indep` (from `T`) lies inside `[σ⁻, σ⁺]` and equals the independence-assumption engine through arithmetic, broadcasting, interpolation, repeated indexing, reductions and export, with multiple alternatives (`stage1_bound.py` covers the single-alternative case).
- Public `guaranteed` bracket cannot be emitted before S1-C coverage and model-status fields are supported; test incomplete/unquantified output labeling.
- Sphinx navigation/build, numerical docs examples, module cross-links, scientific reference links and documentation-to-code parity before each release.
- Duplicate ComponentId fold.
- Independent versus reused SourceId and LineageId.
- No double-count after lineage promotion.
- Fit-covariance partition, including the unscaled-covariance and residual-PSD checks.
- Strict reduction with unresolved scatter.
- `S`-aware module coverage (B7 fallback triggered and logged).
- 2D/1D end-to-end reproduction and serialisation round-trip.

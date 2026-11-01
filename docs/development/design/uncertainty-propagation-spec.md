# MoDaCor uncertainty propagation: normative specification, Stages 0 and 1

| | |
| --- | --- |
| **Version** | 1.0-draft, 2026-11-01 |
| **Status** | Normative once merged. Defines required behaviour for Stage 0 and Stage 1. Not yet implemented. |
| **Basis** | RFC v8, agreed 2026-11-01 (§0 checklist P0, A1–A4, B1–B9, C1–C4, D1–D3), plus clarification **SC-1** (estimator selection at the shared-bound boundary, §6.1) |
| **Reference implementation** | `stage1_bound.py`, a standalone mathematical prototype of §6. Its numbers are the expected values in §11. |
| **Change control** | Changes to Stage 0/1 behaviour are made in this document. The RFC remains the evidence and rationale record and is not revised further. Spec-level decisions that the RFC left open are listed in Appendix B. |

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT** and **MAY** are used as in RFC 2119/8174.

All uncertainties are absolute one-standard-deviation (1σ) values in the units of their signal, broadcastable to the signal shape, unless explicitly marked as a descriptive spread.

---

## 1. Scope

**In scope:**
- the `BaseData` uncertainty representation;
- binary, in-place and unary arithmetic;
- `copy`, `indexed` and unit conversion;
- reductions and linear maps;
- uncertainty-producing modules;
- `CombineUncertainties` and `CombineUncertaintiesMax`;
- sinks;
- pipeline configuration keys;
- statuses;
- acceptance tests;
- documentation gates.

**Out of scope:** Stage 2, meaning signed source responses, statistical lineages and fit-covariance partitioning. §12 lists the constraints Stage 1 must respect so that Stage 2 is not blocked; it introduces no Stage 2 behaviour.

## 2. Principles

| ID | Principle |
| --- | --- |
| **P0** | **No backward compatibility.** Uncertainty handling may change between releases without compatibility layers, deprecation adapters, legacy views or bit-identical outputs. Release notes state each change. Earlier results are reproduced by running the earlier release. |
| **P1** | **No silent loss.** No operation silently discards a nonzero uncertainty contribution. |
| **P2** | **Identity per realisation.** An elementary identity names a realisation of an uncertain quantity (this thickness measurement, this background acquisition). It never names a storage path, key name, shape or broadcast pattern. Paths and broadcasting MAY trigger conservative routing (§6.1); they MUST NOT establish independence or common identity. |
| **P3** | **Non-overlap.** Each elementary contribution is held in exactly one representation. In Stage 1 that is one of `est`, `ek` or `S`/`T`. Moving a contribution between representations removes it from the source representation. |
| **P4** | **Conditional claims only.** Stage 1 bounds are first-order and conditional on the declared stochastic model and the eligibility rules (§6.8). Nothing in Stage 0 or 1 is described as correlation-correct. |

## 3. Terminology

| Term | Definition |
| --- | --- |
| **Estimate** | A named σ array stored on a `BaseData`. |
| **Estimator target** | The pair `(measurand, statistic)` an estimate describes. `measurand` is a free-text quantity name (e.g. `intensity`, `Q`); `statistic ∈ {standard_uncertainty, descriptive_spread}`. |
| **Alternative estimates** | Two or more estimates on one `BaseData` with the same target and `statistic = standard_uncertainty`, produced by competing statistical models (e.g. `Poisson`, `SEM`). They are compared or selected, never added in quadrature. |
| **Independent component** | A contribution known to be uncorrelated with everything else in the estimate it joins. It is folded in quadrature exactly once. |
| **Descriptive spread** | A dispersion of values (e.g. the STD of Q values in a bin, or the STD of intensities). It is not a standard uncertainty and is never an alternative. |
| **Residual `est`** | The part of an estimate whose elementary latents reach exactly one element and are not shared. |
| **Multiplicity class `ek`, `k`** | The residual part whose latents reach up to `k` elements, created by many-to-many operators (§6.5). |
| **Shared bound `S`** | An unsigned array bounding the 2-norm of all shared contributions, propagated linearly. |
| **Independence approximation `T`** | The same contributions as `S`, propagated in quadrature, with no `√k`. |
| **Fan-out** | The largest number of output elements that any single input element contributes to in an operator. |
| **Origin** | The elementary identity of an ingested array-valued measurement whose noise a `BaseData` depends on. |
| **Bracket** | `[σ⁻, σ⁺]`, the first-order bounds on the standard uncertainty under the declared model (§6.7). |
| **Strict / permissive mode** | Pipeline-level behaviour for ineligible situations: strict raises, permissive labels and warns (§9). |

---

## 4. Stage 0 (correctness hotfix)

Stage 0 keeps the representation `BaseData.uncertainties: dict[str, ndarray]` and the independent-operand propagation formulas. It ships on its own, before Stage 1.

### 4.1 Invariants

| ID | Requirement |
| --- | --- |
| I0.1 | Every uncertainty array MUST be absolute, nonnegative, in signal units, and broadcastable to the signal shape. NaN handling follows the existing arithmetic policy. |
| I0.2 | The reserved key `propagate_to_all` MUST be either the sole key or absent. |
| I0.3 | No operation MAY silently discard a nonzero uncertainty (P1). |
| I0.4 | An empty mapping means *no quantified contribution supplied*, not *exact*. Stage 0 arithmetic treats absent terms as zero, and the documentation MUST say so. |
| I0.5 | In-place and out-of-place forms of an operation MUST produce identical values, units, keys and errors. |
| I0.6 | Invalid input MUST raise before any mutation of signal, units or uncertainty arrays. |
| I0.7 | A `BaseData` owns its uncertainty arrays after construction. No operation MAY mutate an array it does not exclusively own. |
| I0.8 | Self-aliased operands (`a op= a`, or operands that share memory) MUST give the result of the same operation on snapshots. This is computational alias-safety only; Stage 0 does not model the physical correlation in `a − a`. |
| I0.9 | Unary operations (`log`, `exp`, trigonometric functions, and so on) MUST NOT mutate their input, including its units. A required unit conversion MUST be applied as an **inline factor**. One shared helper MUST return the converted signal *and* every converted uncertainty array together, and every unary operation MUST obtain its inputs through it, so that no operation scales the signal but forgets the uncertainties. No copy of the input is made. |

### 4.2 Reserved key `propagate_to_all`

| ID | Requirement |
| --- | --- |
| R0.1 | A mapping containing `propagate_to_all` together with any other key is **mixed**. A mixed mapping MUST raise `ValueError` before any mutation. |
| R0.2 | Validation MUST run at construction, at every binary, in-place and unary operation, in `indexed`, `copy` and `to_units`, in `combine_uncertainty_keys`, and in every uncertainty-producing module. Constructor-only validation is insufficient, because the mapping is mutable. |
| R0.3 | `combine_uncertainty_keys` MUST reject `propagate_to_all` as a source key and MUST NOT create a mixed mapping. |
| R0.4 | **Producer transition (Stage 0 only).** A producer adding one or more named estimates `X₁…Xₙ` in one call to a `BaseData` whose only key is `propagate_to_all` MUST fold the **same original** global term into **each** new estimate in quadrature: `σ_Xᵢ ← hypot(σ_Xᵢ,new, σ_global)`. This is the same as the binary rule G∘N (§4.3), where the global term contributes to every named key. All folded estimates MUST be computed first and then installed by replacing the mapping with `{X₁: σ_X₁, …, Xₙ: σ_Xₙ}` in one assignment, with `propagate_to_all` removed; no intermediate or mixed mapping is ever visible. One INFO log per call lists the folded keys as a *legacy fold*. A later producer call that adds a key to an already-named mapping performs no fold, because the global term no longer exists separately. That key therefore lacks the global contribution; this is a documented Stage 0 limitation, resolved in Stage 1 (§6.9). |
| R0.5 | A global-only mapping MAY remain on the working signal until a named estimate is introduced. |

### 4.3 Binary key resolution

Each operand's mapping falls into one category:

| Category | Mapping |
| --- | --- |
| **E** | empty |
| **N** | one or more keys, none of them `propagate_to_all` |
| **G** | exactly `{propagate_to_all}` |
| **M** | mixed; invalid |

| Left | Right | Result keys and behaviour |
| --- | --- | --- |
| E | E | E |
| N | E (either order) | N, propagated from the N operand |
| G | E (either order) | **G**, propagated from the G operand (fixes the audited loss F1) |
| G | N (either order) | the N keys; the G term contributes to each key |
| G | G | G |
| N | N, identical key sets | pairwise |
| N | N, key sets differ | union; a missing term contributes zero; log at DEBUG |
| M | any (either order) | `ValueError` before mutation |

Formulas are first-order and assume independent operands: `σ² = (f_A σ_A)² + (f_B σ_B)²`, with unit conversion factors applied.

### 4.4 Implementation requirements

| ID | Requirement |
| --- | --- |
| S0.1 | One key planner implements §4.3 and is used by the generic binary path and by every in-place fast path. |
| S0.2 | Constructors and `dict_signal_converter` MUST copy array inputs. `copy()` MUST deep-copy uncertainty arrays. `to_units` MUST rebind arrays rather than scale them in place, unless ownership is guaranteed. |
| S0.3 | In-place operations MUST complete shape, unit, key-plan and broadcast checks before the first write (I0.6). |
| S0.4 | If `other` shares memory with `target` (`np.shares_memory` on the signal or any uncertainty array), the operation MUST snapshot `other` first or use the out-of-place path (I0.8). |
| S0.5 | `__imul__` MUST be implemented in place, with the same guarantees as `__iadd__`, `__isub__` and `__itruediv__` (S0.2–S0.4, I0.5–I0.8). It SHOULD follow the existing in-place division path. |
| S0.6 | A Python real or Pint quantity operand MUST be coerced to a 0-d exact operand with **no** uncertainty keys. Full-size zero-uncertainty arrays MUST NOT be allocated, and terms with absent σ MUST be skipped. |
| S0.7 | A binary operation SHOULD allocate at most (number of output keys + 1) frame-sized arrays of transient memory. |

### 4.5 Stage 0 tests

The tests are listed in §11.1.

---

## 5. Stage 1 data model

### 5.1 Per-`BaseData` uncertainty state

| Field | Type | Meaning |
| --- | --- | --- |
| `uncertainty_semantics` | `"alternatives" \| "components"` | Default `"components"` for objects not created by a declaring producer (correction factors, geometry, arbitrary arrays). |
| `estimates` | `dict[str, ndarray]` | Under `alternatives`: the residual `est_k` per alternative `k`. Under `components`: the component keys. The reserved name `propagate_to_all` remains the model-agnostic single estimate (§4.2). |
| `estimate_targets` | `dict[str, EstimatorTarget]` | Required for every estimate under `alternatives`. |
| `spreads` | `dict[str, ndarray]` | Descriptive spreads with their targets. They are never entered into brackets, `S`/`T` or selection. |
| `ek` | `dict[str, ndarray]` | Multiplicity class per alternative; absent means zero. |
| `k` | `int ≥ 1` | One multiplicity per `BaseData`, the maximum over alternatives. |
| `S`, `T` | `ndarray \| None` | Shared bound and independence approximation. Each is a single array shared by all alternatives; `None` means zero. An implementation MAY store `S = ρ·|signal|` with a scalar `ρ` while every shared contribution is multiplicative. |
| `origins` | `dict[ElementaryId, set[{"residual", "shared", "fit"}]]` | Origins the object depends on, and through which channel. |
| `bound_status` | `valid_for_declared_model \| not_guaranteed \| legacy_folded` | Combination is worst-of: `legacy_folded` > `not_guaranteed` > `valid_for_declared_model`. |
| `envelope_used` | `bool` | True once any envelope entry (§6.1, SC-1 permissive fallback) has entered this object's `S`/`T` history. It is sticky. |
| `coverage` | `exact \| quantified \| unquantified` | Input coverage (§7). |
| `provenance` | list of records | Component folds, selections, producer transitions, B7 events and fills (§8.3). |

All fields MUST survive `copy`, `indexed`, unit conversion, unary operations and serialisation.

### 5.2 Estimator targets

- **Declaration.** Every producer of an estimate under `alternatives` MUST declare `EstimatorTarget(measurand, statistic)`.
  - `PoissonUncertainties`: `(signal measurand, standard_uncertainty)`.
  - Scatter SEM of a reduced intensity: `(intensity, standard_uncertainty)`.
  - Scatter STD of intensities or of Q values: `descriptive_spread`, stored in `spreads`.
- **Eligibility.** An estimate is *eligible* as an alternative of a `BaseData` only if its `statistic` is `standard_uncertainty` and its `measurand` equals that object's measurand.
- **Compatible targets.** `CombineUncertaintiesMax` and output selection MUST reject inputs with incompatible targets.

### 5.3 Elementary identities

`ElementaryId = (kind ∈ {component, origin, source, lineage}, uid, label, created_by)`. `uid` is a UUID4 string, `label` is human-readable, and `created_by` is the step ID.

- Issuance:
  - An origin ID is issued **per load event** of an array-valued measurement.
  - A component ID is issued per declared independent component.
  - A source ID is issued per declared shared contribution.
  - None is ever derived from a path (P2).
- The kinds `source` and `lineage` are reserved for Stage 2. Stage 1 issues `origin` and `component`, and MAY issue `source` for declared shared contributions as provenance only.
- IDs MUST be serialised and MUST round-trip unchanged.

---

## 6. Stage 1 operator rules

### 6.1 Entry into `S`/`T` and estimator selection (rules B1, B2, B2′, and SC-1)

A contribution **enters `S`/`T`** in three cases:

| Rule | Case |
| --- | --- |
| **B1** | Ingestion of a size-1 `BaseData` (a scalar parameter). |
| **B2** | An operand whose *signal* broadcasts in a binary operation (fewer elements than the output; full or partial broadcast). |
| **B2′** | A producer-declared shared contribution: response magnitudes from thickness, transmission or μ-derived factors and from geometry jitters, or the estimates of a measurement declared as reused (§6.8, E1). |

The entering σ (`σ_entry`) is determined as follows:

1. **Components.** If the operand's semantics is `components`, `σ_entry` is the quadrature sum of its declared independent component keys. No selection takes place.
2. **Eligible estimates.** If the semantics is `alternatives`, the eligible estimates are those with a matching standard-uncertainty target (§5.2). Descriptive spreads are never eligible.
3. **Sole eligible estimate.** If exactly one estimate is eligible, it is selected; the provenance source is `sole_eligible`.
4. **Two or more eligible estimates.** An **explicit selection is required**, taken from the first of:
   1. the step-level configuration `shared_estimate: {<operand role>: <estimate name>}`;
   2. the pipeline-level `uncertainty.shared_estimate_selection: {<measurand>: <estimate name>}`.

   The selected estimate MUST exist and MUST be eligible; otherwise the operation raises `ValueError` in both modes.
5. **No selection.**
   - In **strict mode** the operation MUST raise before mutation.
   - In **permissive mode** the envelope MAY be used: `σ_entry` is the element-wise maximum of the eligible estimates. The receiving object gets `envelope_used = true`, and a warning is logged. An envelope entry MUST NOT be labelled as a candidate-specific independence approximation (§8.2).
   - An envelope entry does not change `bound_status`, because `S` then bounds every eligible choice.
6. **Multiplicity class.** For a selected alternative `k*`, `σ_entry = hypot(est_k*, ek_k*)`.
7. **Entering.**
   - The operand's residual leaves `est`/`ek`.
   - `S_eff = sqrt(σ_entry² + S_op²)` and `T_eff = sqrt(σ_entry² + T_op²)`.
   - For B2′ response magnitudes, `σ_entry` is the declared magnitude.
8. **Provenance.** Every entry records the operand's origins, the available and eligible estimates with their targets, the selected estimate (or `None`), and the source (`explicit_step`, `explicit_pipeline`, `sole_eligible`, `components`, `envelope`).

The selection at the boundary is final for that contribution. Selecting the receiving object's own alternative later does not change it.

### 6.2 Binary operations (B3)

**Semantics combination** (applied after §6.1 for any broadcast operand):

| Left | Right | Rule |
| --- | --- | --- |
| alternatives | alternatives | Table below |
| alternatives | components (either order, non-broadcast) | The components operand's declared independent keys are combined in quadrature and folded **once** into each alternative. ComponentIds are recorded per estimate. A ComponentId already present in an estimate raises `ValueError`. Result: `alternatives`. |
| components | components | Stage 0 rules (§4.3). Result: `components`. |

**Alternatives rules:**

| Left estimates | Right estimates | Result |
| --- | --- | --- |
| ∅ | ∅ | ∅ |
| `K` | ∅ (either order) | `K` |
| `propagate_to_all` | ∅ (either order) | `propagate_to_all` |
| `propagate_to_all` | `K` (either order) | `K`; the wildcard term contributes to each |
| `propagate_to_all` | `propagate_to_all` | `propagate_to_all` |
| `K` | `K` (identical sets) | pairwise |
| `K` | `{k'}`, a single estimate (either order) | `K`; `k'` fills every alternative; `filled_from` is recorded; strict mode MAY reject via `uncertainty.reject_fill: true` |
| `K₁` | `K₂`, two or more each, sets differ | `ValueError`, unless the step configures an explicit `fill` mapping |
| mixed wildcard | any | `ValueError` before mutation |

**Propagation,** with `f_A` and `f_B` the partial derivatives:

- `est_R,k = hypot(|f_A| est_A,k, |f_B| est_B,k)`;
- `ek_R,k` likewise;
- `k_R = max(k_A, k_B)` over the operands with nonzero `ek`;
- `S_R = |f_A| S_A + |f_B| S_B`;
- `T_R = hypot(|f_A| T_A, |f_B| T_B)`;
- origins are the union of both operands' origins; status follows §6.8.

### 6.3 Unary operations (B4)

`est_k ← |f'| est_k`, `ek_k ← |f'| ek_k`, `S ← |f'| S`, `T ← |f'| T`. `k` is unchanged, and the input is not mutated (I0.9).

### 6.4 Reductions (B5)

This applies to a weighted mean, sum or integral with **fixed** coefficients `w_i` (weights, masks and NaN omission), where each input element contributes to at most one output element:

- `est_k ← sqrt(Σ w_i² est_k,i²) / |Σ w_i|`;
- `S_lin = Σ |w_i| S_i / |Σ w_i|` and `T_q = sqrt(Σ w_i² T_i²) / |Σ w_i|`;
- for sums and integrals, these formulas and the `est` formula above apply without the denominator `|Σ w_i|`;
- **B6c consumption** of `ek`, if present: `S ← hypot(S_lin, √k · q_k)` and `T ← hypot(T_q, q_k)`, where `q_k = sqrt(Σ w_i² ek_k,i²) / |Σ w_i|`;
- afterwards `ek ← 0` and `k ← 1`.

Further requirements:

- **Scatter estimators** create new alternatives (SEM) or spreads (STD) with declared targets. Their name collision policy is unchanged (default: error).
- **Data-dependent weights, masks or bin membership** set `bound_status = not_guaranteed` (E4), unless the operator declares the dependence handled.

### 6.5 Linear maps and indexing (B6)

Every operator that can combine, duplicate or redistribute elements MUST declare its fan-out `f` (an integer) or `None` (unknown).

| Rule | Operator | `est` | `ek` | `S` / `T` |
| --- | --- | --- | --- | --- |
| **B6a** | Pointwise transform, unique indexing, permutation, unit conversion, linear map with `f = 1` | first-order law (exact marginal) | consumed (B6c) if the map combines elements, otherwise kept with `k` | `|C|·S`; `T` in quadrature |
| **B6b** | Linear map with `f > 1` (interpolation, smoothing, overlapping bins, resampling) | moves into `ek` as the quadrature marginal; `est ← 0`; `k ← f` | consumed first (B6c) | `|C|·S`; `T` in quadrature |
| **B6b-idx** | Repeated indexing with maximum repeat `r` | `ek ← hypot(est, ek)`; `est ← 0` | `k ← k·r` | indexed |
| **B6c** | Consumption, whenever elements of `ek` are combined | — | `ek ← 0` | `S ← hypot(S_lin, √k · quad(ek))`; `T ← hypot(T_q, quad(ek))` |
| **B6d** | Fan-out `None` | — | — | `bound_status ← not_guaranteed`; strict mode raises |

**Operators in the current code base** that MUST declare fan-out:

| Operator | Fan-out |
| --- | --- |
| `SubtractInterpolated1D`, linear mode | computed from the remap coefficients |
| `SubtractInterpolated1D`, nearest mode | repeated indexing (B6b-idx) |
| The interpolation in `FindScaleFactor1D.prepare_scale_fit_data` | computed |
| `BaseData.indexed` with advanced indices | repeat count |
| `ConcatenateDatabundles` | 1; operands must have disjoint origins, otherwise E2 |
| `ReduceDimensionality`, `IndexedAverager`, `Integrate1D` | B5, `f = 1` |

### 6.6 Modules without Stage 1 support (B7)

A module that is not Stage 1-aware and receives a `BaseData` with `S`, `T` or `ek` nonzero:

- **Permissive mode:** folds them into each estimate (`sqrt(est² + ek² + S²)`), sets `S = T = ek = 0`, sets `bound_status = legacy_folded`, and warns once per step.
- **Strict mode:** raises.

Every in-repo module MUST be Stage 1-aware at public activation, so B7 is reached only by external modules.

### 6.7 Output assembly (B8)

For each alternative `k`:

| Quantity | Formula |
| --- | --- |
| Lower bound | `σ⁻_k = hypot(est_k, ek_k)` |
| Independence approximation | `σ_indep,k = sqrt(est_k² + ek_k² + T²)` |
| Upper bound | `σ⁺_k = sqrt(est_k² + ek_k² + S²)` |

The normative claim, made only when `bound_status = valid_for_declared_model`, is that at first order `σ⁻_k ≤ σ_true,k ≤ σ⁺_k` and `σ⁻_k ≤ σ_indep,k ≤ σ⁺_k`, where `σ_true` is the standard uncertainty under the declared stochastic model, with the estimates selected per §6.1.

### 6.8 Origin tracking, exclusions and status

**Origin tracking:**

- Origins propagate as a union through every operation.
- A fitted parameter carries the origins of every fit input with channel `fit`.
- A binary operation sets `bound_status = not_guaranteed` on its result when its operands share an origin through `residual` (`est`/`ek`) on at least one side, or through `fit` on either side.
- Sharing only through `shared` on both sides is permitted.

**Exclusions:**

| ID | Situation | Required behaviour |
| --- | --- | --- |
| E1 | Reuse of a full-shape measurement held in separate objects | Not detectable. Users MUST declare the measurement shared (`declare_shared`, §9.2) or stack the series so the measurement broadcasts. The documentation MUST show this case. |
| E2 | The same value reaching an output by two paths (`a − a`, recombination with an input) | `not_guaranteed` |
| E3 | Fitted parameters applied to data that entered the fit | `not_guaranteed` |
| E4 | Data-dependent weights, masks or bins | `not_guaranteed` |
| E5 | Nonlinearity beyond first order | documented; no status change |
| E6 | Many-to-many operator with unknown fan-out | B6d |
| E7 | Uncovered or misspecified stochastic model | reported through coverage and model status (§7); no bound status change |

In strict mode, any result that becomes `not_guaranteed` or `legacy_folded` raises at the operation that caused it.

### 6.9 Producer transitions from `propagate_to_all` (Stage 1)

A producer introducing a named estimate into a `BaseData` whose only key is `propagate_to_all` MUST resolve the wildcard's declared role (`propagate_to_all_role`, §9.2) before any mutation:

| Role | Action |
| --- | --- |
| `independent_component` | Fold once into each new estimate with a new ComponentId, and remove the wildcard. |
| `alternative` | Keep it as a separately named alternative; a non-colliding name and a target are required. No quadrature. |
| `shared_correction` | Transfer it to `S`/`T` per §6.1 B2′ and remove it from the estimates. |
| undeclared | Permissive: legacy fold (R0.4) with `bound_status = legacy_folded`. Strict: raise before mutation. |

No transition may expose a mixed mapping.

### 6.10 Combination modules

- `CombineUncertainties` (quadrature) constructs an alternative from **declared independent components**. Applied to `alternatives`, it raises.
- `CombineUncertaintiesMax` selects among alternatives with compatible targets. Applied to `components`, or to incompatible targets, it raises.
- Neither module may combine `propagate_to_all` or create a mixed mapping.

---

## 7. Coverage and model completeness (S1-C)

| ID | Requirement |
| --- | --- |
| C7.1 | Input coverage is `exact`, `quantified` or `unquantified`. Arithmetic literals and explicit exact Pint quantities are `exact`. |
| C7.2 | Data loaded through an IO source is `unquantified` unless an uncertainty source is configured for it, in which case it is `quantified`. |
| C7.3 | Module-constructed values declare their coverage. Masks, pixel indices, bin edges and fit diagnostics are `exact`. Deterministic factors computed from geometry inherit their inputs' coverage. |
| C7.4 | Coverage propagates as worst-of (`unquantified` > `quantified` > `exact`). Arithmetic never converts `unquantified` to `exact`. |
| C7.5 | Model completeness is **one pipeline-level field**, `uncertainty.model_completeness ∈ {not_assessed, incomplete, declared_complete}`, default `not_assessed`. `declared_complete` requires `uncertainty.model_reference` (free text naming where the model is documented). The field is written to every output and is not propagated per object. |

---

## 8. Outputs and sinks

### 8.1 Named fields

Sinks that can store auxiliary fields MUST write:

| Field | Content |
| --- | --- |
| `statistical_uncertainty` | `σ⁻` of the selected alternative |
| `uncertainty_lower_bound` | `σ⁻` |
| `uncertainty_upper_bound` | `σ⁺` |
| `shared_uncertainty_bound` | `S` |
| `independence_approximation` | `σ_indep` |
| `uncertainty_bound_status` | §5.1 values |
| `uncertainty_estimator` | the selected alternative's name and target |
| `uncertainty_coverage` | §7 |
| `uncertainty_model_completeness` | §7 |

`σ⁺` MUST NOT be written or described as an ordinary 1σ estimate.

### 8.2 Single-field formats

Formats with one mandatory uncertainty field (NXcanSAS `Idev`, three-column ASCII) use `uncertainty.single_field_policy`:

| Policy | Value written | Label |
| --- | --- | --- |
| `independence_approximation` **(default)** | `σ_indep` | `independence_approximation_v1`; if `envelope_used`, `envelope_approximation_v1` |
| `upper_bound_as_conservative_proxy` | `σ⁺` | `upper_bound_conservative_proxy_v1`; allowed only with `bound_status = valid_for_declared_model` |
| `strict_requires_stage2` | — | export fails |

The label and policy MUST be written to the file's provenance, or to a consistently named side file if the format has no metadata slot; strict export fails if neither is possible. `σ_indep` is not bit-identical to pre-Stage-1 releases (P0).

### 8.3 Provenance record

Each output carries:

- the selected alternative and the selection policy;
- every shared-entry record (§6.1, item 8);
- component folds with their ComponentIds;
- `filled_from` records;
- producer transitions;
- B7 events;
- origins, with labels;
- the configuration of `uncertainty.*`;
- the MoDaCor version and commit.

### 8.4 Public status gate

A sink MUST NOT write `valid_for_declared_model` publicly unless S1-C is active and the coverage and model-completeness fields are present in the same output (§10).

---

## 9. Configuration

### 9.1 Pipeline level

```yaml
uncertainty:
  mode: permissive                       # permissive | strict
  model_completeness: not_assessed       # not_assessed | incomplete | declared_complete
  model_reference: null                  # required if declared_complete
  shared_estimate_selection:             # SC-1: measurand -> estimate name
    intensity: Poisson
  single_field_policy: independence_approximation   # | upper_bound_as_conservative_proxy | strict_requires_stage2
  reject_fill: false                     # strict-mode option for the single-estimate fill rule (§6.2)
```

**Mode semantics:**
- **Strict** raises, before mutation, on:
  - mixed mappings;
  - an ambiguous selection (SC-1);
  - an unknown fan-out;
  - `not_guaranteed` or `legacy_folded` results;
  - an undeclared `propagate_to_all` role;
  - B7;
  - fills, when `reject_fill` is set.
- **Permissive** labels each of these and warns once per step.

### 9.2 Step and data level

| Key or API | Where | Meaning |
| --- | --- | --- |
| `shared_estimate: {<operand role>: <name>}` | any step with uncertain operands | SC-1 selection for that step (precedence over the pipeline level) |
| `propagate_to_all_role` | producers | `independent_component \| alternative \| shared_correction` (§6.9) |
| `<operand>_uncertainties_roles: {<key>: component \| shared}` | steps taking `*_uncertainties_sources` | declares B2′ shared contributions or independent components; keys of a size-1 operand default to `shared` |
| `declare_shared: true` / `BaseData.declare_shared(label)` | IO and steps producing reused measurements | E1 mitigation: the estimates enter `S`/`T` via §6.1 at declaration |
| `fill: {<missing estimate>: <source estimate>}` | binary steps | explicit fill for differing alternative sets (§6.2) |
| fan-out declaration | operator implementations (code, not configuration) | §6.5 |

---

## 10. Activation and release gates

| Package | Contents | Gate |
| --- | --- | --- |
| **S0** + **S0-DOC** | §4; documentation of the actual Stage 0 rules in `docs/data-model/uncertainty-propagation.md` | §11.1 passes; Sphinx build and link check pass; documentation describes Stage 0 only |
| **S1-0** | `ElementaryId` type, issuance and serialisation (§5.3) | reviewed before S1-A starts |
| **S1-A** | semantics, targets, bridging, alternatives rules, component deduplication, combination modules, producer transitions | behind feature flag `uncertainty.stage1` |
| **S1-B** | `S`, `T`, `ek`/`k`, fan-out declarations, origins and status, B7, SC-1 selection, sinks | behind the same flag; **activated together with S1-A** |
| **S1-C** | coverage, model completeness | required before public activation |
| **S1-DOC** | public guide regenerated from this specification | required for public activation |

**Public Stage 1 activation** requires S1-A, S1-B, S1-C and S1-DOC, verified end to end. Before activation, the flag MAY be enabled internally. S1-A MUST NOT be enabled without S1-B, so that shared terms are never folded as independent.

---

## 11. Acceptance tests

Numbers are first-order expected values from the reference implementation. Every Stage 1 test MUST be run against **real `BaseData` objects and modules**, not only against the prototype.

### 11.1 Stage 0

| ID | Test | Expected |
| --- | --- | --- |
| T0-01 | §4.3 table: all rows × `+ − × ÷` × both operand orders × in-place and out-of-place | as tabulated; I0.5 holds |
| T0-02 | `a / t`, with `a` having no uncertainty and `t = {propagate_to_all: σ}` | result `{propagate_to_all: …}`, not `{}` (F1) |
| T0-03 | Mixed mapping, in-place and out-of-place, at construction and in `combine_uncertainty_keys` | `ValueError`; target unchanged (F2, F3) |
| T0-04 | `PoissonUncertainties` on `{propagate_to_all: 5}`, signal 100, then `/ 2` | `{Poisson: 5.59}`; INFO log; no transient mixed mapping |
| T0-04b | A producer adding two estimates (σ 10 and 20) in one call to `{propagate_to_all: 5}` | `{X₁: hypot(10, 5) = 11.18, X₂: hypot(20, 5) = 20.62}`; installed in one assignment; `propagate_to_all` absent; one INFO log; a subsequent producer adding `X₃` to the result performs no fold |
| T0-05 | `{Poisson: 10, SEM: 15} − {Poisson: 7}` | Poisson 12.21, SEM 15.0 (union rule; documented Stage 0 limitation) |
| T0-06 | Two objects built from one array; `p /= 3` | the other object unchanged (F5) |
| T0-07 | `a /= a` with `[4, 9] ± [2, 3]` | equals `a / a`: `[0.707, 0.471]` (F6) |
| T0-08 | Failing `-=` (uncertainty broadcast error) | signal unchanged (F7) |
| T0-09 | `frame * 2.0`, 2048² frame, 3 keys | peak transient allocation ≤ 4 × 32 MiB (F9) |
| T0-10 | `log`, `exp`, `sin` on a `BaseData` whose units need conversion (percent → dimensionless, degree → radian) | input object unchanged, including units; output signal *and* uncertainties equal those obtained by converting a copy first (I0.9) |
| T0-11 | `a − a` (alias-safe) | equals the snapshot computation; documented as not physical |
| T0-12 | `a *= b` for scalar, array and self-aliased (`a *= a`) operands | in place (same object identity, other references see the change); results equal `a * b`; failed operation leaves `a` unchanged (S0.5) |

### 11.2 Stage 1

**Shared bound and independence approximation:**

| ID | Test | Expected (`σ⁻` / `σ_indep` / `σ⁺`; first-order truth) |
| --- | --- | --- |
| T1-01 | 100 px, `I = 100 ± 10`, `/ (t = 2.00 ± 0.10)`, mean | 0.500 / 0.559 / 2.55; truth 2.55 |
| T1-02 | `(s/t − d) − (b/t − d)`, `s = 1000`, `b = 900` (Poisson), `t = 10 ± 0.5`, `d = 5 ± 1`, shared | 4.36 / 8.14 / 12.3; truth 4.39 |
| T1-03 | Operando stack 10 × 20 px, background `(1, P)` broadcast; pixel mean then time mean | 2.24 / 3.08 / 30.08; truth 7.07 |
| T1-04 | Same pixel, separate frame objects; background with `declare_shared` | upper 31.62 = truth 31.62. Without the declaration: documented E1 failure (13.78) |
| T1-05 | Example 2: 50 frames of 1000 minus one background of 900, stacked | time mean: upper 30.33 = truth. Frame difference: lower 44.72 = truth, upper 74.83 |

**Many-to-many operators:**

| ID | Test | Expected (`σ⁻` / `σ_indep` / `σ⁺`; first-order truth) |
| --- | --- | --- |
| T1-06 | `y1 = 0.75 x1 + 0.25 x2`, `y2 = 0.25 x1 + 0.75 x2`, unit σ, then mean | 0 / 0.559 / 0.791; truth 0.707 |
| T1-07 | `x[[0, 0]]` then mean, unit σ | 0 / 0.707 / 1.000; truth 1.000 |
| T1-08 | Background on 200 log-spaced points interpolated to 1000 (`k = 11`), subtracted from sample, mean over the q window 0.05–0.08 | 16.18 / 18.69 / 34.96; truth 30.29 |
| T1-09 | Interpolation then mean: 100→100 (`k = 3`), 100→400 (`k = 9`), 60→600 (`k = 21`) | upper 0.151 / 0.123 / 0.153; truth 0.100 / 0.100 / 0.130 |
| T1-10 | Property test: ≥ 1 200 random pipelines, all B3–B6 operator classes, 3 seeds | `σ⁻ ≤ truth ≤ σ⁺` and `σ⁻ ≤ σ_indep ≤ σ⁺` on every element |
| T1-11 | The same property test with multiplicity tracking disabled | MUST detect a violation |

**Status, origins and fits:**

| ID | Test | Expected |
| --- | --- | --- |
| T1-12 | `a − a`; recombination with an input; an origin in `est` on one side and in `S` on the other | `not_guaranteed`; strict raises |
| T1-13 | `FindScaleFactor1D` scale/background applied to its own working curve | `not_guaranteed` (E3) |

**SC-1 selection:**

| ID | Test | Expected |
| --- | --- | --- |
| T1-14 | Signal 100 px `1000 ± 30`, minus a broadcast scalar with `{Poisson: 1, SEM: 10}` (same target), then mean; Poisson selected | 3.000 / 3.002 / 3.162; truth 3.162. Provenance records available `[Poisson, SEM]`, selected `Poisson`, source `explicit_*` |
| T1-15 | Same, SEM selected | 3.000 / 3.162 / 10.44; truth 10.44 |
| T1-16 | Same, no selection | strict: `ValueError` before mutation. Permissive: envelope (σ = 10); `envelope_used = true`; single-field label `envelope_approximation_v1`; warning |
| T1-17 | Selection of an estimate with target `descriptive_spread`, or of an absent name | `ValueError` in both modes. A spread is never eligible: `{Poisson, STD(spread)}` selects Poisson as `sole_eligible` |

**Semantics, producers and modules:**

| ID | Test | Expected |
| --- | --- | --- |
| T1-18 | `{Poisson: 10, SEM: 15} − {Poisson: 7}` under `alternatives` | SEM = 16.55, with `filled_from` recorded; with `reject_fill` in strict mode, raises |
| T1-19 | Two estimates differing with ≥ 2 keys each, no `fill` | `ValueError` |
| T1-20 | The same ComponentId folded twice into one estimate | `ValueError` |
| T1-21 | Producer transitions: all four roles of §6.9, including the wildcard already on the working signal | as tabulated; never a mixed mapping |
| T1-22 | External non-aware module receiving `S ≠ 0` | permissive: `legacy_folded` and a warning; strict: raises |
| T1-23 | `CombineUncertainties` on alternatives; `CombineUncertaintiesMax` on components or on mismatched targets; Q-bin STD | raise, raise; the STD is stored in `spreads` |

**Coverage, sinks, serialisation and validation:**

| ID | Test | Expected |
| --- | --- | --- |
| T1-24 | Coverage: mask `exact`, IO data `unquantified`, configured source `quantified`; propagation worst-of | as §7 |
| T1-25 | Sinks: named fields; single-field default writes `σ_indep` with its label; public `valid_for_declared_model` blocked without S1-C fields | as §8 |
| T1-26 | Serialisation round trip of `S`, `T`, `ek`, `k`, origins, IDs, status, coverage and provenance | identical |
| T1-27 | Memory benchmark on a representative Eiger2 4M pipeline | `S`, `T` and `ek` overhead reported; target ≤ +2 frame-sized arrays per `BaseData` beyond the estimates |
| T1-28 | Monte Carlo agreement for T1-05 and the fit cases | within 4 standard errors of the sample std, plus 0.5 % |

---

## 12. Constraints reserved for Stage 2 (non-normative for Stage 1 behaviour)

Stage 1 MUST NOT preclude the following:

- `ElementaryId` (§5.3) is the identity type for Stage 2 SourceIds and LineageIds.
- Stage 2 replaces `S`, `T` and `ek` with signed source and lineage responses; no Stage 1 structure is retained for compatibility (P0).
- Lineages will be declared where the estimate is created (e.g. `PoissonUncertainties` on a raw background frame). Stage 1 producers SHOULD therefore already accept a `declare_shared` request at that point.
- Fit outputs will export the full parameter covariance partitioned on the **unscaled** covariance, with no automatic χ² inflation.

## 13. Documentation requirements

| Gate | Requirement |
| --- | --- |
| **S0-DOC** | Update `docs/data-model/uncertainty-propagation.md` with §4 exactly: the key table in both operand orders, the producer legacy fold, the `ValueError` contract, unquantified ≠ exact, and alias safety ≠ physical correlation. |
| **S1-DOC** | Regenerate the public guide from this specification (§§3, 5–9). Include worked examples T1-01, T1-05 (with the E1 representation caveat, T1-04), T1-06/T1-07, T1-14–T1-16, the estimator-target example (T1-23) and the producer transition example (T1-21), each with dimensions, units and status. Cite the GUM / NIST TN 1297, JCGM 101/102, Scipp and Heybrock et al., ESSreduce (same `√k` factor applied at broadcast), pyFAI error models, and `uncertainties`/GTC, stating what each establishes and where MoDaCor's choices differ. |
| **CI** | Build the Sphinx site and check links and the toctree. Run the documentation's worked examples as tests against the released code. Check that the documentation's operator tables match this specification. |

---

## Appendix A. Traceability to RFC v8 §0

| RFC item | Specification |
| --- | --- |
| P0 | §2 P0; §8.2 |
| A1–A4 | §4.2 R0.1–R0.2; §4.3; §4.4 S0.3–S0.4, S0.7 |
| B1 | §5.1, §5.2, §6.2, §6.10 |
| B2 | §10 (joint activation) |
| B3 | §6.1–§6.7 |
| B4 | §6.5 |
| B5 | §6.8 |
| B6, B7 | §7 |
| B8 | §6.7, §8 |
| B9 | §5.3, §10 S1-0 |
| C1–C4 | §2 P2–P3, §12 |
| D1–D3 | §10, §13 |
| SC-1 (clarification) | §6.1 items 2–8; §8.2 envelope label; §9 `shared_estimate*`; T1-14–T1-17 |

## Appendix B. Decisions made in this specification (not fixed by the RFC)

1. **SC-1, estimator selection at the shared-bound boundary** (§6.1). It also covers B1 and B2′, not only B2, because all three enter the same single-array `S`/`T`. A unique eligible estimate is selected implicitly; two or more require an explicit selection. Envelope is a labelled permissive fallback only.
2. **Status names:** `valid_for_declared_model`, `not_guaranteed`, `legacy_folded`. There is no interim status; the activation gate (§10) prevents early public emission.
3. **Default mode `permissive`.** MoDaCor is in beta, and permissive produces labelled outputs while pipelines acquire declarations. `strict` is recommended for publication runs.
4. **One `k` per `BaseData`** (maximum over alternatives). This is conservative and simpler than one `k` per alternative.
5. **Two statistic values** (`standard_uncertainty`, `descriptive_spread`) plus a separate `spreads` container, instead of a richer statistic vocabulary.
6. **Stage 0 additions from the audit's "other observations"** (decided 2026-11-01): unary operations do not mutate their input, using an inline conversion factor through one shared signal-and-uncertainty helper (I0.9); `__imul__` is implemented in place (S0.5).
7. **Configuration key names** in §9. These are proposals; final spelling may follow MoDaCor's existing configuration conventions, but semantics are as specified.

## Appendix C. Reference implementation

`stage1_bound.py` (19 tests) implements §6 for a single alternative, including SC-1 (`select_for_shared`). It does not implement origins, status, multiple alternatives or modules; those are covered by §11 tests against MoDaCor itself.

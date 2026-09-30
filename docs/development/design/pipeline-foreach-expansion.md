# Pipeline `for_each` schema expansion

Status: initial implementation and grouped graph rendering complete

## Initial implementation result

The first implementation compiles non-nested `step_blocks` with typed
whole-value parameters and local prerequisites into the existing flat DAG. It
retains origin metadata per generated node, enforces an optional expanded-step
limit, preserves child-level partial reruns, and stores authored and expanded
YAML/spec provenance through both ordinary and chunked HDF sinks.

The I22 USAXS example is the acceptance case. Its authored pipeline decreased
from 1,255 to 496 lines while still expanding to 123 nodes with the same module
counts. An end-to-end run against sample scans 978497--978500 and background
scans 977724--977727 reproduced the previously stored pooled signal, pooled Q,
and transmission scalar exactly.

The expanded spec carries stable block, item, and local-step origin metadata.
The DOT and Mermaid renderers use it to group the complete execution graph into
blocks and item lanes, with an explicit option to retain the flat view. DOT
also requests matching local stages at the same rank. Rendering remains
strictly presentational and does not change scheduler nodes or dependencies.

## Decision summary

MoDaCor should not add a runtime `MapStep` and does not need to replace its
`graphlib.TopologicalSorter` scheduler. Repetition should be expanded while the
pipeline document is loaded, before `ProcessStep` objects and the execution DAG
are constructed.

The useful primitive is a repeated **step block**, not a separate mapped-step
feature. A block contains one or more ordinary steps and an explicit mapping of
item names to parameters. A one-step block covers the original mapped-step use
case; a multi-step block covers statements such as “for both sample and
background, perform x, y, and z.” All expanded children remain ordinary,
individually addressable `ProcessStep` nodes.

This is worth implementing if it remains a small, declarative compilation
layer. It stops being worthwhile if it grows into a Python/Jinja expression
language, runtime looping system, or second scheduler.

## Findings from the current implementation

`Pipeline.from_yaml()` currently performs three relevant operations directly:

1. parse the `steps` mapping;
2. instantiate and validate one `ProcessStep` per entry;
3. translate `requires_steps` into a flat `node -> prerequisites` mapping.

`Pipeline.create_scheduler()` then passes that flat mapping to the standard
library `TopologicalSorter`. Partial reruns, tracing, timing, error reporting,
serialization, and dependency contracts all operate on the instantiated child
steps and their step ids.

Consequently, schema expansion is compatible with the current scheduler. The
scheduler does not need to know that some nodes came from a repeated block. In
fact, keeping that information out of the execution semantics is desirable:
every expanded step retains its own configuration validation, dependency
contract, trace events, duration, failure location, and partial-rerun boundary.

The current DOT and Mermaid methods are renderers over `Pipeline.to_spec()`, not
features of `graphlib`. Grouping and lane layout should therefore be expressed
as optional metadata in the exported spec. They are independent of execution.

The server stores the submitted YAML and reconstructs a `Pipeline` from it for
planning and execution. Deterministic expansion therefore gives the dry-run,
full-run, and partial-run paths the same child ids. Server sinks already have
paths for both submitted YAML and an execution spec. The command-line path can
currently call `Pipeline.to_yaml()` after loading, so it will need an explicit
way to retain the authored compact source if identical provenance behavior is
required there.

## Evidence from the I22 USAXS pipeline

The initial I22 USAXS pipeline expands to 123 execution steps. Some important
repeated regions are:

| Repeated operation | Instances | Steps per instance | Expanded nodes |
| --- | ---: | ---: | ---: |
| Diode load, metadata load, mask, and time normalization | 8 | 5 | 40 |
| I0 load, mask, apply mask, and time normalization | 4 | 4 | 16 |
| I0 normalization of a diode readout | 8 | 1 | 8 |
| Beam-centre determination and propagation | 4 physical scan pairs | several | 8 initially |
| Background subtraction and negative-wing preparation | 4 sample readouts | 6 | 24 |
| Relative response scaling | 3 non-anchor readouts | 2 | 6 |

The execution graph should still contain these nodes. The avoidable burden is
authoring and reviewing their nearly identical YAML definitions. Repeated
blocks also supply the lane/stage metadata needed to render the eight diode
readouts and four monitor readouts consistently.

## Alternatives considered

| Alternative | Advantage | Problem | Recommendation |
| --- | --- | --- | --- |
| Keep all steps explicit | No new schema | The current file is difficult to review and keep internally consistent | Retain as a fully supported fallback and canonical expanded form |
| YAML anchors and merge keys | No MoDaCor code | They reuse mappings, but cannot safely generate ids, local dependencies, or parameterized source paths | Do not use as the pipeline abstraction |
| Runtime `MapStep` wrapper | One visible scheduler node | Hides child failures and traces, coarsens partial reruns, unions dependency contracts, and encourages reuse of stateful step instances | Reject |
| Schema-level single-step maps | Moderate YAML reduction | Does not naturally express repeated chains or subgraphs and creates a second construct once blocks are added | Generalize directly to repeated blocks |
| Python/Jinja-generated YAML | Extremely flexible | Arbitrary execution or opaque string templating, poor validation locations, and weak reproducibility | Reject for the public schema |
| A Python pipeline-builder API | Natural loops and functions | Less portable through server APIs and harder to preserve as data/provenance | May be an advanced API later, using the same compiler output |
| NetworkX | Rich graph queries and attributes | It does not define macro/schema expansion; scheduling needs are already met | Not needed for this feature |
| Prefect, Dask, Luigi, or similar workflow engines | Native mapping/subflows in some systems | Large execution-model change; poor fit with MoDaCor's shared `ProcessingData`, module registry, and current partial reruns | Not justified |

Graphviz clusters, Mermaid subgraphs, or a more capable browser renderer may
improve presentation, but they do not replace the schema expander or scheduler.

## Recommended public model

Use one top-level construct, provisionally named `step_blocks`, with an explicit
`for_each` table and a `steps` template. For example:

```yaml
name: I22 USAXS correction pipeline

step_blocks:
  prepare_diode:
    for_each:
      SLF:
        processing_key: SLF
        signal_location: sample::/entry1/low_gain_front/signal
        signal_units_location: sample::/entry1/low_gain_front/signal@units
        time_location: sample::/entry1/low_gain_front/count_time
        time_units_location: sample::/entry1/low_gain_front/count_time@units
      SLR:
        processing_key: SLR
        signal_location: sample::/entry1/low_gain_rear/signal
        signal_units_location: sample::/entry1/low_gain_rear/signal@units
        time_location: sample::/entry1/low_gain_rear/count_time
        time_units_location: sample::/entry1/low_gain_rear/count_time@units

    steps:
      load_signal:
        module: AppendProcessingData
        configuration:
          processing_key: "${processing_key}"
          databundle_output_key: signal
          signal_location: "${signal_location}"
          units_location: "${signal_units_location}"
          rank_of_data: 1

      dynamic_mask:
        module: ThresholdMask
        requires_steps: [.load_signal]
        configuration:
          with_processing_keys: ["${processing_key}"]
          source_basedata_key: signal
          target_mask_key: Mask
          lower_bound: 0
          mask_mode: outside

      normalize_time:
        module: Divide
        requires_steps: [.dynamic_mask]
        configuration:
          with_processing_keys: ["${processing_key}"]
          divisor_source: "${time_location}"
          divisor_units_source: "${time_units_location}"

steps:
  later_operation:
    module: SomeModule
    requires_steps:
      - prepare_diode.SLF.normalize_time
      - prepare_diode.SLR.normalize_time
```

This is the implemented version-1 syntax. Its deliberately small reference and
substitution grammar is fixed by the rules below and by the schema-expansion
tests.

### Expansion rules

The recommended initial rules are deliberately limited:

- Each `for_each` key is a stable item id and each value is a parameter
  mapping.
- The expanded id is deterministic:
  `<block_id>.<item_id>.<local_step_id>`.
- A prerequisite beginning with `.` refers to a step in the same block
  instance. An unprefixed prerequisite is an absolute expanded or ordinary
  step id.
- A scalar consisting entirely of `${parameter}` is replaced with the native
  parameter value, retaining booleans, numbers, mappings, and lists. Version 1
  should not evaluate expressions.
- Missing parameters, unknown local prerequisites, duplicate expanded ids, and
  collisions with ordinary step ids are load-time errors. Errors name the
  block, item, local step, and field path.
- Expansion is linear in `number of items * number of local steps`. A runtime
  policy limit on expanded child count prevents a small submitted document
  from creating an unreasonably large plan.
- Block nesting, conditional steps, Cartesian products, and recursive template
  use are out of scope for version 1. Exceptional lanes can be placed in a
  separate block or written explicitly.
- Ordinary `steps` remain valid and can depend on expanded steps, while
  expanded steps can depend on ordinary steps.

Only whole-value substitution is required initially. Embedded string
interpolation is convenient for titles, but full source locations and titles
can instead be item parameters. Deferring interpolation keeps values typed and
error messages precise.

### Repeating a multi-step sample/background operation

The same block mechanism handles a larger repeated unit:

```yaml
step_blocks:
  prepare_acquisition:
    for_each:
      sample:
        input_key: merged_sample
        output_key: normalized_sample
      background:
        input_key: merged_background
        output_key: normalized_background
    steps:
      x:
        module: FirstOperation
        configuration:
          with_processing_keys: ["${input_key}"]
      y:
        module: SecondOperation
        requires_steps: [.x]
      z:
        module: ThirdOperation
        requires_steps: [.y]
```

It expands to six ordinary nodes. Therefore the earlier single-step mapping
proposal would not itself solve “for both background and sample, do x, y, z,”
but the same schema-expansion engine can solve both if the public primitive is
a block whose body may contain one or more steps.

Cross-instance aggregation remains explicit. For example, a transmission step
outside the block can depend on both `prepare_acquisition.sample.z` and
`prepare_acquisition.background.z`. This keeps scientifically important joins
visible rather than hiding them in the repetition construct.

## Intelligibility for pipeline authors

The compact form is likely easier to understand when all of the following are
provided:

- meaningful block, item, and local-step names rather than positional indices;
- an explicit parameter table, with no implicit Cartesian products;
- local dependencies written next to the local steps;
- a command/API operation that validates and displays the expanded ids and
  dependencies without running the pipeline;
- errors reported against both the authored location and expanded child id;
- documentation showing compact and expanded forms side by side;
- graph views that can group by block and item while still exposing every
  physical child step.

The main cognitive cost is that authors must understand two views: concise
source and expanded execution graph. That cost is acceptable only if expansion
is deterministic and readily inspectable. `Pipeline.to_yaml()` currently
serializes instantiated nodes, so it should continue to produce the canonical
expanded `steps` form. The original authored document must be retained
separately for editing and provenance; reconstructing a block template from a
flat graph is ambiguous and should not be attempted.

## Graph and provenance representation

The runtime graph remains flat. Expanded nodes should additionally carry
origin metadata such as:

```json
{
  "block": "prepare_diode",
  "item": "SLF",
  "local_step": "normalize_time"
}
```

`Pipeline.to_spec()` can expose this metadata without changing its `nodes` and
`edges` contract. Renderers may then:

- show every expanded node;
- group nodes by item as processing lanes;
- align identical local stages across lanes;
- optionally present a collapsed block summary.

True interactive collapse belongs in a UI. DOT clusters and Mermaid subgraphs
can provide useful static grouping, but neither should affect scheduler nodes.
Renderer node identifiers should be generated independently of step ids so
that sanitizing punctuation cannot create collisions.

For reproducibility, server and sink provenance should retain both the authored
description and the expanded execution description. In the HDF processing
sink, the preferred layout is:

```text
/processing/pipeline/<run-id>/
    authored/
        yaml       # exact submitted YAML text
        spec       # normalized compact document: ordinary steps + step blocks
    expanded/
        yaml       # canonical executable YAML containing ordinary steps only
        spec       # flat execution graph from Pipeline.to_spec()
```

`authored` is preferable to `input`: it describes the semantic role and remains
accurate whether the pipeline eventually enters through YAML, an API object, or
a graph editor. `expanded` is preferable to `unfolded` because it matches the
compiler operation and generated-id terminology.

The two objects called `spec` have related but distinct schemas. The authored
spec is the normalized, JSON-serializable compiler input and remains suitable
for editing or re-expansion. The expanded spec is the executable node/edge
graph, including block/item/local-step origin metadata. A collapsed graph view
does not need a fifth stored representation: it can be derived from the
expanded spec's origin metadata, while the authored spec preserves the actual
compact description.

The current HDF layout stores `yaml` and `spec` directly below the run group.
Today those already have mixed semantics: `yaml` is submitted text while
`spec` is the runtime graph. No production reader in the repository depends on
those paths, so the new schema should replace them rather than retain aliases
or duplicate datasets. The affected sink tests should be updated to assert the
four explicit paths. Both ordinary and chunked HDF sinks should call one shared
provenance writer so their layouts cannot diverge.

The run group should also identify the compact-schema/compiler version and may
store stable hashes of the authored and expanded representations. This makes it
possible to prove which expanded DAG was executed without discarding the more
readable source supplied by the user.

Partial-rerun selection continues to operate on expanded ids and exact child
dependency contracts. Selecting a block or item in a UI is a convenience that
resolves to a set of child ids before execution.

## Implemented sequence

1. **Freeze the compact schema.** Add schema examples and failure examples;
   decide the final names, exact placeholder form, local-reference syntax, and
   whether whole-value substitution alone is sufficient for the first release.
2. **Introduce a pure document compiler.** Parse a loaded mapping into an
   expanded ordinary `steps` mapping plus origin metadata. Keep this separate
   from `Pipeline`, the module registry, and execution. Give the returned
   representation an explicit type rather than passing loosely related
   dictionaries between parser stages.
3. **Validate deterministically.** Reject malformed blocks, missing parameters,
   unknown local references, invalid ids, and all collisions before any module
   is instantiated. Preserve item order only for authoring/rendering; do not
   make execution correctness depend on it.
4. **Integrate at the YAML boundary.** `Pipeline.from_yaml()` invokes the
   compiler, then follows its existing instantiation and graph-validation path.
   `from_spec()` and programmatically constructed pipelines need no expansion.
5. **Carry origin metadata.** Associate origin records with expanded nodes and
   expose them from `to_spec()`. Keep the scheduler graph unchanged.
6. **Add inspection and provenance.** Provide an expansion/validation API (and
   optionally CLI command), continue exporting expanded YAML, and persist the
   authored and expanded YAML/spec pairs under explicit HDF groups. Route the
   ordinary and chunked sinks through a shared provenance representation and
   writer. Apply an expanded-child limit in server policy before module
   instantiation.
7. **Add grouped rendering.** Stable group/lane/stage metadata is present in
   the spec. DOT clusters and Mermaid subgraphs now show block and item lanes;
   the flat representation remains available explicitly.
8. **Migrate the USAXS example.** Express diode preparation, I0 preparation,
   repeated normalization, paired centering, and final lane preparation as
   blocks where their contracts really are identical. Leave joins and
   scientifically exceptional paths explicit.
9. **Compare equivalence.** Assert that the compact USAXS document expands to
   the same child modules, configurations, dependency edges, dependency
   contracts, partial-rerun sets, and processing result as its explicit form.

## Acceptance criteria

- Existing explicit YAML files load unchanged.
- A one-step block supports loaders and arbitrary registered modules without a
  special loading API.
- A multi-step block supports the sample/background x-y-z case.
- Expansion produces one new `ProcessStep` instance per child; instances are
  never reused across items.
- Child-level tracing, failures, timing, stop-after, and partial reruns remain
  available.
- The expanded pipeline can be exported and run without the compact schema.
- HDF provenance contains the authored YAML/spec and expanded YAML/spec at the
  four explicit grouped paths.
- Users can inspect all generated ids and dependencies before execution.
- The USAXS compact source is materially shorter and groups the eight diode
  lanes coherently, while its expanded DAG remains scientifically explicit.

## Cost-benefit conclusion

The feature is a medium, cross-cutting change because it touches parsing,
validation, serialization/provenance, and graph presentation. It is not a
scheduler rewrite. For the USAXS workflow and other instrument pipelines with
several homogeneous acquisition lanes, the benefit justifies that cost.

Implementing only a single-step mapper has a weaker benefit and risks immediate
redesign when repeated subpipelines are needed. The recommended first version
should therefore implement the small common denominator: one non-nested
`for_each` block containing one or more ordinary step templates, compiled to
the existing flat DAG.

# Glossary

`ref::path`
: Address of data in a registered source. `ref` selects the source; `path`
  selects content inside it. An optional `@attribute` suffix selects metadata.

basedata key
: Key of one `BaseData` object inside a `DataBundle`, such as `signal`, `Q`, or
  `mask`.

processing key
: Key of a `DataBundle` inside `ProcessingData`, such as `sample` or
  `background`.

dependency contract
: A step's declared external source references, processing reads, and processing
  writes.

dirty step
: A step selected for partial re-execution because it reads changed input or is
  downstream of another dirty step.

named uncertainty
: Absolute one-standard-deviation contribution stored under a stable name.

rank of data
: Number of trailing dimensions representing scalar, curve, image, or volume
  data, separate from leading scan/frame dimensions.

chunk plan
: Immutable description of expected chunk identities and destination layouts.

chunk spec
: Selection and placement contract for one chunk in a plan.

runtime policy
: Trusted or restricted rules controlling filesystem roots, custom imports,
  payload limits, and other server capabilities.

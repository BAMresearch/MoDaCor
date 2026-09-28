# Terminology

`BaseData`
: One scientific quantity: an array-like signal, units, named uncertainties,
  weights, axes, and rank metadata.

`DataBundle`
: A named collection of related `BaseData` quantities, such as intensity,
  scattering-vector coordinates, and masks for one sample.

`ProcessingData`
: The pipeline-wide mapping of bundle names to `DataBundle` objects.

`ProcessStep`
: One configured pipeline operation. It reads from sources or
  `ProcessingData`, writes authoritative results to `ProcessingData`, and
  declares its dependencies.

Source and sink
: Registered adapters for reading external data and publishing outputs.

Trace event
: A compact record of what a step observed or changed during one run.

Session
: Server-owned pipeline, registration, result, history, and error state used
  for repeated or partial execution.

For exact path syntax and API terms, see the [glossary](../reference/glossary.md).

# ProcessingData

`ProcessingData` is the mutable, in-memory workspace shared by all steps in one
pipeline run. It maps stable string names such as `sample`, `background`,
`calibration`, or `result` to `DataBundle` objects.

```python
from modacor.dataclasses.processing_data import ProcessingData

processing_data = ProcessingData()
processing_data["sample"] = sample
```

`ProcessStep.calculate()` makes authoritative changes directly in this object.
A step's optional return value is execution bookkeeping; it is not merged into
`ProcessingData` by the runner.

Keeping intermediate bundles makes changes inspectable and enables partial
reruns. The server can retain a previous snapshot, identify steps affected by a
changed source or path, and recompute the selected downstream graph.

A processing path such as `/sample/signal/uncertainties/Poisson` identifies:

1. the `sample` bundle;
2. its `signal` `BaseData` entry; and
3. the `Poisson` item in that entry's `uncertainties` mapping.

See [Paths, axes, weights, and masks](paths-axes-weights-and-masks.md) and the
[glossary](../reference/glossary.md) for path forms used by sinks and the API.

# DataBundle

`DataBundle` groups related `BaseData` quantities. A detector measurement might
contain `signal`, `Q`, `Psi`, `mask`, and `solid_angle`; a calibration bundle
might contain a scalar exposure time or transmission.

```python
from modacor.dataclasses.databundle import DataBundle

sample = DataBundle(signal=intensity)
sample.description = "Synthetic detector counts"
sample.default_plot = "signal"
```

Keys must be non-empty strings and values must be `BaseData` instances. Raw
arrays do not belong directly in a bundle: wrapping them makes units and
uncertainty semantics explicit.

Separate physical or logical quantities should use separate keys. Do not encode
units, uncertainty kinds, or processing history into key names; those have
dedicated fields and trace records.

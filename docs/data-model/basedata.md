# BaseData

`BaseData` represents one scientific quantity. Its required fields are a NumPy
array-like `signal` and a Pint `units` value. It can additionally carry named
absolute uncertainties, weights, coordinate axes, and `rank_of_data`.

```python
import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData

intensity = BaseData(
    signal=np.array([100.0, 144.0]),
    units=ureg.count,
    uncertainties={"Poisson": np.array([10.0, 12.0])},
    rank_of_data=1,
)
```

Scalar and list inputs are converted to floating NumPy arrays. Existing NumPy
arrays remain arrays without an unconditional dtype conversion. Every
uncertainty and weight must be scalar or broadcast exactly to the signal shape;
an operand that would enlarge the signal is rejected.

`uncertainties` stores one-standard-deviation values. The dictionary-like
`variances` property exposes their squares and accepts variances on assignment.
Use `uncertainties` in new code unless a variance-based algorithm specifically
requires the view.

`BaseData` arithmetic applies Pint unit algebra and the
[documented uncertainty rules](uncertainty-propagation.md). It preserves axes,
rank, and weights only when their structural metadata is compatible. Modules
requiring stricter coordinate equality must validate that domain rule before
arithmetic.

Useful operations include unit conversion, copying, slicing, squeezing,
addition, subtraction, multiplication, division, powers, square roots,
logarithms, exponentials, and trigonometric functions. Invalid units or domains
fail explicitly or produce documented NaNs; uncertainty arrays are never
silently discarded merely to make an operation succeed.

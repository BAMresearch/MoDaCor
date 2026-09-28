# average or sum, weighted or unweighted, over axes

## Summary
Compute a weighted or unweighted mean/sum over the given axes, propagate existing uncertainties, and optionally add scatter-derived uncertainty estimates.

## Metadata
- **Import path:** `modacor.modules.base_modules.reduce_dimensionality.ReduceDimensionality`
- **Source:** [`src/modacor/modules/base_modules/reduce_dimensionality.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/reduce_dimensionality.py)
- **Module ID:** ReduceDimensionality
- **Module version:** 20260927.1
- **Keywords:** average, mean, weighted, nanmean, reduce, axis, sum

## Required data keys
- signal

## Modifies
- **signal**: signal, uncertainties, units, weights

## Required arguments
- _None_

## Default configuration
```json
{
  "axes": null,
  "mask_bits": null,
  "mask_key": null,
  "nan_policy": "omit",
  "reduction": "mean",
  "uncertainty_estimation": null,
  "use_weights": true
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axes` | int or list or tuple or str or NoneType | No | - | - | Axis or axes to reduce (int, list/tuple, or None for all). Use 'non_data' to reduce every leading axis before the final rank_of_data axes. |
| `mask_bits` | int or list or tuple or NoneType | No | - | - | Optional uint32 bit value or iterable of bit values to exclude. None excludes any nonzero mask. |
| `mask_key` | str or NoneType | No | - | - | Optional BaseData mask key in each selected DataBundle. Nonzero mask values are excluded without modifying the input signal. |
| `nan_policy` | str | No | omit | - | NaN handling policy: 'omit' or 'propagate'. |
| `reduction` | str | No | mean | - | Reduction method: 'mean' or 'sum'. |
| `uncertainty_estimation` | dict or NoneType | No | - | - | Optional mapping containing estimator definitions and a collision policy for scatter-derived uncertainty components. |
| `use_weights` | bool | No | True | - | Use BaseData weights for weighted reduction. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
This step reduces the dimensionality of the signal by averaging over one or more axes. With axes='non_data', it automatically reduces leading acquisition axes until signal.ndim equals rank_of_data. An optional integer mask can exclude selected reason bits without modifying the input signal. Units are preserved; complete axes metadata is reduced along the same axes.

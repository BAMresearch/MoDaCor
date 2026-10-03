# Subtract remapped one-dimensional background

## Summary
Remap a 1D background by nearest neighbour or linear interpolation, then subtract it.

## Metadata
- **Import path:** `modacor.modules.base_modules.subtract_interpolated_1d.SubtractInterpolated1D`
- **Source:** [`src/modacor/modules/base_modules/subtract_interpolated_1d.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/subtract_interpolated_1d.py)
- **Module ID:** SubtractInterpolated1D
- **Module version:** 20260929.1
- **Keywords:** subtract, background, nearest, linear, remap, 1D

## Required data keys
- signal
- Q

## Modifies
- **signal**: signal, uncertainties, weights, units
- **remap_mask**: signal

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "axis_key": "Q",
  "mask_key": null,
  "max_gap": null,
  "max_gap_units": null,
  "mode": "nearest",
  "output_mask_key": "remap_mask",
  "output_signal_key": "signal",
  "remapped_background_key": null,
  "signal_key": "signal",
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axis_key` | str | No | Q | - | One-dimensional coordinate key in both bundles. |
| `mask_key` | str or NoneType | No | - | - | Optional input mask key used in both bundles; nonzero means invalid. |
| `max_gap` | float or int or NoneType | No | - | - | Optional maximum linear-interpolation bracket width; unsupported in nearest mode. |
| `max_gap_units` | str or NoneType | No | - | - | Units of max_gap; defaults to the sample axis units. |
| `mode` | str | No | nearest | - | Remapping mode: nearest or linear. |
| `output_mask_key` | str or NoneType | No | remap_mask | - | Optional output key marking targets invalid for subtraction. |
| `output_signal_key` | str | No | signal | - | Sample-bundle key receiving the subtracted signal. |
| `remapped_background_key` | str or NoneType | No | - | - | Optional sample-bundle key receiving the remapped background. |
| `signal_key` | str | No | signal | - | BaseData signal key in both bundles. |
| `with_processing_keys` | list | Yes | - | - | Two processing keys: sample/minuend then background/subtrahend. |

## Notes
Nearest mode always chooses the nearest valid background coordinate and has no tolerance setting. Neither mode extrapolates: sample points outside the background domain are invalidated. max_gap applies only to linear bracket width. Linear uncertainties are propagated from interpolation coefficients before BaseData subtraction.

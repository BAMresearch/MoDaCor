# Find one-dimensional intensity centre of mass

## Summary
Find a masked, windowed intensity centroid without assuming a peak shape.

## Metadata
- **Import path:** `modacor.modules.base_modules.find_center_of_mass_1d.FindCenterOfMass1D`
- **Source:** [`src/modacor/modules/base_modules/find_center_of_mass_1d.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/find_center_of_mass_1d.py)
- **Module ID:** FindCenterOfMass1D
- **Module version:** 20260929.1
- **Keywords:** centroid, center of mass, peak, beam centre, 1D

## Required data keys
- signal

## Modifies
- **beam_center**: signal, uncertainties, units
- **centroid_count**: signal
- **centroid_converged**: signal
- **centroid_window_min**: signal, units
- **centroid_window_max**: signal, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "axis_key": "yaw",
  "baseline": 0.0,
  "baseline_units": null,
  "convergence_tolerance": 0.0,
  "diagnostic_prefix": "centroid",
  "half_width": null,
  "mask_key": null,
  "maximum_iterations": 5,
  "output_key": "beam_center",
  "signal_key": "signal",
  "width_units": null,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axis_key` | str | No | yaw | - | One-dimensional coordinate BaseData key. |
| `baseline` | float or int | No | 0.0 | - | Scalar baseline in baseline_units, subtracted before clipping negative weights. |
| `baseline_units` | str or NoneType | No | - | - | Units for baseline; defaults to signal units. |
| `convergence_tolerance` | float or int | No | 0.0 | - | Absolute centre-change tolerance in width_units. |
| `diagnostic_prefix` | str | No | centroid | - | Prefix for count, convergence, and window-bound diagnostic keys. |
| `half_width` | float or int or NoneType | No | - | - | Optional physical half-width around the iterated centre. |
| `mask_key` | str or NoneType | No | - | - | Optional mask key; nonzero values are excluded. |
| `maximum_iterations` | int | No | 5 | - | Maximum number of window-recentring iterations. |
| `output_key` | str | No | beam_center | - | Scalar centroid output key. |
| `signal_key` | str | No | signal | - | Intensity BaseData key. |
| `width_units` | str or NoneType | No | - | - | Units for half_width and convergence_tolerance; defaults to axis units. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | DataBundle key or keys containing peaks to centre. |

## Notes
The coarse maximum selects a contiguous valid peak region. After optional baseline subtraction, negative intensities are clipped to zero. Signal and coordinate uncertainty components are propagated separately to the scalar centre.

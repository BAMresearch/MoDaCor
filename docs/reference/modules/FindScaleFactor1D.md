# Scale 1D curve to reference (compute-only)

## Summary
Compute a normal robust-fit or uncertainty-weighted lognormal scale between two 1D curves.

## Metadata
- **Import path:** `modacor.modules.base_modules.find_scale_factor1d.FindScaleFactor1D`
- **Source:** [`src/modacor/modules/base_modules/find_scale_factor1d.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/find_scale_factor1d.py)
- **Module ID:** FindScaleFactor1D
- **Module version:** 20261002.1
- **Keywords:** scale, calibration, lognormal, 1D

## Required data keys
- signal
- Q

## Modifies
- **scale_factor**: signal, uncertainties, units
- **scale_background**: signal, uncertainties, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "background_output_key": "scale_background",
  "background_uncertainty_key": "propagate_to_all",
  "diagnostic_prefix": null,
  "fit_background": false,
  "fit_max_val": null,
  "fit_min_val": null,
  "fit_model": "normal",
  "fit_val_units": null,
  "independent_axis_key": "Q",
  "interpolation_kind": "linear",
  "require_overlap": true,
  "robust_fscale": 1.0,
  "robust_loss": "huber",
  "scale_output_key": "scale_factor",
  "scale_uncertainty_key": "propagate_to_all",
  "signal_key": "signal",
  "uncertainty_weight_key": null,
  "use_basedata_weights": true,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `background_output_key` | str | No | scale_background | - | BaseData key to store the fitted background output. |
| `background_uncertainty_key` | str | No | propagate_to_all | - | Named uncertainty component used for the fitted background uncertainty. |
| `diagnostic_prefix` | str or NoneType | No | - | - | Optional prefix for scalar '<prefix>_point_count', '<prefix>_x_min', '<prefix>_x_max', and '<prefix>_reduced_chi_square' outputs. |
| `fit_background` | bool | No | False | - | Whether to fit a constant background offset. |
| `fit_max_val` | float or int or NoneType | No | - | - | Maximum x-value for the fit (in fit_val_units). |
| `fit_min_val` | float or int or NoneType | No | - | - | Minimum x-value for the fit (in fit_val_units). |
| `fit_model` | str | No | normal | - | Scale estimator: normal or lognormal. |
| `fit_val_units` | str or NoneType | No | - | - | Units for fit_min_val/fit_max_val if provided. |
| `independent_axis_key` | str | No | Q | - | BaseData key for the independent axis. |
| `interpolation_kind` | str | No | linear | - | Interpolation kind passed to scipy/numpy interpolation. |
| `require_overlap` | bool | No | True | - | Require overlapping x-range between reference and work data. |
| `robust_fscale` | float or int | No | 1.0 | - | Robust loss scale parameter. |
| `robust_loss` | str | No | huber | - | Robust loss function name for the fit. |
| `scale_output_key` | str | No | scale_factor | - | BaseData key to store the scale factor output. |
| `scale_uncertainty_key` | str | No | propagate_to_all | - | Named uncertainty component used for the fitted scale-factor uncertainty. |
| `signal_key` | str | No | signal | - | BaseData key for the dependent variable signal. Working and reference signals must have compatible units; fitting uses the reference units. |
| `uncertainty_weight_key` | str or NoneType | No | - | - | Named propagated uncertainty component used for weighting on both curves. Required for lognormal fitting; normal fitting combines components only when this is None. |
| `use_basedata_weights` | bool | No | True | - | Use BaseData weights when fitting. |
| `with_processing_keys` | list | Yes | - | - | Two processing keys: working curve then reference curve. |

## References
DOI 10.1107/S1600577513030117

## Notes
The fitted scale uncertainty is stored under scale_uncertainty_key. Reduced chi-square is a goodness-of-fit diagnostic; it does not automatically inflate the formal fitted uncertainty.

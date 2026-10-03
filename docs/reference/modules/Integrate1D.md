# Integrate one-dimensional curves

## Summary
Integrate sampled 1D BaseData while propagating coordinate units and independent uncertainty components.

## Metadata
- **Import path:** `modacor.modules.base_modules.integrate_1d.Integrate1D`
- **Source:** [`src/modacor/modules/base_modules/integrate_1d.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/integrate_1d.py)
- **Module ID:** Integrate1D
- **Module version:** 20260929.2
- **Keywords:** integrate, quadrature, trapezoid, simpson, 1D

## Required data keys
- signal
- q

## Modifies
- **integral**: signal, uncertainties, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "axis_key": "q",
  "duplicate_axis": "error",
  "mask_key": null,
  "method": "trapezoid",
  "output_key": "integral",
  "output_processing_keys": null,
  "signal_key": "signal",
  "sort_axis": false,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axis_key` | str | No | q | - | One-dimensional integration coordinate BaseData key. |
| `duplicate_axis` | str | No | error | - | Handling of repeated coordinates: error or mean. |
| `mask_key` | str or NoneType | No | - | - | Optional mask BaseData key; nonzero values are excluded. |
| `method` | str | No | trapezoid | - | Quadrature rule: trapezoid or simpson. |
| `output_key` | str | No | integral | - | Output key when the integral remains in its input DataBundle. |
| `output_processing_keys` | list or NoneType | No | - | - | Optional new DataBundle key per input; each result is stored as signal. |
| `signal_key` | str | No | signal | - | One-dimensional BaseData entry to integrate. |
| `sort_axis` | bool | No | False | - | Stably sort the common valid coordinate before integration. |
| `with_processing_keys` | list | Yes | - | - | DataBundles integrated over their common valid domain. |

## Notes
The shared coordinate may be nonuniform. It must be monotonic unless sort_axis is enabled. Repeated coordinates are rejected by default; duplicate_axis='mean' replaces each run with its arithmetic mean signal and independently propagated uncertainty of that mean. Invalid or masked samples in any input are omitted from every integral.

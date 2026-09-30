# Indexed Averager

## Summary
Average signal and geometry using precomputed pixel bin indices.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.indexed_averager.IndexedAverager`
- **Source:** [`src/modacor/modules/technique_modules/scattering/indexed_averager.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/indexed_averager.py)
- **Module ID:** IndexedAverager
- **Module version:** 20260929.1
- **Keywords:** radial, azimuthal, averaging, binning, scattering

## Required data keys
- signal
- Q
- pixel_index

## Modifies
- **signal**: signal, uncertainties
- **Q**: signal, uncertainties
- **Psi**: signal, uncertainties
- **bin_count**: signal
- **positive_weight_count**: signal
- **sum_weights**: signal
- **effective_sample_size**: signal

## Required arguments
- with_processing_keys
- averaging_direction

## Default configuration
```json
{
  "averaging_direction": "azimuthal",
  "output_processing_key": null,
  "stats_keys": null,
  "uncertainty_weight_key": null,
  "use_signal_uncertainty_weights": false,
  "use_signal_weights": true,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `averaging_direction` | str | Yes | azimuthal | - | Averaging direction: 'radial' or 'azimuthal'. |
| `output_processing_key` | str or NoneType | No | - | - | Optional key for a new reduced DataBundle. This requires exactly one input and preserves the source DataBundle unchanged. |
| `stats_keys` | list or str or NoneType | No | - | - | BaseData keys to receive SEM/STD statistics (e.g. ['signal', 'Q']). If None, statistics are computed for all outputs. |
| `uncertainty_weight_key` | str or NoneType | No | - | - | Uncertainty key to use as weights if enabled. |
| `use_signal_uncertainty_weights` | bool | No | False | - | Use signal uncertainty as weights. |
| `use_signal_weights` | bool | No | True | - | Use BaseData weights when averaging signal. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | ProcessingData key or list of keys to average. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
IndexedAverager expects a 'pixel_index' map from a previous IndexPixels step and performs per-bin weighted means of signal, Q and Psi, including uncertainty propagation.

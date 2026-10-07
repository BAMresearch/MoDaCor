# Indexed Averager

## Summary
Compute weighted group means from a precomputed integer index map.

## Metadata
- **Import path:** `modacor.modules.base_modules.indexed_averager.IndexedAverager`
- **Source:** [`src/modacor/modules/base_modules/indexed_averager.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/indexed_averager.py)
- **Module ID:** IndexedAverager
- **Module version:** 20261001.1
- **Keywords:** indexed, grouped, weighted average, reduction

## Required data keys
- _None_

## Modifies
- **configured value**: signal, uncertainties, axes
- **configured axis**: signal, uncertainties
- **bin_id**: signal
- **bin_count**: signal
- **positive_weight_count**: signal
- **sum_weights**: signal
- **effective_sample_size**: signal

## Required arguments
- _None_

## Default configuration
```json
{
  "axis_key": "Q",
  "bin_id_key": "bin_id",
  "index_key": "bin_index",
  "mask_key": "Mask",
  "output_processing_key": null,
  "stats_keys": null,
  "uncertainty_weight_key": null,
  "use_value_uncertainty_weights": false,
  "use_value_weights": true,
  "value_key": "signal",
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axis_key` | str or NoneType | No | Q | - | Optional BaseData to average over the same points and attach as the output axis. |
| `bin_id_key` | str | No | bin_id | - | Output BaseData key recording the original populated bin IDs. |
| `index_key` | str | No | bin_index | - | BaseData containing precomputed integer group indices. |
| `mask_key` | str or NoneType | No | Mask | - | Optional mask BaseData; true values are excluded when the key is present. |
| `output_processing_key` | str or NoneType | No | - | - | Optional distinct output bundle; requires exactly one input bundle. |
| `stats_keys` | list or str or NoneType | No | - | - | Value and/or axis keys receiving scatter-derived SEM and STD; None selects both. |
| `uncertainty_weight_key` | str or NoneType | No | - | - | Value uncertainty component used for inverse-variance weighting. |
| `use_value_uncertainty_weights` | bool | No | False | - | Also weight by inverse variance from one named value uncertainty. |
| `use_value_weights` | bool | No | True | - | Use the value BaseData weights for both the value and output axis. |
| `value_key` | str | No | signal | - | BaseData value to average. |
| `with_processing_keys` | str or list or NoneType | No | - | - | ProcessingData key(s) to reduce. |

## Notes
The index map alone determines membership. The optional axis is averaged over the same valid, unmasked observations and weights; bin edges are neither required nor read.

# Create a threshold mask from a BaseData entry

## Summary
Create a uint32 mask by evaluating bounds on a selected BaseData signal array.

## Metadata
- **Import path:** `modacor.modules.base_modules.threshold_mask.ThresholdMask`
- **Source:** [`src/modacor/modules/base_modules/threshold_mask.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/threshold_mask.py)
- **Module ID:** ThresholdMask
- **Module version:** 20260927.3
- **Keywords:** mask, threshold, databundle

## Required data keys
- signal

## Modifies
- **threshold_mask**: signal, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "lower_bound": null,
  "mask_mode": "outside",
  "source_basedata_key": "signal",
  "target_mask_key": "threshold_mask",
  "threshold": null,
  "upper_bound": null,
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `lower_bound` | float or int or NoneType | No | - | - | Optional inclusive lower-bound magnitude in the source signal's current units. |
| `mask_mode` | str | No | outside | - | Use 'outside' to mask values below/above the bounds, or 'inside' to mask values within them. |
| `source_basedata_key` | str | No | signal | processing_read_basedata_key | BaseData key whose signal array is evaluated to create the mask. |
| `target_mask_key` | str | No | threshold_mask | processing_write_basedata_key | BaseData key for the mask to create inside the DataBundle. |
| `threshold` | float or int or NoneType | No | - | - | Deprecated compatibility alias for upper_bound when upper_bound is not set; interpreted in the source signal's current units. |
| `upper_bound` | float or int or NoneType | No | - | - | Optional inclusive upper-bound magnitude in the source signal's current units. |
| `with_processing_keys` | list | Yes | ["sample"] | - | Single processing key identifying the DataBundle to update. |

## Notes

            Configuration:
              with_processing_keys: [sample]        # required, single databundle key
              source_basedata_key: flatfield        # optional, default: signal
              target_mask_key: flatfield_mask       # optional, default: threshold_mask
              lower_bound: 0.8                      # optional
              upper_bound: 1.2                      # optional
              mask_mode: outside                    # outside or inside

            Performs without changing the source array dimensionality:
              outside: mask = (source < lower_bound) | (source > upper_bound)
              inside:  mask = (source >= lower_bound) & (source <= upper_bound)

            The legacy threshold option is treated as upper_bound when
            upper_bound is not configured. Bounds are numeric magnitudes in
            the source signal's current units; this step does not convert them.

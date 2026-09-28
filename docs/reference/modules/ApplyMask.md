# Apply mask to the signal in a DataBundle

## Summary
Apply a uint32 bitfield mask to one or more BaseData signal arrays in the same DataBundle.

## Metadata
- **Import path:** `modacor.modules.base_modules.apply_mask.ApplyMask`
- **Source:** [`src/modacor/modules/base_modules/apply_mask.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/apply_mask.py)
- **Module ID:** ApplyMask
- **Module version:** 20260927.2
- **Keywords:** mask, signal, apply, databundle

## Required data keys
- mask
- signal

## Modifies
- **signal**: signal

## Required arguments
- with_processing_keys
- basedata_to_mask

## Default configuration
```json
{
  "basedata_to_mask": [
    "signal"
  ],
  "mask_key": "mask",
  "masked_value": "nan",
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `basedata_to_mask` | list | Yes | ["signal"] | processing_read_write_basedata_key_list | List of BaseData keys to apply the mask to. |
| `mask_key` | str | No | mask | processing_read_basedata_key | BaseData key for the mask to be used inside the DataBundle. |
| `masked_value` | str or int or float or NoneType | No | nan | - | Replacement value for masked pixels. Use 'nan' for NaN, or a numeric value such as 0 or -1. |
| `with_processing_keys` | list | Yes | ["sample"] | - | Single processing key identifying the DataBundle to update. |

## References
NeXus mask bit-field convention (NXdata/NXdetector masks)

## Notes

            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              mask_key: mask                     # optional, default: mask
              basedata_to_mask: [signal, ...]    # optional, default: [signal]
              masked_value: nan                   # optional; use 0 or -1 for explicit sentinels

            Performs:
              basedata[mask != 0] = masked_value  (in-place, for each source)

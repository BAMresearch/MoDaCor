# Combine masks within one DataBundle (bitwise OR)

## Summary
Combine multiple mask arrays stored as different BaseData keys in the same DataBundle.

## Metadata
- **Import path:** `modacor.modules.base_modules.bitwise_or_masks.BitwiseOrMasks`
- **Source:** [`src/modacor/modules/base_modules/bitwise_or_masks.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/bitwise_or_masks.py)
- **Module ID:** BitwiseOrMasks
- **Module version:** 20260927.2
- **Keywords:** mask, bitmask, bitwise, or, databundle

## Required data keys
- mask

## Modifies
- **mask**: signal

## Required arguments
- with_processing_keys
- source_mask_keys

## Default configuration
```json
{
  "source_mask_keys": [],
  "target_mask_key": "mask",
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `source_mask_keys` | list | Yes | [] | processing_read_basedata_key_list | List of BaseData keys to OR into the target mask. |
| `target_mask_key` | str | No | mask | processing_read_write_basedata_key | BaseData key for the target mask inside the DataBundle. |
| `with_processing_keys` | list | Yes | ["sample"] | - | Single processing key identifying the DataBundle to update. |

## References
NeXus mask bit-field convention (NXdata/NXdetector masks)

## Notes

            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              target_mask_key: mask              # optional, default: mask
              source_mask_keys: [bs_mask, ...]   # required, one or more

            Performs:
              target_mask |= source_mask  (in-place, for each source)

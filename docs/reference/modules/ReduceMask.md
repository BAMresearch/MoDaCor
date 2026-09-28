# Reduce mask over axes

## Summary
Reduce an integer mask over configured axes while preserving uint32 reason bits.

## Metadata
- **Import path:** `modacor.modules.base_modules.reduce_mask.ReduceMask`
- **Source:** [`src/modacor/modules/base_modules/reduce_mask.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/reduce_mask.py)
- **Module ID:** ReduceMask
- **Module version:** 20260927.2
- **Keywords:** mask, reduce, bitfield

## Required data keys
- mask

## Modifies
- **mask**: signal, units, axes

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "axes": null,
  "reduction": "any",
  "source_mask_key": "mask",
  "target_mask_key": "mask",
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `axes` | int or list or tuple or str or NoneType | No | - | - | Axis or axes to reduce. Use None to reduce all axes, or 'non_data' to reduce every leading axis before the final rank_of_data axes. |
| `reduction` | str | No | any | - | Use 'any' to OR mask bits across axes, or 'all' to AND bits across axes. |
| `source_mask_key` | str | No | mask | - | BaseData key for the mask to reduce. |
| `target_mask_key` | str | No | mask | - | BaseData key for the reduced mask output. |
| `with_processing_keys` | list | Yes | ["sample"] | - | Single processing key identifying the DataBundle to update. |

## Notes
For 'any', a bit remains set if it is set in any reduced element. For 'all', a bit remains set only if it is set in all reduced elements.

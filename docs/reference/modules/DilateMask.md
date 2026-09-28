# Dilate mask

## Summary
Dilate a 2D integer mask over selected axes while preserving uint32 reason bits.

## Metadata
- **Import path:** `modacor.modules.base_modules.dilate_mask.DilateMask`
- **Source:** [`src/modacor/modules/base_modules/dilate_mask.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/dilate_mask.py)
- **Module ID:** DilateMask
- **Module version:** 20260927.2
- **Keywords:** mask, dilate, morphology, bitfield

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
  "backend": "auto",
  "footprint_shape": "square",
  "radius": 1,
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
| `axes` | list or tuple or NoneType | No | - | - | Two axes to dilate over. Defaults to the last two axes. |
| `backend` | str | No | auto | - | Use 'auto', 'skimage', or 'scipy'. 'auto' prefers scikit-image when installed. |
| `footprint_shape` | str | No | square | - | 2D dilation footprint: 'square', 'disk', or 'cross'. |
| `radius` | int | No | 1 | - | Dilation radius in pixels. Aliases: number_of_pixels, number_of_pixels_to_dilate. |
| `source_mask_key` | str | No | mask | - | BaseData key for the mask to dilate. |
| `target_mask_key` | str | No | mask | - | BaseData key for the dilated mask output. |
| `with_processing_keys` | list | Yes | ["sample"] | - | Single processing key identifying the DataBundle to update. |

## Notes

            Configuration:
              with_processing_keys: [static]     # required, single databundle key
              source_mask_key: mask              # optional, default: mask
              target_mask_key: mask              # optional, default: mask
              radius: 1                          # optional
              footprint_shape: square            # square, disk, or cross

            If the mask has leading dimensions, dilation is applied plane-by-plane
            over the configured 2D axes. Each NeXus bit is dilated independently.

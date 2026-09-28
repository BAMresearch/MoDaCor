# Copy DataBundle keys

## Summary
Copy selected BaseData entries between DataBundles.

## Metadata
- **Import path:** `modacor.modules.base_modules.copy_databundle_keys.CopyDataBundleKeys`
- **Source:** [`src/modacor/modules/base_modules/copy_databundle_keys.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/copy_databundle_keys.py)
- **Module ID:** CopyDataBundleKeys
- **Module version:** 20260927.2
- **Keywords:** copy, databundle, static

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "copy": true,
  "copy_axes": true,
  "data_keys": null,
  "key_map": null,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `copy` | bool | No | True | - | Deep-copy copied values. If False, attach the same objects by reference. |
| `copy_axes` | bool | No | True | - | Pass with_axes to BaseData.copy when copy is True. |
| `data_keys` | list or str or NoneType | No | - | - | Source keys to copy to the target DataBundle under the same names. |
| `key_map` | dict or NoneType | No | - | - | Mapping of source key to target key. Mutually exclusive with data_keys. |
| `with_processing_keys` | list | Yes | - | - | Two processing keys: target then source. |

## Notes
Use this to attach static maps such as Q, Psi, Omega, pixel_index, or masks to each sample without recomputing the static branch.

# Concatenate DataBundles

## Summary
Concatenate matching 1D BaseData entries, optionally sorting every entry together.

## Metadata
- **Import path:** `modacor.modules.base_modules.concatenate_databundles.ConcatenateDatabundles`
- **Source:** [`src/modacor/modules/base_modules/concatenate_databundles.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/concatenate_databundles.py)
- **Module ID:** ConcatenateDatabundles
- **Module version:** 20260929.1
- **Keywords:** concatenate, pool, curves, sort

## Required data keys
- _None_

## Modifies
- **configured data keys**: signal, uncertainties, weights, units, axes
- **source_index**: signal

## Required arguments
- with_processing_keys
- data_keys
- output_processing_key

## Default configuration
```json
{
  "data_keys": [
    "signal",
    "Q"
  ],
  "descending": false,
  "output_processing_key": "concatenated",
  "sort_by": null,
  "source_index_key": "source_index",
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `data_keys` | list or str | Yes | ["signal", "Q"] | - | Matching one-dimensional BaseData entries to concatenate. |
| `descending` | bool | No | False | - | Sort descending instead of ascending when sort_by is set. |
| `output_processing_key` | str | Yes | concatenated | - | ProcessingData key receiving the pooled DataBundle. |
| `sort_by` | str or NoneType | No | - | - | Optional concatenated data key used for coordinated stable sorting. |
| `source_index_key` | str or NoneType | No | source_index | - | Optional output key recording each point's zero-based input-bundle index. |
| `with_processing_keys` | list | Yes | - | - | Input DataBundle keys, in concatenation order. |

## Notes
Input order is preserved when sort_by is None. Units are converted to those of the first input. Uncertainty component names must match across inputs. Sorting is stable and is not required by IndexPixels.

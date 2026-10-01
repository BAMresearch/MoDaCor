# Index by Coordinate

## Summary
Assign one coordinate value per input point to a one-dimensional bin index.

## Metadata
- **Import path:** `modacor.modules.base_modules.index_by_coordinate.IndexByCoordinate`
- **Source:** [`src/modacor/modules/base_modules/index_by_coordinate.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/index_by_coordinate.py)
- **Module ID:** IndexByCoordinate
- **Module version:** 20261001.1
- **Keywords:** indexing, binning, coordinate, reduction

## Required data keys
- _None_

## Modifies
- **bin_index**: signal, units, axes
- **bin_edges**: signal, units

## Required arguments
- _None_

## Default configuration
```json
{
  "bin_edges": null,
  "bin_max": null,
  "bin_min": null,
  "bin_units": null,
  "coordinate_key": "Q",
  "edges_key": "bin_edges",
  "index_key": "bin_index",
  "n_bins": null,
  "spacing": null,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `bin_edges` | list or tuple or NoneType | No | - | - | Explicit bin edges in bin_units; mutually exclusive with generated-edge settings. |
| `bin_max` | float or int or NoneType | No | - | - | Generated upper edge in bin_units; inferred when omitted. |
| `bin_min` | float or int or NoneType | No | - | - | Generated lower edge in bin_units; inferred when omitted. |
| `bin_units` | str or NoneType | No | - | - | Units for configured limits or edges; defaults to coordinate units. |
| `coordinate_key` | str | No | Q | processing_read_basedata_key | BaseData coordinate used to assign bins. |
| `edges_key` | str | No | bin_edges | processing_write_basedata_key | Output BaseData key recording the physical bin edges. |
| `index_key` | str | No | bin_index | processing_write_basedata_key | Output BaseData key for the integer index map. |
| `n_bins` | int or NoneType | No | - | - | Generated bin count; defaults to 100 when omitted. |
| `spacing` | str or NoneType | No | - | - | Generated edge spacing, 'linear' or 'log'; defaults to 'linear'. |
| `with_processing_keys` | str or list or NoneType | No | - | - | ProcessingData key(s) whose coordinate should be indexed. |

## Notes
Bins are left-inclusive and right-exclusive except for the included final right edge. Non-finite and out-of-range coordinates receive index -1. Masks are applied by downstream reducers.

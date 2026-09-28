# Update unit labels

## Summary
Update unit labels of one or more BaseData elements (no conversion).

## Metadata
- **Import path:** `modacor.modules.base_modules.units_label_update.UnitsLabelUpdate`
- **Source:** [`src/modacor/modules/base_modules/units_label_update.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/units_label_update.py)
- **Module ID:** UnitsLabelUpdate
- **Module version:** 20260927.2
- **Keywords:** units, update, standardize

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- update_pairs

## Default configuration
```json
{
  "update_pairs": {}
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `update_pairs` | dict | Yes | {} | - | Mapping of BaseData key to unit string or {'units': str}. |

## References
DOI 10.1088/0953-8984/25/38/383201

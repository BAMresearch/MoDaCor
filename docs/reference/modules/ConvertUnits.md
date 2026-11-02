# Convert units

## Summary
Convert selected BaseData signals and all their named uncertainties to compatible units.

## Metadata
- **Import path:** `modacor.modules.base_modules.convert_units.ConvertUnits`
- **Source:** [`src/modacor/modules/base_modules/convert_units.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/convert_units.py)
- **Module ID:** ConvertUnits
- **Module version:** 20261102.1
- **Keywords:** units, conversion, output

## Required data keys
- _None_

## Modifies
- _None_

## Data dependencies
_No fixed data keys declared._

| Role argument | Access | Default |
|---|---|---|
| `units` | read_write | `{}` |

## Required arguments
- units

## Default configuration
```json
{
  "units": {},
  "with_processing_keys": "*"
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `units` | dict | Yes | {} | processing_read_write_basedata_key | Mapping of BaseData key to target units, for example {Q: '1/nm'}. |
| `with_processing_keys` | list or str | No | * | - | List of DataBundle keys, or '*' for all bundles. Missing data keys are skipped only with '*'. |

## Notes
Conversion mutates each BaseData in place, preserving axis references and dimensionless relative weights. With '*' missing data keys are skipped, with one INFO summary per step; with explicit bundles they raise KeyError.

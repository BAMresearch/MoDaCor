# Divide by another DataBundle

## Summary
Divide a DataBundle entry using another DataBundle.

## Metadata
- **Import path:** `modacor.modules.base_modules.divide_databundles.DivideDatabundles`
- **Source:** [`src/modacor/modules/base_modules/divide_databundles.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/divide_databundles.py)
- **Module ID:** DivideDatabundles
- **Module version:** 20260927.2
- **Keywords:** divide, normalize, databundle

## Required data keys
- signal

## Modifies
- **signal**: signal, uncertainties, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "dividend_data_key": "signal",
  "divisor_data_key": "signal",
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `dividend_data_key` | str | No | signal | - | BaseData key to modify in the dividend DataBundle. |
| `divisor_data_key` | str | No | signal | - | BaseData key to read from the divisor DataBundle. |
| `with_processing_keys` | list | Yes | - | - | Two processing keys: dividend then divisor. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
with_processing_keys contains the dividend first and divisor second. BaseData supplies broadcasting, unit handling, and uncertainty propagation.

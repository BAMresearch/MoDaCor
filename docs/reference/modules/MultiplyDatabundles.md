# Multiply another DataBundle

## Summary
Multiply a DataBundle element using another DataBundle

## Metadata
- **Import path:** `modacor.modules.base_modules.multiply_databundles.MultiplyDatabundles`
- **Source:** [`src/modacor/modules/base_modules/multiply_databundles.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/multiply_databundles.py)
- **Module ID:** MultiplyDatabundles
- **Module version:** 20260927.2
- **Keywords:** multiply, scaling, databundle

## Required data keys
- signal

## Modifies
- **signal**: signal, uncertainties, units

## Required arguments
- _None_

## Default configuration
```json
{
  "multiplicand_data_key": "signal",
  "multiplier_data_key": "signal"
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `multiplicand_data_key` | str | No | signal | - | BaseData key to modify in the multiplicand DataBundle. |
| `multiplier_data_key` | str | No | signal | - | BaseData key to read from the multiplier DataBundle. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes

            This multiplies one DataBundle's signal with another, useful for scaling or combining data.
            'with_processing_keys' in the configuration should contain two keys, the operation
            will multiply the first key's DataBundle by the second key's DataBundle.

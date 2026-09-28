# Multiply by IoSource data

## Summary
Multiply a DataBundle element by a multiplier loaded from a data source

## Metadata
- **Import path:** `modacor.modules.base_modules.multiply.Multiply`
- **Source:** [`src/modacor/modules/base_modules/multiply.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/multiply.py)
- **Module ID:** Multiply
- **Module version:** 20260927.2
- **Keywords:** multiply, scalar, array

## Required data keys
- signal

## Modifies
- **signal**: signal, uncertainties, units

## Required arguments
- _None_

## Default configuration
```json
{
  "multiplier_source": null,
  "multiplier_uncertainties_sources": {},
  "multiplier_units_source": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `multiplier_source` | str | No | - | - | IoSources key for the multiplier signal. |
| `multiplier_uncertainties_sources` | dict | No | {} | - | Mapping of uncertainty name to IoSources key. |
| `multiplier_units_source` | str | No | - | - | IoSources key for multiplier units metadata. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal

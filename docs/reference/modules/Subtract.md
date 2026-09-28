# Subtract by IoSource data

## Summary
Subtract a DataBundle element by a subtrahend loaded from a data source

## Metadata
- **Import path:** `modacor.modules.base_modules.subtract.Subtract`
- **Source:** [`src/modacor/modules/base_modules/subtract.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/subtract.py)
- **Module ID:** Subtract
- **Module version:** 20260927.2
- **Keywords:** subtract, scalar, array

## Required data keys
- signal

## Modifies
- **signal**: signal, uncertainties, units

## Required arguments
- _None_

## Default configuration
```json
{
  "subtrahend_source": null,
  "subtrahend_uncertainties_sources": {},
  "subtrahend_units_source": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `subtrahend_source` | str | No | - | - | IoSources key for the subtrahend signal. |
| `subtrahend_uncertainties_sources` | dict | No | {} | - | Mapping of uncertainty name to IoSources key. |
| `subtrahend_units_source` | str | No | - | - | IoSources key for subtrahend units metadata. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal

# Add Poisson Uncertainties

## Summary
Add Poisson uncertainties to the data

## Metadata
- **Import path:** `modacor.modules.base_modules.poisson_uncertainties.PoissonUncertainties`
- **Source:** [`src/modacor/modules/base_modules/poisson_uncertainties.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/poisson_uncertainties.py)
- **Module ID:** PoissonUncertainties
- **Module version:** 20260927.2
- **Keywords:** uncertainties, Poisson

## Required data keys
- signal

## Modifies
- **signal**: uncertainties

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `with_processing_keys` | list | Yes | - | - | ProcessingData keys whose signal receives a named Poisson uncertainty. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
This is a simple Poisson uncertainty calculation based on the signal intensity

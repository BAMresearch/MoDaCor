# Flat-plate sample self-absorption correction

## Summary
Correct angle-dependent self-absorption for a transmission-normalized flat sample.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.flat_plate_self_absorption_correction.FlatPlateSelfAbsorptionCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/flat_plate_self_absorption_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/flat_plate_self_absorption_correction.py)
- **Module ID:** FlatPlateSelfAbsorptionCorrection
- **Module version:** 20260927.2
- **Keywords:** sample, self absorption, flat plate, transmission

## Required data keys
- signal
- CosAlpha

## Modifies
- **signal**: signal, uncertainties
- **flat_plate_self_absorption**: signal, uncertainties, units

## Required arguments
- with_processing_keys
- transmission_source

## Default configuration
```json
{
  "correction_key": "flat_plate_self_absorption",
  "cos_alpha_key": "CosAlpha",
  "minimum_attenuation_factor": 1e-12,
  "minimum_cos_alpha": 1e-12,
  "transmission_source": null,
  "transmission_uncertainties_sources": {},
  "transmission_units_source": null,
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `correction_key` | str | No | flat_plate_self_absorption | - | BaseData key used to store the relative attenuation factor. |
| `cos_alpha_key` | str | No | CosAlpha | - | BaseData key containing cos(2theta) for the sample exit angle. |
| `minimum_attenuation_factor` | int or float | No | 1e-12 | - | Lower allowed relative attenuation factor before division. |
| `minimum_cos_alpha` | int or float | No | 1e-12 | - | Lower numerical clip for positive exit-angle cosines. |
| `transmission_source` | str | Yes | - | - | IoSources key for the sample transmission factor. |
| `transmission_uncertainties_sources` | dict | No | {} | - | Mapping of uncertainty names to IoSources keys for sample transmission. |
| `transmission_units_source` | str or NoneType | No | - | - | Optional IoSources key for transmission units. |
| `with_processing_keys` | list | Yes | ["sample"] | - | ProcessingData keys whose signal should be corrected. |

## Notes
For s=1/cos(2theta), x=(s-1) ln(T), the relative attenuation is expm1(x)/x. The expression is the depth integral for scattering generated uniformly through a plate and is not the same as transmission through a separate downstream attenuator.

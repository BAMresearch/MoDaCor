# Solid Angle Correction

## Summary
Divide the pixels in a signal by their solid angle coverage

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.solid_angle_correction.SolidAngleCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/solid_angle_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/solid_angle_correction.py)
- **Module ID:** SolidAngleCorrection
- **Module version:** 20260927.1
- **Keywords:** divide, normalize, solid angle

## Required data keys
- signal
- Omega

## Modifies
- **signal**: signal, uncertainties, units

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
| `with_processing_keys` | list | Yes | - | - | ProcessingData keys whose signal should be divided by Omega. |

## References
DOI 10.1088/0953-8984/25/38/383201

## Notes
This divides the signal by the value previously calculated
            using XSGeometryFromPixelCoordinates

# Convert angle to signed Q

## Summary
Convert an angle relative to its direct-beam or diffraction centre into signed Q.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.angle_to_q.AngleToQ`
- **Source:** [`src/modacor/modules/technique_modules/scattering/angle_to_q.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/angle_to_q.py)
- **Module ID:** AngleToQ
- **Module version:** 20261003.1
- **Keywords:** scattering angle, two theta, Bragg angle, Q, momentum transfer, geometry

## Required data keys
- angle

## Modifies
- **Q**: signal, uncertainties, units, axes

## Required arguments
- with_processing_keys
- photon_source
- photon_units_source

## Default configuration
```json
{
  "angle_convention": "scattering_angle",
  "angle_key": "angle",
  "angle_zero": null,
  "angle_zero_units": null,
  "center_key": "beam_center",
  "output_key": "Q",
  "output_units": "1/nm",
  "photon_source": null,
  "photon_uncertainties_sources": {},
  "photon_units_source": null,
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `angle_convention` | str | No | scattering_angle | - | Angle convention: scattering_angle/two_theta uses sin(angle/2); bragg_angle/theta uses sin(angle). |
| `angle_key` | str | No | angle | - | Angular-coordinate BaseData key. |
| `angle_zero` | float or int or NoneType | No | - | - | Optional configured angle zero, overriding center_key. |
| `angle_zero_units` | str or NoneType | No | - | - | Units for angle_zero; defaults to the input angle units. |
| `center_key` | str or NoneType | No | beam_center | - | Angle-zero BaseData key used when angle_zero is None; None means zero angle. |
| `output_key` | str | No | Q | - | Output BaseData key. |
| `output_units` | str | No | 1/nm | - | Momentum-transfer output units. |
| `photon_source` | str or NoneType | Yes | - | - | IoSources key for scalar or angle-shaped photon energy or wavelength metadata. |
| `photon_uncertainties_sources` | dict | No | {} | - | Uncertainty sources for photon energy or wavelength metadata. |
| `photon_units_source` | str or NoneType | Yes | - | - | IoSources key for photon energy or wavelength units. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | DataBundle key or keys whose angular coordinates are converted. |

## Notes
Uses Q = 4*pi/lambda*sin(angle/2) for scattering_angle/two_theta and Q = 4*pi/lambda*sin(angle) for bragg_angle/theta. Photon metadata is loaded from IoSources; energy or wavelength representation is inferred from its units and converted with uncertainty-aware BaseData arithmetic. The angle sign is retained.

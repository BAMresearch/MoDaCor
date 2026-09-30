# Convert angle to signed Q

## Summary
Convert an angle relative to its direct-beam or diffraction centre into signed Q.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.angle_to_q.AngleToQ`
- **Source:** [`src/modacor/modules/technique_modules/scattering/angle_to_q.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/angle_to_q.py)
- **Module ID:** AngleToQ
- **Module version:** 20260930.1
- **Keywords:** scattering angle, two theta, Bragg angle, Q, momentum transfer, geometry

## Required data keys
- angle
- energy

## Modifies
- **Q**: signal, uncertainties, units, axes

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "angle_convention": "scattering_angle",
  "angle_key": "angle",
  "angle_zero": null,
  "angle_zero_units": null,
  "center_key": "beam_center",
  "incident_key": "energy",
  "incident_quantity": "energy",
  "output_key": "Q",
  "output_units": "1/nm",
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
| `incident_key` | str | No | energy | - | Scalar or angle-shaped photon-energy or wavelength BaseData key. |
| `incident_quantity` | str | No | energy | - | Quantity stored under incident_key: 'energy' or 'wavelength'. |
| `output_key` | str | No | Q | - | Output BaseData key. |
| `output_units` | str | No | 1/nm | - | Momentum-transfer output units. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | DataBundle key or keys whose angular coordinates are converted. |

## Notes
Uses Q = 4*pi/lambda*sin(angle/2) for scattering_angle/two_theta and Q = 4*pi/lambda*sin(angle) for bragg_angle/theta. Photon energy is converted to wavelength with uncertainty-aware BaseData arithmetic. The angle sign is retained.

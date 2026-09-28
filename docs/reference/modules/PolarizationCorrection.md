# Polarization correction

## Summary
Divide signal by the linear-polarization intensity factor.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.polarization_correction.PolarizationCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/polarization_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/polarization_correction.py)
- **Module ID:** PolarizationCorrection
- **Module version:** 20260927.2
- **Keywords:** polarization, polarisation, scattering, correction

## Required data keys
- signal
- TwoTheta
- Psi

## Modifies
- **signal**: signal, uncertainties
- **polarization_factor_map**: signal, units

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "correction_key": "polarization_factor_map",
  "minimum_polarization_factor": 1e-12,
  "mode": "linear_fraction",
  "polarisation_angular_offset": null,
  "polarisation_angular_offset_units": null,
  "polarisation_factor": null,
  "polarization_angular_offset": 0.0,
  "polarization_angular_offset_units": "degree",
  "polarization_factor": null,
  "psi_key": "Psi",
  "two_theta_key": "TwoTheta",
  "with_processing_keys": [
    "sample"
  ]
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `correction_key` | str | No | polarization_factor_map | - | BaseData key used to store the applied polarization factor. |
| `minimum_polarization_factor` | int or float | No | 1e-12 | - | Lower allowed polarization factor before division. |
| `mode` | str | No | linear_fraction | - | Polarization model. Currently supports 'linear_fraction'; 'stokes' is reserved. |
| `polarisation_angular_offset` | int or float or NoneType | No | - | - | British-spelling alias for polarization_angular_offset. |
| `polarisation_angular_offset_units` | str or NoneType | No | - | - | British-spelling alias for polarization_angular_offset_units. |
| `polarisation_factor` | int or float or NoneType | No | - | - | British-spelling alias for polarization_factor. |
| `polarization_angular_offset` | int or float | No | 0.0 | - | Angular offset between Psi=0 and the horizontal polarization axis. |
| `polarization_angular_offset_units` | str | No | degree | - | Units for polarization_angular_offset. |
| `polarization_factor` | int or float or NoneType | No | - | - | Fraction of horizontally polarized intensity. 0.5 is unpolarized. |
| `psi_key` | str | No | Psi | - | BaseData key containing the detector azimuth. |
| `two_theta_key` | str | No | TwoTheta | - | BaseData key containing the scattering angle 2theta. |
| `with_processing_keys` | list | Yes | ["sample"] | - | ProcessingData keys whose signal should be corrected. |

## Notes

            The implemented mode is DAWN/pyFAI-style linear_fraction:

              P = f * (1 - sin(2theta)^2 * cos(Psi - offset)^2)
                + (1 - f) * (1 - sin(2theta)^2 * sin(Psi - offset)^2)

            mode='stokes' is reserved for a future implementation once the
            Stokes reference-frame convention has been fixed.

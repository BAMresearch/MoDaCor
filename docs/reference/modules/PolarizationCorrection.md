# Polarization correction

## Summary
Divide signal by the linear-polarization intensity factor.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.polarization_correction.PolarizationCorrection`
- **Source:** [`src/modacor/modules/technique_modules/scattering/polarization_correction.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/polarization_correction.py)
- **Module ID:** PolarizationCorrection
- **Module version:** 20261030.2
- **Keywords:** polarization, polarisation, scattering, correction

## Required data keys
- signal
- TwoTheta
- Psi

## Modifies
- **signal**: signal, uncertainties
- **polarization_factor_map**: signal, units

## Data dependencies
| Fixed key | Access |
|---|---|
| `signal` | read_write |

| Role argument | Access | Default |
|---|---|---|
| `correction_key` | write | `"polarization_factor_map"` |
| `psi_key` | read | `"Psi"` |
| `two_theta_key` | read | `"TwoTheta"` |

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "correction_key": "polarization_factor_map",
  "minimum_polarization_factor": 1e-12,
  "mode": "linear_fraction",
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
| `correction_key` | str | No | polarization_factor_map | processing_write_basedata_key | BaseData key used to store the applied polarization factor. |
| `minimum_polarization_factor` | int or float | No | 1e-12 | - | Lower allowed polarization factor before division. |
| `mode` | str | No | linear_fraction | - | Polarization model. Currently supports 'linear_fraction'; 'stokes' is reserved. |
| `polarization_angular_offset` | int or float | No | 0.0 | - | Angular offset between Psi=0 and the horizontal polarization axis. |
| `polarization_angular_offset_units` | str | No | degree | - | Units for polarization_angular_offset. |
| `polarization_factor` | int or float or NoneType | No | - | - | Fraction of horizontally polarized intensity. Defaults to 0.5 (unpolarized) when unset or None. |
| `psi_key` | str | No | Psi | processing_read_basedata_key | BaseData key containing the detector azimuth. |
| `two_theta_key` | str | No | TwoTheta | processing_read_basedata_key | BaseData key containing the scattering angle 2theta. |
| `with_processing_keys` | list | Yes | ["sample"] | - | ProcessingData keys whose signal should be corrected. |

## Notes

            The implemented mode is DAWN/pyFAI-style linear_fraction:

              P = f * (1 - sin(2theta)^2 * cos(Psi - offset)^2)
                + (1 - f) * (1 - sin(2theta)^2 * sin(Psi - offset)^2)

            mode='stokes' is reserved for a future implementation once the
            Stokes reference-frame convention has been fixed.

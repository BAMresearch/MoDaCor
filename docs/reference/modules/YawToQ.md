# Convert analyser yaw to signed Q

## Summary
Convert analyser yaw relative to its direct-beam centre into signed Q.

## Metadata
- **Import path:** `modacor.modules.technique_modules.scattering.yaw_to_q.YawToQ`
- **Source:** [`src/modacor/modules/technique_modules/scattering/yaw_to_q.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/technique_modules/scattering/yaw_to_q.py)
- **Module ID:** YawToQ
- **Module version:** 20260929.1
- **Keywords:** USAXS, yaw, Q, momentum transfer, geometry

## Required data keys
- yaw
- energy

## Modifies
- **Q**: signal, uncertainties, units, axes

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "center_key": "beam_center",
  "energy_key": "energy",
  "output_key": "Q",
  "output_units": "1/nm",
  "with_processing_keys": null,
  "yaw_key": "yaw",
  "yaw_zero": null,
  "yaw_zero_units": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `center_key` | str or NoneType | No | beam_center | - | Yaw-zero BaseData key used when yaw_zero is None; None means zero yaw. |
| `energy_key` | str | No | energy | - | Scalar or yaw-shaped photon-energy BaseData key. |
| `output_key` | str | No | Q | - | Output BaseData key. |
| `output_units` | str | No | 1/nm | - | Momentum-transfer output units. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | DataBundle key or keys whose yaw coordinates are converted. |
| `yaw_key` | str | No | yaw | - | Analyser-yaw BaseData key. |
| `yaw_zero` | float or int or NoneType | No | - | - | Optional configured yaw zero, overriding center_key. |
| `yaw_zero_units` | str or NoneType | No | - | - | Units for yaw_zero; defaults to yaw units. |

## Notes
Uses Q = 4*pi/lambda*sin((yaw-yaw_zero)/2), with wavelength derived from measured photon energy. The sign is retained so the negative and positive analyser wings remain distinguishable.

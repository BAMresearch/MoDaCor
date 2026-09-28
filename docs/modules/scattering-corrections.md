# Scattering corrections

Technique-specific corrections encode physical assumptions that must match the
experiment.

- [`DetectorEfficiencyCorrection`](../reference/modules/DetectorEfficiencyCorrection.md)
  corrects detector response using material and beam-energy information.
- [`AttenuatorPlateCorrection`](../reference/modules/AttenuatorPlateCorrection.md)
  corrects configured attenuator transmission.
- [`PolarizationCorrection`](../reference/modules/PolarizationCorrection.md)
  applies the documented beam-polarization geometry.
- [`SolidAngleCorrection`](../reference/modules/SolidAngleCorrection.md)
  accounts for detector-pixel solid angle.
- [`FlatPlateSelfAbsorptionCorrection`](../reference/modules/FlatPlateSelfAbsorptionCorrection.md)
  models attenuation in flat samples.
- [`CapillarySelfAbsorptionCorrection`](../reference/modules/CapillarySelfAbsorptionCorrection.md)
  and [`CapillarySampleContainerCorrection`](../reference/modules/CapillarySampleContainerCorrection.md)
  model cylindrical sample and wall attenuation. See the detailed
  [capillary guide](capillary-self-absorption.md).

The generated pages define exact inputs; they do not decide whether a correction
is scientifically appropriate. Record the origin and uncertainty of thickness,
attenuation, polarization, detector, beam-profile, and geometry parameters near
the pipeline configuration. Validate correction maps and limiting cases before
using a pipeline for reference results.

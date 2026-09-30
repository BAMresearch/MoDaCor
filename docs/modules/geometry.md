# Geometry

Geometry modules translate detector indices and instrument metadata into
physical coordinates used by corrections and integration.

[`IndexPixels`](../reference/modules/IndexPixels.md) creates detector-index maps.
`Psi` is optional for pure one-dimensional Q binning when no azimuthal region
of interest is configured.
[`PixelCoordinates3D`](../reference/modules/PixelCoordinates3D.md) uses explicit
or NeXus-derived detector frames to locate pixel centers in laboratory
coordinates. [`XSGeometryFromPixelCoordinates`](../reference/modules/XSGeometryFromPixelCoordinates.md)
derives scattering quantities such as `Q`, azimuth, and solid angle from those
positions and beam metadata.

[`YawToQ`](../reference/modules/YawToQ.md) converts analyser yaw to signed Q
using a configured or measured beam centre and photon energy. Keeping the sign
allows asymmetric analyser wings to remain separate until a pipeline selects
one or combines them deliberately.

Prefer NeXus transformation chains when the input describes them correctly.
Explicit basis vectors, pitches, beam centers, or sample positions remain
available for non-NeXus data and reviewed overrides.

Detector indices are dimensionless; pixel pitches and coordinates are lengths.
Preserve that distinction. Build static detector geometry once when possible
and reuse it across measurements, while keeping measurement-dependent sample or
beam metadata in its appropriate source.

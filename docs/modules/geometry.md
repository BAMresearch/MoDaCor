# Geometry

Geometry modules translate detector indices and instrument metadata into
physical coordinates used by corrections and integration.

[`PixelCoordinates3D`](../reference/modules/PixelCoordinates3D.md) uses explicit
or NeXus-derived detector frames to locate pixel centers in laboratory
coordinates. [`XSGeometryFromPixelCoordinates`](../reference/modules/XSGeometryFromPixelCoordinates.md)
derives scattering quantities such as `Q`, azimuth, and solid angle from those
positions and beam metadata.

[`AngleToQ`](../reference/modules/AngleToQ.md) converts scattering-angle or
Bragg-angle coordinates to signed Q using a configured or measured centre and
photon metadata loaded from an IO source. `AngleToQ` and
[`XSGeometryFromPixelCoordinates`](../reference/modules/XSGeometryFromPixelCoordinates.md)
share the same `photon_source`, `photon_units_source`, and
`photon_uncertainties_sources` interface. Photon energy versus wavelength is
inferred from Pint dimensionality, and energy-to-wavelength conversion uses
`BaseData` arithmetic so units and separate uncertainty components propagate.
Keeping the sign allows asymmetric analyser wings to remain separate until a
pipeline selects one or combines them deliberately.

This is a breaking configuration change for
`XSGeometryFromPixelCoordinates`: its former `wavelength_source`,
`wavelength_units_source`, and `wavelength_uncertainties_sources` fields are no
longer accepted. See the
[interface migration guide](../processing/interface-migrations.md) for the
direct replacements. Wavelength-specific fields on other correction modules
are unaffected.

Prefer NeXus transformation chains when the input describes them correctly.
Explicit basis vectors, pitches, beam centers, or sample positions remain
available for non-NeXus data and reviewed overrides.

Detector indices are dimensionless; pixel pitches and coordinates are lengths.
Preserve that distinction. Build static detector geometry once when possible
and reuse it across measurements, while keeping measurement-dependent sample or
beam metadata in its appropriate source.

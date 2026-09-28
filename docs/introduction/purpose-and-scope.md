# Purpose and scope

MoDaCor is a modular data-correction framework for scientific measurements. It
is intended for workflows in which the corrected values alone are insufficient:
the units, uncertainty contributions, processing history, and assumptions must
remain inspectable as well.

Its present application scope is monochromatic X-ray and neutron techniques
that produce scalar, one-dimensional, or two-dimensional data, including
scattering, diffraction, and imaging. The data model is technique-neutral;
technique-specific modules supply detector geometry and physical corrections.

MoDaCor can be used in two ways:

- as the primary correction system for reference-quality results; or
- as a transparent reference implementation for validating a faster, more
  tightly integrated beamline or instrument pipeline.

MoDaCor deliberately prioritizes correction quality, traceability, and
scientific review over maximum event throughput. It is not an acquisition or
instrument-control system, and it does not replace facility authentication,
storage policy, or experiment-specific scientific judgment.

The modular workflow follows the concepts described by
[Pauw et al. (2017)](https://doi.org/10.1107/S1600576717015096).

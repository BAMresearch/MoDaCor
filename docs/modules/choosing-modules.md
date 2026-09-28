# Choosing modules

Build a correction pipeline from the physical measurement model, not from the
alphabetical module list.

1. Load measured and calibration quantities into stable `ProcessingData`
   bundles.
2. Create uncertainties while the signal is still in the domain in which their
   model is valid.
3. Construct detector and scattering geometry once and reuse it.
4. Create and combine masks while preserving reason bits.
5. Apply detector, transmission, polarization, solid-angle, background, and
   sample-container corrections in an order justified by the experiment.
6. Reduce or integrate only after pixel-level corrections that require detector
   geometry.
7. Publish results and provenance through a sink.

This is a starting pattern, not a universal scientific ordering. A correction's
guide and reference page state its assumptions. When two steps operate on the
same physical quantity, check their units, named uncertainties, masks, and
dependency contracts as well as their graph order.

The [generated catalogue](../reference/modules/index.md) groups every supported
public step by function and also provides an alphabetical index.

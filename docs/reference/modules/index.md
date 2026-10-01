# Process-step reference

Every supported public `ProcessStep` is listed below by function and alphabetically.
Configuration tables are generated from each step's `ProcessStepDescriber` metadata.

## Data movement and copying

- [AppendProcessingData](AppendProcessingData.md)
- [AppendSink](AppendSink.md)
- [AppendSource](AppendSource.md)
- [ConcatenateDatabundles](ConcatenateDatabundles.md)
- [CopyDataBundleKeys](CopyDataBundleKeys.md)
- [SinkProcessingData](SinkProcessingData.md)

## Arithmetic and normalization

- [Divide](Divide.md)
- [DivideDatabundles](DivideDatabundles.md)
- [FindScaleFactor1D](FindScaleFactor1D.md)
- [Multiply](Multiply.md)
- [MultiplyDatabundles](MultiplyDatabundles.md)
- [Negate](Negate.md)
- [Subtract](Subtract.md)
- [SubtractDatabundles](SubtractDatabundles.md)
- [SubtractInterpolated1D](SubtractInterpolated1D.md)
- [UnitsLabelUpdate](UnitsLabelUpdate.md)

## Masks

- [ApplyMask](ApplyMask.md)
- [BitwiseOrMasks](BitwiseOrMasks.md)
- [DilateMask](DilateMask.md)
- [ReduceMask](ReduceMask.md)
- [ThresholdMask](ThresholdMask.md)

## Uncertainty creation and combination

- [CombineUncertainties](CombineUncertainties.md)
- [CombineUncertaintiesMax](CombineUncertaintiesMax.md)
- [PoissonUncertainties](PoissonUncertainties.md)

## Geometry and coordinates

- [PixelCoordinates3D](PixelCoordinates3D.md)
- [XSGeometryFromPixelCoordinates](XSGeometryFromPixelCoordinates.md)
- [AngleToQ](AngleToQ.md)

## Reduction and integration

- [FindCenterOfMass1D](FindCenterOfMass1D.md)
- [IndexByCoordinate](IndexByCoordinate.md)
- [IndexedAverager](IndexedAverager.md)
- [Integrate1D](Integrate1D.md)
- [ReduceDimensionality](ReduceDimensionality.md)

## Visualization

- [Plot1DVisualization](Plot1DVisualization.md)
- [Plot2DVisualization](Plot2DVisualization.md)

## Technique-specific corrections

- [AttenuatorPlateCorrection](AttenuatorPlateCorrection.md)
- [CapillarySampleContainerCorrection](CapillarySampleContainerCorrection.md)
- [CapillarySelfAbsorptionCorrection](CapillarySelfAbsorptionCorrection.md)
- [DetectorEfficiencyCorrection](DetectorEfficiencyCorrection.md)
- [FlatPlateSelfAbsorptionCorrection](FlatPlateSelfAbsorptionCorrection.md)
- [PolarizationCorrection](PolarizationCorrection.md)
- [SolidAngleCorrection](SolidAngleCorrection.md)

## Alphabetical index

```{toctree}
:maxdepth: 1

AngleToQ
AppendProcessingData
AppendSink
AppendSource
ApplyMask
AttenuatorPlateCorrection
BitwiseOrMasks
CapillarySampleContainerCorrection
CapillarySelfAbsorptionCorrection
CombineUncertainties
CombineUncertaintiesMax
ConcatenateDatabundles
CopyDataBundleKeys
DetectorEfficiencyCorrection
DilateMask
Divide
DivideDatabundles
FindCenterOfMass1D
FindScaleFactor1D
FlatPlateSelfAbsorptionCorrection
IndexByCoordinate
IndexedAverager
Integrate1D
Multiply
MultiplyDatabundles
Negate
PixelCoordinates3D
Plot1DVisualization
Plot2DVisualization
PoissonUncertainties
PolarizationCorrection
ReduceDimensionality
ReduceMask
SinkProcessingData
SolidAngleCorrection
Subtract
SubtractDatabundles
SubtractInterpolated1D
ThresholdMask
UnitsLabelUpdate
XSGeometryFromPixelCoordinates
```

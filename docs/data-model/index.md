# Data model

MoDaCor uses three nested containers so numerical values, physical meaning, and
pipeline organization remain explicit:

```text
ProcessingData
└── "sample"                         DataBundle
    ├── "signal"                     BaseData
    │   ├── signal                    numerical array
    │   ├── units                     Pint unit
    │   ├── uncertainties             named absolute 1σ arrays
    │   ├── weights and axes
    │   └── rank_of_data
    ├── "Q"                          BaseData
    └── "mask"                       BaseData
```

```{toctree}
:maxdepth: 1

basedata
databundle
processingdata
units-and-dimensionality
uncertainty-propagation
paths-axes-weights-and-masks
```

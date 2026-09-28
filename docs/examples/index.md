# Examples

Start with the self-contained [Quickstart](../getting-started/quickstart.md).
The two examples in this manual then show the same processing model under local
and client-server control:

```{toctree}
:maxdepth: 1

local-operation
remote-server-operation
```

## Instrument examples

[MoDaCor-examples](https://github.com/BAMResearch/MoDaCor-examples) is the
canonical repository for instrument notebooks, pipeline YAML, data manifests,
download/verification tools, and facility-specific guidance. Instrument assets
are not duplicated in the MoDaCor package.

| Facility and instrument | Workflow | Catalogue |
| --- | --- | --- |
| BAM MOUSE | SAXS solids, backgrounds, operando patterns | [BAM/MOUSE](https://github.com/BAMResearch/MoDaCor-examples/tree/main/BAM/MOUSE) |
| BAM SAXSess I | Absolute-intensity SAXS | [BAM/SAXSess_I](https://github.com/BAMResearch/MoDaCor-examples/tree/main/BAM/SAXSess_I) |
| Diamond I22 | SAXS/WAXS and chunked Buffer/HDF/Tiled operation | [DLS/I22](https://github.com/BAMResearch/MoDaCor-examples/tree/main/DLS/I22) |
| Diamond B21 | Frame quality and chunked processing design | [DLS/B21](https://github.com/BAMResearch/MoDaCor-examples/tree/main/DLS/B21) |

Read each external example's status and data manifest before downloading data.
Development entries may exist before a versioned dataset release. Released
examples should pin a compatible MoDaCor release or commit.

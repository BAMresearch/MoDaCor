# Sink Processing Data

## Summary
Write selected ProcessingData leaves to an IoSink.

## Metadata
- **Import path:** `modacor.modules.base_modules.sink_processing_data.SinkProcessingData`
- **Source:** [`src/modacor/modules/base_modules/sink_processing_data.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/sink_processing_data.py)
- **Module ID:** SinkProcessingData
- **Module version:** 20260901.1
- **Keywords:** sink, export, write

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- target
- data_paths

## Default configuration
```json
{
  "data_paths": [],
  "target": ""
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `data_paths` | str or list | Yes | [] | - | ProcessingData paths to write (string or list of strings). |
| `target` | str | Yes |  | - | Sink target in the form 'sink_id::subpath'. |

## Notes
This step performs an export side-effect and returns an empty output dict.

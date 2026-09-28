# Append Source

## Summary
Append an ioSource to the available data sources

## Metadata
- **Import path:** `modacor.modules.base_modules.append_source.AppendSource`
- **Source:** [`src/modacor/modules/base_modules/append_source.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/append_source.py)
- **Module ID:** AppendSource
- **Module version:** 20260927.1
- **Keywords:** append, source

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- source_identifier
- source_location
- iosource_module

## Default configuration
```json
{
  "iosource_method_kwargs": {},
  "iosource_module": "",
  "source_identifier": "",
  "source_location": ""
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `iosource_method_kwargs` | dict | No | {} | - | Keyword arguments forwarded to the ioSource constructor. |
| `iosource_module` | str | Yes |  | - | Fully qualified module path to the ioSource class. |
| `source_identifier` | str or list | Yes |  | - | Identifier(s) to register the ioSource(s) under. |
| `source_location` | str or list | Yes |  | - | Resource location(s) to load for the ioSource(s). |

## Notes
This adds an ioSource to the data sources of the databundle.

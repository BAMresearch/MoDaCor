# Negate BaseData values

## Summary
Negate a BaseData entry using its uncertainty-preserving unary operation.

## Metadata
- **Import path:** `modacor.modules.base_modules.negate.Negate`
- **Source:** [`src/modacor/modules/base_modules/negate.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/negate.py)
- **Module ID:** Negate
- **Module version:** 20260929.1
- **Keywords:** negate, sign, BaseData

## Required data keys
- _None_

## Modifies
- **configured data key**: signal

## Required arguments
- with_processing_keys

## Default configuration
```json
{
  "data_key": "signal",
  "with_processing_keys": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `data_key` | str | No | signal | - | BaseData key whose nominal values are negated. |
| `with_processing_keys` | str or list or NoneType | Yes | - | - | DataBundle key or keys to update. |

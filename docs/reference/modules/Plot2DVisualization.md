# Plot 2D Visualization

## Summary
Publish a Plotly-compatible 2D heatmap payload.

## Metadata
- **Import path:** `modacor.modules.base_modules.plot_2d_visualization.Plot2DVisualization`
- **Source:** [`src/modacor/modules/base_modules/plot_2d_visualization.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/plot_2d_visualization.py)
- **Module ID:** Plot2DVisualization
- **Module version:** 20260909.1
- **Keywords:** plot, visualization, plotly, 2d, image

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- target
- data_path

## Default configuration
```json
{
  "auto_zmax_percentile": 99.0,
  "auto_zmin_percentile": 5.0,
  "colormap": "Plasma",
  "data_path": "",
  "reverse_y": true,
  "scale": "log10",
  "target": "",
  "title": null,
  "transpose": false,
  "uirevision": null,
  "zmax": null,
  "zmin": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `auto_zmax_percentile` | int or float | No | 99.0 | - | Percentile used for automatic zmax on finite, unmasked pixels in the displayed frame. |
| `auto_zmin_percentile` | int or float | No | 5.0 | - | Percentile used for automatic zmin on finite, unmasked pixels in the displayed frame. |
| `colormap` | str | No | Plasma | - | Plotly colorscale name. Use the 'colormap' key and Plotly's canonical capitalization. |
| `data_path` | str | Yes |  | - | ProcessingData path for the 2D array or BaseData root. |
| `reverse_y` | bool | No | True | - | Reverse the y axis for detector-image style display. |
| `scale` | str | No | log10 | - | Color scaling: 'log10' or 'linear'. |
| `target` | str | Yes |  | - | Sink target in the form 'plot_sink::plot_id'. |
| `title` | str or NoneType | No | - | - | Optional plot title. |
| `transpose` | bool | No | False | - | Transpose the displayed 2D frame. |
| `uirevision` | str or NoneType | No | - | - | Stable Plotly UI revision key. Keep unchanged to preserve zoom/pan during live updates. |
| `zmax` | int or float or NoneType | No | - | - | Optional upper color scale bound. |
| `zmin` | int or float or NoneType | No | - | - | Optional lower color scale bound. |

## Notes
Higher-dimensional data is sliced to the first frame over leading dimensions. Automatic color bounds default to the 5th and 99th percentiles of finite pixels; log10 scaling uses only positive finite pixels.

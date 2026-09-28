# Plot 1D Visualization

## Summary
Publish a Plotly-compatible 1D data plot payload.

## Metadata
- **Import path:** `modacor.modules.base_modules.plot_1d_visualization.Plot1DVisualization`
- **Source:** [`src/modacor/modules/base_modules/plot_1d_visualization.py`](https://github.com/BAMresearch/MoDaCor/blob/main/src/modacor/modules/base_modules/plot_1d_visualization.py)
- **Module ID:** Plot1DVisualization
- **Module version:** 20260903.1
- **Keywords:** plot, visualization, plotly, 1d

## Required data keys
- _None_

## Modifies
- _None_

## Required arguments
- target
- x_path
- y_path

## Default configuration
```json
{
  "auto_log_x": true,
  "auto_log_y": true,
  "error_bar_opacity": 0.65,
  "error_bar_thickness": 2.0,
  "error_bar_width": 3,
  "legend_marker_size": 9,
  "target": "",
  "title": null,
  "uirevision": null,
  "x_path": "",
  "x_units": null,
  "xerr_path": null,
  "xerr_uncertainty_names": null,
  "y_path": "",
  "y_units": null,
  "yerr_path": null,
  "yerr_uncertainty_names": null
}
```

## Argument specification
| Argument | Type | Required | Default | Dependency role | Description |
|---|---|---|---|---|---|
| `auto_log_x` | bool | No | True | - | Use a logarithmic x axis when all valid x values are positive. |
| `auto_log_y` | bool | No | True | - | Use a logarithmic y axis when all valid y values are positive. |
| `error_bar_opacity` | int or float | No | 0.65 | - | Error bar opacity encoded in the error bar colour. |
| `error_bar_thickness` | int or float | No | 2.0 | - | Plotly error bar line thickness. |
| `error_bar_width` | int or float | No | 3 | - | Plotly error bar cap width. |
| `legend_marker_size` | int or float | No | 9 | - | Marker size for legend-only uncertainty entries. |
| `target` | str | Yes |  | - | Sink target in the form 'plot_sink::plot_id'. |
| `title` | str or NoneType | No | - | - | Optional plot title. |
| `uirevision` | str or NoneType | No | - | - | Stable Plotly UI revision key. Keep unchanged to preserve zoom/pan during live updates. |
| `x_path` | str | Yes |  | - | ProcessingData path for the x array. |
| `x_units` | str or NoneType | No | - | - | Optional display units for x values and x error bars. |
| `xerr_path` | str or NoneType | No | - | - | Optional ProcessingData path for x error bars. |
| `xerr_uncertainty_names` | list or str or NoneType | No | - | - | Optional uncertainty-name fallbacks below the x BaseData. |
| `y_path` | str | Yes |  | - | ProcessingData path for the y array. |
| `y_units` | str or NoneType | No | - | - | Optional display units for y values and y error bars. |
| `yerr_path` | str or NoneType | No | - | - | Optional ProcessingData path for y error bars. |
| `yerr_uncertainty_names` | list or str or NoneType | No | - | - | Optional uncertainty-name fallbacks below the y BaseData. |

## Notes
This step performs a visualization side-effect and returns an empty output dict.

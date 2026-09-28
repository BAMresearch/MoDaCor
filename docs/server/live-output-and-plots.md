# Live output and plots

Live plots are sink outputs. Plot modules publish Plotly-compatible JSON through
a registered visualization sink; the server exposes the latest plot document at
a stable session URL.

Install the plotting extra:

```bash
uv pip install "modacor[server,plotting]"
```

After registering the configured sink and running the pipeline, obtain a URL
with:

```python
plot_url = session.plot_url("live", "corrected-intensity")
```

Clients may poll that resource or use runtime events to decide when a new run is
available. A plot is a derived view, not the authoritative scientific output;
write corrected data and provenance through an appropriate persistent sink.

Keep plot payloads small enough for interactive use. Reduce or sample large
detector arrays before visualization rather than making plotting a hidden
high-volume transport.

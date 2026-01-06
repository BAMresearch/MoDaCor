# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "13/12/2025"
__status__ = "Development"  # "Development", "Production"
__version__ = "20251213.2"

from typing import Any, Protocol

import numpy as np
from attrs import define, field

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.processing_data import ProcessingData

# --- Rendering ---------------------------------------------------------------


UNICODE = {
    "tee": "├─",
    "elbow": "└─",
    "pipe": "│ ",
    "space": "  ",
    "bullet": "•",
    "changed": "🟪",  # purple square
    "same": "🟩",  # green square
}

ANSI = {
    "reset": "\x1b[0m",
    "green": "\x1b[32m",
    "purple": "\x1b[35m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
}


class ReportRenderer(Protocol):
    def header(self, text: str) -> str:
        ...

    def dim(self, text: str) -> str:
        ...

    def ok(self, text: str) -> str:
        ...

    def changed(self, text: str) -> str:
        ...

    def badge_ok(self) -> str:
        ...

    def badge_changed(self) -> str:
        ...

    def codewrap(self, text: str) -> str:
        ...


@define(frozen=True)
class PlainUnicodeRenderer:
    """Unicode tree + emoji badges; no colors."""

    wrap_in_markdown_codeblock: bool = False

    def header(self, text: str) -> str:
        return text

    def dim(self, text: str) -> str:
        return text

    def ok(self, text: str) -> str:
        return text

    def changed(self, text: str) -> str:
        return text

    def badge_ok(self) -> str:
        return UNICODE["same"]

    def badge_changed(self) -> str:
        return UNICODE["changed"]

    def codewrap(self, text: str) -> str:
        if not self.wrap_in_markdown_codeblock:
            return text
        return "```text\n" + text + "\n```"


@define(frozen=True)
class AnsiUnicodeRenderer:
    """Unicode tree + ANSI colors (green unchanged, purple changed)."""

    wrap_in_markdown_codeblock: bool = False
    enable: bool = True

    def _c(self, text: str, color: str) -> str:
        if not self.enable:
            return text
        return f"{ANSI[color]}{text}{ANSI['reset']}"

    def header(self, text: str) -> str:
        return self._c(text, "bold")

    def dim(self, text: str) -> str:
        return self._c(text, "dim")

    def ok(self, text: str) -> str:
        return self._c(text, "green")

    def changed(self, text: str) -> str:
        return self._c(text, "purple")

    def badge_ok(self) -> str:
        return self.ok(UNICODE["same"])

    def badge_changed(self) -> str:
        return self.changed(UNICODE["changed"])

    def codewrap(self, text: str) -> str:
        if not self.wrap_in_markdown_codeblock:
            return text
        return "```text\n" + text + "\n```"


@define(frozen=True)
class MarkdownCssRenderer:
    """
    Markdown + HTML spans for styling via CSS classes.
    Works well in MkDocs / Sphinx / Jupyter HTML outputs.
    (GitHub won't apply your custom CSS, but text still reads fine.)
    """

    wrap_in_markdown_codeblock: bool = False

    def header(self, text: str) -> str:
        return f"**{text}**"

    def dim(self, text: str) -> str:
        return f"<span class='mdc-dim'>{text}</span>"

    def ok(self, text: str) -> str:
        return f"<span class='mdc-ok'>{text}</span>"

    def changed(self, text: str) -> str:
        return f"<span class='mdc-changed'>{text}</span>"

    def badge_ok(self) -> str:
        return self.ok(UNICODE["same"])

    def badge_changed(self) -> str:
        return self.changed(UNICODE["changed"])

    def codewrap(self, text: str) -> str:
        # Prefer <pre> so CSS can style spans inside; code fences often strip HTML
        if self.wrap_in_markdown_codeblock:
            return "```text\n" + text + "\n```"
        return "<pre class='mdc-pre'>\n" + text + "\n</pre>"


def _nan_count(x: np.ndarray) -> int:
    return int(np.isnan(x).sum()) if x.size else 0


def _finite_min_max(x: np.ndarray) -> tuple[float | None, float | None]:
    if x.size == 0 or not np.isfinite(x).any():
        return None, None
    return float(np.nanmin(x)), float(np.nanmax(x))


@define(frozen=True)
class BaseDataProbe:
    """
    A tiny, array-free fingerprint of a BaseData at a point in the pipeline.

    Notes:
    - `ndim` is derived from the signal array.
    - `rank_of_data` is taken from BaseData metadata if present.
    - `dimensionality_str` is Pint dimensionality (not shape dimensions).
    """

    shape: tuple[int, ...]
    ndim: int
    rank_of_data: int | None

    units_str: str
    dimensionality_str: str

    nan_signal: int
    nan_unc: dict[str, int] = field(factory=dict)

    # Optional scalar diagnostics
    min_signal: float | None = None
    max_signal: float | None = None

    @classmethod
    def from_basedata(cls, bd: BaseData, *, compute_min_max: bool = False) -> "BaseDataProbe":
        sig = np.asarray(bd.signal, dtype=float)

        nan_unc = {k: _nan_count(np.asarray(v, dtype=float)) for k, v in bd.uncertainties.items()}

        # Pint dimensionality is often more robust than raw unit string equality
        dimensionality = getattr(bd.units, "dimensionality", None)
        dimensionality_str = str(dimensionality) if dimensionality is not None else "<?>"

        smin: float | None = None
        smax: float | None = None
        if compute_min_max:
            smin, smax = _finite_min_max(sig)

        return cls(
            shape=tuple(sig.shape),
            ndim=int(sig.ndim),
            rank_of_data=getattr(bd, "rank_of_data", None),
            units_str=str(bd.units),
            dimensionality_str=dimensionality_str,
            nan_signal=_nan_count(sig),
            nan_unc=nan_unc,
            min_signal=smin,
            max_signal=smax,
        )


@define
class PipelineTracer:
    """
    Records only *small* per-step probes, optionally only when relevant deltas occur.

    Example watch:
        {"sample": ["signal"], "sample_background": ["signal"]}
    """

    watch: dict[str, list[str]] = field(factory=dict)

    # Keep history small by default: only record when deltas occur (as defined by change_kinds)
    record_only_on_change: bool = True
    record_empty_step_events: bool = False

    # Which changes trigger recording an event (min/max are *not* triggers by default)
    change_kinds: set[str] = field(
        factory=lambda: {
            "units",
            "dimensionality",
            "shape",
            "ndim",
            "rank",
            "nan_signal",
            "nan_unc",
        }
    )

    # Include scalar min/max in probes (does not affect change detection unless you add "minmax" to change_kinds)
    compute_min_max: bool = False

    # Guards (fail fast at the *first* step that introduces the issue)
    fail_on_expected_mismatch: bool = False
    fail_on_nan_increase: bool = False
    fail_on_units_change: bool = False
    fail_on_dimensionality_change: bool = False
    fail_on_shape_change: bool = False
    fail_on_rank_change: bool = False

    # Optional expectations: step_id -> expected value
    expected_units_by_step: dict[str, str] = field(factory=dict)
    expected_dimensionality_by_step: dict[str, str] = field(factory=dict)
    expected_ndim_by_step: dict[str, int] = field(factory=dict)
    expected_rank_by_step: dict[str, int] = field(factory=dict)

    _last: dict[tuple[str, str], BaseDataProbe] = field(factory=dict)
    events: list[dict[str, Any]] = field(factory=list)

    def _diff_kinds(self, prev: BaseDataProbe, now: BaseDataProbe) -> set[str]:
        kinds: set[str] = set()

        if "units" in self.change_kinds and now.units_str != prev.units_str:
            kinds.add("units")
        if "dimensionality" in self.change_kinds and now.dimensionality_str != prev.dimensionality_str:
            kinds.add("dimensionality")
        if "shape" in self.change_kinds and now.shape != prev.shape:
            kinds.add("shape")
        if "ndim" in self.change_kinds and now.ndim != prev.ndim:
            kinds.add("ndim")
        if "rank" in self.change_kinds and now.rank_of_data != prev.rank_of_data:
            kinds.add("rank")

        if "nan_signal" in self.change_kinds and now.nan_signal != prev.nan_signal:
            kinds.add("nan_signal")

        if "nan_unc" in self.change_kinds:
            keys = set(prev.nan_unc) | set(now.nan_unc)
            if any(now.nan_unc.get(k, 0) != prev.nan_unc.get(k, 0) for k in keys):
                kinds.add("nan_unc")

        # Optional: treat min/max as a trigger if explicitly requested
        if "minmax" in self.change_kinds and (now.min_signal, now.max_signal) != (prev.min_signal, prev.max_signal):
            kinds.add("minmax")

        return kinds

    def after_step(  # noqa: C901 # too complex, resolve later
        self,
        step: ProcessStep,
        data: ProcessingData,
        *,
        duration_s: float | None = None,
    ) -> None:
        step_id = getattr(step, "step_id", "<??>")
        module = getattr(step.documentation, "calling_id", None) or step.__class__.__name__
        name = getattr(step.documentation, "calling_name", "")

        changed: dict[tuple[str, str], dict[str, Any]] = {}

        for bundle_key, ds_keys in self.watch.items():
            if bundle_key not in data:
                continue

            db = data[bundle_key]
            for ds_key in ds_keys:
                if ds_key not in db:
                    continue

                bd = db[ds_key]
                if not isinstance(bd, BaseData):
                    continue

                now = BaseDataProbe.from_basedata(bd, compute_min_max=self.compute_min_max)
                prev = self._last.get((bundle_key, ds_key))
                self._last[(bundle_key, ds_key)] = now

                # Expectations (exact string/int match)
                exp_units = self.expected_units_by_step.get(step_id)
                if exp_units is not None and now.units_str != exp_units and self.fail_on_expected_mismatch:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} units mismatch: "
                        f"got '{now.units_str}', expected '{exp_units}'"
                    )

                exp_dim = self.expected_dimensionality_by_step.get(step_id)
                if exp_dim is not None and now.dimensionality_str != exp_dim and self.fail_on_expected_mismatch:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} dimensionality mismatch: "
                        f"got '{now.dimensionality_str}', expected '{exp_dim}'. units='{now.units_str}'"
                    )

                exp_ndim = self.expected_ndim_by_step.get(step_id)
                if exp_ndim is not None and now.ndim != exp_ndim and self.fail_on_expected_mismatch:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} ndim mismatch: got {now.ndim}, expected {exp_ndim}"
                    )

                exp_rank = self.expected_rank_by_step.get(step_id)
                if exp_rank is not None and now.rank_of_data != exp_rank and self.fail_on_expected_mismatch:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} rank_of_data mismatch: "
                        f"got {now.rank_of_data}, expected {exp_rank}"
                    )

                # Delta-driven recording / guards
                if prev is None:
                    # Always record first probe for a watched target
                    changed[(bundle_key, ds_key)] = {"prev": None, "now": now, "diff": {"first_seen"}}
                    continue

                diff = self._diff_kinds(prev, now)

                if self.fail_on_units_change and now.units_str != prev.units_str:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} units changed: "
                        f"'{prev.units_str}' -> '{now.units_str}'"
                    )
                if self.fail_on_dimensionality_change and now.dimensionality_str != prev.dimensionality_str:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} dimensionality changed: "
                        f"'{prev.dimensionality_str}' -> '{now.dimensionality_str}' (units='{now.units_str}')"
                    )
                if self.fail_on_shape_change and now.shape != prev.shape:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} shape changed: {prev.shape} -> {now.shape}"
                    )
                if self.fail_on_rank_change and now.rank_of_data != prev.rank_of_data:
                    raise RuntimeError(
                        f"[{step_id} {module}] {bundle_key}.{ds_key} rank_of_data changed: "
                        f"{prev.rank_of_data} -> {now.rank_of_data}"
                    )

                if self.fail_on_nan_increase:
                    if now.nan_signal > prev.nan_signal:
                        raise RuntimeError(
                            f"[{step_id} {module}] {bundle_key}.{ds_key} signal NaNs increased: "
                            f"{prev.nan_signal} -> {now.nan_signal}"
                        )
                    keys = set(prev.nan_unc) | set(now.nan_unc)
                    for k in keys:
                        n_prev = prev.nan_unc.get(k, 0)
                        n_now = now.nan_unc.get(k, 0)
                        if n_now > n_prev:
                            raise RuntimeError(
                                f"[{step_id} {module}] {bundle_key}.{ds_key} unc['{k}'] NaNs increased: "
                                f"{n_prev} -> {n_now}"
                            )

                if diff:
                    changed[(bundle_key, ds_key)] = {"prev": prev, "now": now, "diff": diff}

        if (not self.record_only_on_change) or changed or self.record_empty_step_events:
            self.events.append(
                {
                    "step_id": step_id,
                    "module": module,
                    "name": name,
                    "changed": changed,
                    "duration_s": duration_s,
                }
            )

    def last_report(self, n: int = 20, *, renderer: ReportRenderer | None = None) -> str:
        r = renderer or PlainUnicodeRenderer(wrap_in_markdown_codeblock=False)
        events = self.events[-n:]
        blocks = [render_tracer_event(ev, renderer=r) for ev in events]
        # render_tracer_event already wraps, so join plainly:
        return "\n\n".join(blocks)


def _probe_to_dict(p: BaseDataProbe) -> dict[str, Any]:
    return {
        "shape": list(p.shape),
        "ndim": p.ndim,
        "rank_of_data": p.rank_of_data,
        "units": p.units_str,
        "dimensionality": p.dimensionality_str,
        "nan_signal": p.nan_signal,
        "nan_unc": dict(p.nan_unc),
        # only include if computed
        **(
            {"min_signal": p.min_signal, "max_signal": p.max_signal}
            if (p.min_signal is not None or p.max_signal is not None)
            else {}
        ),
    }


def tracer_event_to_datasets_payload(tracer_step_event: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single tracer 'events' entry into TraceEvent.datasets payload.

    Input shape:
      {"changed": {(bundle, ds): {"prev": BaseDataProbe|None, "now": BaseDataProbe, "diff": set[str]}}}

    Output shape:
      {"bundle.ds": {"diff": [...], "prev": {...}|None, "now": {...}}}
    """
    out: dict[str, Any] = {}
    changed = tracer_step_event.get("changed", {}) or {}

    # Stable order for UI diffs
    for bundle_key, ds_key in sorted(changed.keys(), key=lambda x: (x[0], x[1])):
        payload = changed[(bundle_key, ds_key)]
        prev = payload.get("prev")
        now = payload.get("now")
        diff = payload.get("diff", set())

        out[f"{bundle_key}.{ds_key}"] = {
            "diff": sorted(diff),
            "prev": None if prev is None else _probe_to_dict(prev),
            "now": _probe_to_dict(now),
        }

    return out


def render_tracer_event(tracer_event: dict[str, Any], *, renderer: ReportRenderer | None = None) -> str:
    """
    Render exactly ONE tracer event (one element from PipelineTracer.events).
    Strictly step-local: no reliance on global tracer state.
    """
    r = renderer or PlainUnicodeRenderer(wrap_in_markdown_codeblock=False)
    lines: list[str] = []

    def fmt_kv(label: str, prev: object | None, now: object, is_changed: bool) -> str:
        badge = r.badge_changed() if is_changed else r.badge_ok()
        if prev is None:
            return f"{badge} {label:<18} {now}"  # noqa: E231
        if is_changed:
            return f"{badge} {label:<18} {r.changed(str(prev))} → {r.changed(str(now))}"  # noqa: E231
        return f"{badge} {label:<18} {r.ok(str(now))}"  # noqa: E231

    step_id = tracer_event.get("step_id", "<??>")
    module = tracer_event.get("module", "")
    name = tracer_event.get("name", "")
    dur = tracer_event.get("duration_s", None)
    dur_txt = ""
    if isinstance(dur, (int, float)):
        dur_txt = f" {r.dim(f'⏱ {dur * 1e3:.2f} ms')}"  # noqa: E231
    lines.append(r.header(f"Step {step_id} — {module} — {name}") + dur_txt)

    changed_map: dict[tuple[str, str], dict[str, Any]] = tracer_event.get("changed", {}) or {}
    items = sorted(changed_map.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    for idx, ((b, d), payload) in enumerate(items):
        is_last_ds = idx == (len(items) - 1)
        joint = UNICODE["elbow"] if is_last_ds else UNICODE["tee"]
        cont = UNICODE["space"] if is_last_ds else UNICODE["pipe"]

        prev: BaseDataProbe | None = payload.get("prev")
        now: BaseDataProbe = payload.get("now")
        diff: set[str] = set(payload.get("diff", set()) or set())
        diff_str = ", ".join(sorted(diff))
        diff_note = f" {r.dim('[' + diff_str + ']')}" if diff_str else ""

        lines.append(f"{joint} {UNICODE['bullet']} {b}.{d}{diff_note}")

        if prev is None:
            lines.append(f"{cont}{UNICODE['space']}{fmt_kv('units', None, now.units_str, True)}")
            lines.append(f"{cont}{UNICODE['space']}{fmt_kv('dimensionality', None, now.dimensionality_str, True)}")
            lines.append(f"{cont}{UNICODE['space']}{fmt_kv('shape', None, now.shape, True)}")
            lines.append(f"{cont}{UNICODE['space']}{fmt_kv('NaN(signal)', None, now.nan_signal, True)}")
        else:
            lines.append(
                f"{cont}{UNICODE['space']}{fmt_kv('units', prev.units_str, now.units_str, now.units_str != prev.units_str)}"
            )
            lines.append(
                f"{cont}{UNICODE['space']}{fmt_kv('dimensionality', prev.dimensionality_str, now.dimensionality_str, now.dimensionality_str != prev.dimensionality_str)}"
            )
            lines.append(f"{cont}{UNICODE['space']}{fmt_kv('shape', prev.shape, now.shape, now.shape != prev.shape)}")
            lines.append(
                f"{cont}{UNICODE['space']}{fmt_kv('NaN(signal)', prev.nan_signal, now.nan_signal, now.nan_signal != prev.nan_signal)}"
            )

        unc_keys = sorted(now.nan_unc.keys() if prev is None else (set(prev.nan_unc) | set(now.nan_unc)))
        if unc_keys:
            lines.append(f"{cont}{UNICODE['space']}{r.dim('uncertainties:')}")
            for uk in unc_keys:
                p = 0 if prev is None else prev.nan_unc.get(uk, 0)
                q = now.nan_unc.get(uk, 0)
                lines.append(
                    f"{cont}{UNICODE['space']}{UNICODE['space']}"
                    + fmt_kv(f"NaN(unc['{uk}'])", None if prev is None else p, q, True if prev is None else (p != q))
                )

    return r.codewrap("\n".join(lines).rstrip())

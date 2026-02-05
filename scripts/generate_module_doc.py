# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

"""Generate a Markdown documentation page for a MoDaCor ProcessStep module."""

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import attr

from modacor.dataclasses.process_step_describer import ProcessStepDescriber


def _load_process_step(target: str):
    """Import and return the ProcessStep subclass designated by *target*.

    Parameters
    ----------
    target:
        Either a fully-qualified dotted path (``package.module.Class``) or the name of a
        class within the ``modacor.modules`` namespace.
    """

    if "." not in target:
        raise ValueError(
            "Target must be a fully-qualified dotted path, e.g. \n  modacor.modules.base_modules.divide.Divide"
        )

    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)

    try:
        process_step_cls = getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Class {class_name!r} not found in module {module_path!r}.") from exc  # noqa: E713

    if not inspect.isclass(process_step_cls):  # pragma: no cover - guard
        raise SystemExit(f"{target!r} is not a class.")

    documentation = getattr(process_step_cls, "documentation", None)
    if documentation is None or not isinstance(documentation, ProcessStepDescriber):
        raise SystemExit(
            f"ProcessStep class {target!r} does not expose a 'documentation' "
            "attribute or it is not a ProcessStepDescriber instance."
        )

    return process_step_cls, documentation


def _discover_targets(module_path: str = "modacor.modules") -> list[str]:
    module = importlib.import_module(module_path)
    names: Iterable[str] = getattr(module, "__all__", []) or []
    targets: list[str] = []
    for name in names:
        obj = getattr(module, name, None)
        if obj is None:
            continue
        targets.append(f"{obj.__module__}.{name}")
    return targets


def _format_list(items: list[Any]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- _None_"


def _format_modifies(modifies: dict[str, list[str]]) -> str:
    if not modifies:
        return "- _None_"
    lines = []
    for basedata, props in modifies.items():
        if props:
            prop_list = ", ".join(props)
            lines.append(f"- **{basedata}**: {prop_list}")
        else:
            lines.append(f"- **{basedata}**")
    return "\n".join(lines)


def _format_arguments(documentation: ProcessStepDescriber) -> str:
    arguments = getattr(documentation, "arguments", {}) or {}
    if not arguments:
        return "_No configuration arguments documented._"

    defaults = {}
    if hasattr(documentation, "initial_configuration"):
        defaults = documentation.initial_configuration() or {}

    header = "| Argument | Type | Required | Default | Description |\n|---|---|---|---|---|"
    rows = []
    for name, spec in sorted(arguments.items()):
        raw_types = spec.get("type", [])
        if raw_types in (None, []):
            type_items = []
        elif isinstance(raw_types, (tuple, list, set)):
            type_items = list(raw_types)
        else:
            type_items = [raw_types]
        type_repr = " or ".join(t.__name__ if hasattr(t, "__name__") else str(t) for t in type_items)
        required = "Yes" if spec.get("required", False) else "No"
        default_value = defaults.get(name, spec.get("default", None))
        if isinstance(default_value, (dict, list, tuple)):
            default_repr = json.dumps(default_value, ensure_ascii=False)
        elif isinstance(default_value, str):
            default_repr = default_value
        elif default_value is None:
            default_repr = "-"
        else:
            default_repr = str(default_value)
        description = spec.get("doc", "") or ""
        rows.append(f"| `{name}` | {type_repr or '-'} | {required} | {default_repr} | {description} |")

    return "\n".join([header, *rows])


def _format_default_config(documentation: ProcessStepDescriber) -> str:
    defaults = {}
    if hasattr(documentation, "initial_configuration"):
        defaults = documentation.initial_configuration() or {}
    else:
        defaults = documentation.default_configuration_copy() or {}
    if not defaults:
        return "_No default configuration defined._"
    return "```json\n" + json.dumps(defaults, indent=2, sort_keys=True) + "\n```"


def _format_required_arguments(documentation: ProcessStepDescriber) -> str:
    if hasattr(documentation, "required_argument_names"):
        required_args = list(documentation.required_argument_names())
    else:
        required_args = getattr(documentation, "required_arguments", []) or []
    return _format_list(required_args)


def _format_summary(documentation: ProcessStepDescriber) -> str:
    metadata = attr.asdict(documentation, recurse=False)
    interesting_keys = [
        ("Module ID", "calling_id"),
        ("Module path", "calling_module_path"),
        ("Module version", "calling_version"),
        ("Keywords", "step_keywords"),
    ]
    lines = []
    for label, key in interesting_keys:
        value = metadata.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"- **{label}:** {value}")  # noqa: E231
    return "\n".join(lines) or "- _No metadata available._"


def build_markdown(step_cls, documentation: ProcessStepDescriber) -> str:
    title = documentation.calling_name or step_cls.__name__
    step_doc = documentation.step_doc or ""
    reference = documentation.step_reference or ""
    note = documentation.step_note or ""

    content = [
        "# " + title,
        "",
        "## Summary",
        step_doc or "_No summary provided._",
        "",
        "## Metadata",
        _format_summary(documentation),
        "",
        "## Required data keys",
        _format_list(documentation.required_data_keys or []),
        "",
        "## Modifies",
        _format_modifies(documentation.modifies or {}),
        "",
        "## Required arguments",
        _format_required_arguments(documentation),
        "",
        "## Default configuration",
        _format_default_config(documentation),
        "",
        "## Argument specification",
        _format_arguments(documentation),
    ]

    if reference:
        content.extend(["", "## References", reference])

    if note:
        content.extend(["", "## Notes", note])

    return "\n".join(content).rstrip() + "\n"


def _write_module_index(index_path: Path, module_files: list[Path]) -> None:
    entries = "\n".join(path.stem for path in module_files)
    content = "\n".join(
        [
            "# Module reference",
            "",
            "```{toctree}",
            ":maxdepth: 1",
            "",
            entries,
            "```",
            "",
        ]
    )
    index_path.write_text(content, encoding="utf-8")


def run_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        help=(
            "Fully-qualified dotted path to the ProcessStep class (e.g. 'modacor.modules.base_modules.divide.Divide')."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output Markdown file. If omitted, the documentation is written to stdout.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate documentation for all modules exposed via modacor.modules.__all__.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for per-module Markdown files (used with --all).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Optional index Markdown file to write a toctree for generated modules.",
    )
    args = parser.parse_args()

    if args.all:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required when using --all.")
        targets = _discover_targets()
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: list[Path] = []
        for target in targets:
            step_cls, documentation = _load_process_step(target)
            markdown = build_markdown(step_cls, documentation)
            output_path = output_dir / f"{step_cls.__name__}.md"
            output_path.write_text(markdown, encoding="utf-8")
            generated_files.append(output_path)

        if args.index:
            _write_module_index(args.index, generated_files)
        return 0

    if not args.target:
        raise SystemExit("Either a target or --all must be provided.")

    step_cls, documentation = _load_process_step(args.target)
    markdown = build_markdown(step_cls, documentation)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(run_cli())

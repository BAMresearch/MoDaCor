# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a Markdown documentation page for a MoDaCor ProcessStep module."""

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# Allow running this script directly from a source checkout without requiring
# an editable install in the active interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPOSITORY_URL = "https://github.com/BAMresearch/MoDaCor"

MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "Data movement and copying": (
        "AppendProcessingData",
        "AppendSink",
        "AppendSource",
        "ConcatenateDatabundles",
        "CopyDataBundleKeys",
        "SinkProcessingData",
    ),
    "Arithmetic and normalization": (
        "Divide",
        "DivideDatabundles",
        "FindScaleFactor1D",
        "Multiply",
        "MultiplyDatabundles",
        "Negate",
        "Subtract",
        "SubtractDatabundles",
        "SubtractInterpolated1D",
        "UnitsLabelUpdate",
    ),
    "Masks": (
        "ApplyMask",
        "BitwiseOrMasks",
        "DilateMask",
        "ReduceMask",
        "ThresholdMask",
    ),
    "Uncertainty creation and combination": (
        "CombineUncertainties",
        "CombineUncertaintiesMax",
        "PoissonUncertainties",
    ),
    "Geometry and coordinates": (
        "IndexPixels",
        "PixelCoordinates3D",
        "XSGeometryFromPixelCoordinates",
        "AngleToQ",
    ),
    "Reduction and integration": (
        "FindCenterOfMass1D",
        "IndexedAverager",
        "Integrate1D",
        "ReduceDimensionality",
    ),
    "Visualization": (
        "Plot1DVisualization",
        "Plot2DVisualization",
    ),
    "Technique-specific corrections": (
        "AttenuatorPlateCorrection",
        "CapillarySampleContainerCorrection",
        "CapillarySelfAbsorptionCorrection",
        "DetectorEfficiencyCorrection",
        "FlatPlateSelfAbsorptionCorrection",
        "PolarizationCorrection",
        "SolidAngleCorrection",
    ),
}
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))


def _maybe_reexec_with_local_venv() -> None:
    """Retry with a project-local interpreter if current Python misses deps."""
    if os.environ.get("MODACOR_DOCGEN_REEXEC") == "1":
        return

    candidate_bins = [
        PROJECT_ROOT / ".venv-dev" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidate_bins:
        if not candidate.exists():
            continue
        if Path(sys.executable).resolve() == candidate.resolve():
            continue

        env = dict(os.environ)
        env["MODACOR_DOCGEN_REEXEC"] = "1"
        completed = subprocess.run([str(candidate), *sys.argv], env=env, check=False)
        raise SystemExit(completed.returncode)


if sys.version_info < (3, 11):
    _maybe_reexec_with_local_venv()


try:
    import attr

    from modacor.dataclasses.process_step_describer import ProcessStepDescriber
except ModuleNotFoundError:
    _maybe_reexec_with_local_venv()
    raise


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

    header = "| Argument | Type | Required | Default | Dependency role | Description |\n" "|---|---|---|---|---|---|"
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
        dependency_role = spec.get("dependency_role")
        if isinstance(dependency_role, (list, tuple, set)):
            dependency_role_repr = ", ".join(str(role) for role in dependency_role)
        elif dependency_role is None:
            dependency_role_repr = "-"
        else:
            dependency_role_repr = str(dependency_role)
        description = spec.get("doc", "") or ""
        rows.append(
            f"| `{name}` | {type_repr or '-'} | {required} | {default_repr} | "
            f"{dependency_role_repr} | {description} |"
        )

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


def _repository_relative_source(documentation: ProcessStepDescriber) -> Path | None:
    source_path = Path(documentation.calling_module_path)
    try:
        return source_path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None


def _format_summary(step_cls, documentation: ProcessStepDescriber) -> str:
    metadata = attr.asdict(documentation, recurse=False)
    interesting_keys = [
        ("Module ID", "calling_id"),
        ("Module version", "calling_version"),
        ("Keywords", "step_keywords"),
    ]
    import_path = f"{step_cls.__module__}.{step_cls.__name__}"
    lines = [f"- **Import path:** `{import_path}`"]
    relative_source = _repository_relative_source(documentation)
    if relative_source is not None:
        source_text = relative_source.as_posix()
        lines.append(f"- **Source:** [`{source_text}`]({REPOSITORY_URL}/blob/main/{source_text})")
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
        _format_summary(step_cls, documentation),
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


def _validate_module_groups(module_names: set[str]) -> None:
    classified_names = [name for names in MODULE_GROUPS.values() for name in names]
    duplicates = sorted({name for name in classified_names if classified_names.count(name) > 1})
    missing = sorted(module_names - set(classified_names))
    unknown = sorted(set(classified_names) - module_names)
    problems = []
    if duplicates:
        problems.append(f"duplicate classifications: {', '.join(duplicates)}")
    if missing:
        problems.append(f"unclassified modules: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown classified modules: {', '.join(unknown)}")
    if problems:
        raise ValueError("Invalid module documentation groups (" + "; ".join(problems) + ").")


def _build_module_index(module_names: set[str]) -> str:
    _validate_module_groups(module_names)
    content = [
        "# Process-step reference",
        "",
        "Every supported public `ProcessStep` is listed below by function and alphabetically.",
        "Configuration tables are generated from each step's `ProcessStepDescriber` metadata.",
    ]
    for group_name, names in MODULE_GROUPS.items():
        content.extend(["", f"## {group_name}", ""])
        content.extend(f"- [{name}]({name}.md)" for name in names)
    content.extend(
        [
            "",
            "## Alphabetical index",
            "",
            "```{toctree}",
            ":maxdepth: 1",
            "",
            *sorted(module_names),
            "```",
            "",
        ]
    )
    return "\n".join(content)


def _generated_pages(targets: list[str]) -> dict[str, str]:
    pages: dict[str, str] = {}
    for target in targets:
        step_cls, documentation = _load_process_step(target)
        pages[f"{step_cls.__name__}.md"] = build_markdown(step_cls, documentation)
    _validate_module_groups({Path(filename).stem for filename in pages})
    return pages


def _check_generated_output(output_dir: Path, index_path: Path | None, pages: dict[str, str]) -> list[str]:
    problems: list[str] = []
    expected_names = set(pages)
    actual_names = {path.name for path in output_dir.glob("*.md") if index_path is None or path != index_path}
    for filename in sorted(expected_names):
        path = output_dir / filename
        if not path.exists():
            problems.append(f"missing {path}")
        elif path.read_text(encoding="utf-8") != pages[filename]:
            problems.append(f"stale {path}")
    for filename in sorted(actual_names - expected_names):
        problems.append(f"unexpected {output_dir / filename}")
    if index_path is not None:
        expected_index = _build_module_index({Path(filename).stem for filename in pages})
        if not index_path.exists():
            problems.append(f"missing {index_path}")
        elif index_path.read_text(encoding="utf-8") != expected_index:
            problems.append(f"stale {index_path}")
    return problems


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that generated files are current without modifying them (used with --all).",
    )
    args = parser.parse_args()

    if args.all:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required when using --all.")
        targets = _discover_targets()
        output_dir = args.output_dir
        pages = _generated_pages(targets)
        if args.check:
            problems = _check_generated_output(output_dir, args.index, pages)
            if problems:
                print("Generated module documentation is not current:", file=sys.stderr)
                for problem in problems:
                    print(f"- {problem}", file=sys.stderr)
                return 1
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        existing_module_pages = set(output_dir.glob("*.md"))
        if args.index is not None:
            existing_module_pages.discard(args.index)

        generated_files: list[Path] = []
        for filename, markdown in pages.items():
            output_path = output_dir / filename
            output_path.write_text(markdown, encoding="utf-8")
            generated_files.append(output_path)

        for stale_path in existing_module_pages - set(generated_files):
            stale_path.unlink()

        if args.index:
            args.index.write_text(
                _build_module_index({path.stem for path in generated_files}),
                encoding="utf-8",
            )
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

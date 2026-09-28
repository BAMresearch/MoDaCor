from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"


def _load_docs_conf():
    spec = spec_from_file_location("modacor_docs_conf", DOCS_ROOT / "conf.py")
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_redirect_sources_are_retired_and_targets_exist():
    redirects = _load_docs_conf().rediraffe_redirects
    assert redirects
    assert not (set(redirects) & set(redirects.values())), "Redirect chains are not allowed."

    for source, target in redirects.items():
        assert not (DOCS_ROOT / f"{source}.md").exists(), source
        assert (DOCS_ROOT / f"{target}.md").exists(), target


def test_primary_navigation_and_examples_repository_are_present():
    index = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    for docname in (
        "introduction/index",
        "getting-started/index",
        "data-model/index",
        "processing/index",
        "modules/index",
        "server/index",
        "examples/index",
        "reference/index",
        "development/index",
    ):
        assert docname in index

    examples_url = "https://github.com/BAMResearch/MoDaCor-examples"
    assert examples_url in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert examples_url in (DOCS_ROOT / "examples/index.md").read_text(encoding="utf-8")


def test_agents_documentation_references_use_current_paths():
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "/Users/" not in agents
    assert "https://github.com/BAMResearch/MoDaCor-examples" in agents

    for retired_prefix in (
        "docs/extending/",
        "docs/design/completed/",
        "docs/getting_started/",
        "docs/pipeline_operations/",
    ):
        assert retired_prefix not in agents

    for current_path in (
        "docs/development/module-author-guide.md",
        "docs/development/contribution-checklist.md",
        "docs/development/io-source-sink-guide.md",
        "docs/development/design/completed/code-coherence.md",
    ):
        assert current_path in agents

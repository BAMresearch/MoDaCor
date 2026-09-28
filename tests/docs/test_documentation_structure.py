from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PROJECT_ROOT / "docs"


def _load_rediraffe_redirects() -> dict[str, str]:
    """Read the literal redirect map without importing optional Sphinx deps."""
    conf_path = DOCS_ROOT / "conf.py"
    tree = ast.parse(conf_path.read_text(encoding="utf-8"), filename=str(conf_path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "rediraffe_redirects" for target in node.targets)
    ]
    assert len(assignments) == 1
    redirects = ast.literal_eval(assignments[0].value)
    assert isinstance(redirects, dict)
    assert all(isinstance(source, str) and isinstance(target, str) for source, target in redirects.items())
    return redirects


def test_redirect_sources_are_retired_and_targets_exist():
    redirects = _load_rediraffe_redirects()
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
    for externalized_example in (
        "MOUSE_solids.yaml",
        "mouse_pipeline.md",
        "saxsess_pipeline.md",
        "dls_i22.md",
    ):
        assert not (DOCS_ROOT / "examples" / externalized_example).exists()


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

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
README = PROJECT_ROOT / "README.md"


def _quickstart_code() -> str:
    text = README.read_text(encoding="utf-8")
    matches = re.findall(
        r"<!-- quickstart-python-start -->\s*```python\n(.*?)\n```\s*<!-- quickstart-python-end -->",
        text,
        flags=re.DOTALL,
    )
    assert len(matches) == 1
    return matches[0]


def test_readme_quickstart_executes(capsys):
    namespace = {"__name__": "readme_quickstart"}
    exec(compile(_quickstart_code(), str(README), "exec"), namespace)

    result = namespace["result"]
    corrected = namespace["corrected"]
    assert result.executed_steps == ["uncertainties", "normalize"]
    np.testing.assert_allclose(corrected.signal, [[2.0, 4.5], [8.0, 12.5]])
    np.testing.assert_allclose(corrected.uncertainties["Poisson"], [[1.0, 1.5], [2.0, 2.5]])
    assert f"{corrected.units:~}" == "count / s"
    assert "['uncertainties', 'normalize']" in capsys.readouterr().out


def test_readme_quickstart_uses_uv_and_no_external_data():
    text = README.read_text(encoding="utf-8")
    assert "uv python install 3.12" in text
    assert "uv venv --python 3.12" in text
    assert "uv pip install modacor" in text

    code = _quickstart_code()
    for forbidden in ("requests", "urlopen", "curl", "HDFSource", "Path(", "open("):
        assert forbidden not in code

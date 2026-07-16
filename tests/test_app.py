"""The app itself.

Every test here is a crash that reached a screenshot or would have. Streamlit
runs the whole script top to bottom on every interaction, which makes it easy
to bind a name inside one tab's branch and read it from another — that works
right up until the user takes a slightly different path.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=400).run()
    assert not at.exception, at.exception
    return at


def test_landing_page_renders(app):
    assert not app.exception


def test_stylesheet_is_the_first_thing_on_the_page(app):
    """CSS injected late means the first paint is unstyled."""
    assert app.markdown[0].value.strip().startswith("<style>")


@pytest.mark.slow
def test_both_reports_build_without_ever_opening_the_charts_tab():
    """The path that used to NameError.

    `rendered` was bound inside `elif specs:` in the Charts tab, so loading
    data and going straight to Export — the most ordinary path there is —
    crashed. Nothing in the test suite went that way.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=500).run()
    at.sidebar.button[0].click().run()
    [b for b in at.button if "Propose" in b.label][0].click().run()
    [b for b in at.button if "Apply all" in b.label][0].click().run()

    for label in ("Build findings report", "Build quality report"):
        matches = [b for b in at.button if b.label == label]
        assert matches, f"no button labelled {label!r}"
        matches[0].click().run()
        assert not at.exception, f"{label} raised: {at.exception}"

    downloads = {d.label for d in at.download_button}
    assert {"Download findings.pdf", "Download quality.pdf"} <= downloads
    assert "Everything (.zip)" in downloads, "bundle must not depend on charts"


@pytest.mark.slow
def test_demo_survey_cleans_and_improves_quality():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=500).run()
    at.sidebar.button[0].click().run()
    scores = {m.label: m.value for m in at.metric}
    assert scores["Rows"] == "612"

    [b for b in at.button if "Propose" in b.label][0].click().run()
    [b for b in at.button if "Apply all" in b.label][0].click().run()
    assert not at.exception, at.exception

    after = {m.label: m.value for m in at.metric}
    assert int(after["Quality score"].split("/")[0]) > 74
    assert after["Rows"] == "600", "duplicates should be gone"


def test_no_name_is_bound_only_inside_a_conditional_but_read_at_module_level():
    """Guard for the class of bug, not just the one instance.

    Any name assigned only inside an `if`/`elif` body but read from module
    level is a NameError waiting for the right click order.
    """
    tree = ast.parse((ROOT / "app.py").read_text())

    conditional_only: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        conditional_only.add(target.id)

    module_level: set[str] = set()
    for node in tree.body:                       # top level only
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    module_level.add(target.id)

    risky = {"rendered", "specs", "plan", "current", "before", "ledger", "df"}
    for name in risky & conditional_only:
        assert name in module_level, (
            f"{name!r} is assigned inside a conditional but other tabs read it. "
            f"Bind it at module level from st.session_state first."
        )

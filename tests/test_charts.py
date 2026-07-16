"""Chart suggestion and rendering.

Rendering code fails in a specific way: it works on the demo data and crashes
on yours. The demo survey only exercises 4 of the 8 chart kinds, so the other
4 could be broken and every test would still pass — which is exactly how
`promote_header_row` stayed broken for a week. So every kind is rendered here
against a frame built to trigger it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hma import charts  # noqa: E402
from hma.agents import plan_from_findings  # noqa: E402
from hma.ledger import Ledger  # noqa: E402
from hma.profiling import profile  # noqa: E402
from scripts.make_messy_survey import build  # noqa: E402

SCALE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


@pytest.fixture(scope="module")
def plottable() -> pd.DataFrame:
    """One frame that can trigger every chart kind."""
    n = 90
    frame = pd.DataFrame(
        {
            "respondent_id": [f"R{i:03d}" for i in range(n)],
            "age": [18 + (i * 7) % 60 for i in range(n)],
            "income": [20_000 + (i * 3_100) % 250_000 for i in range(n)],
            "gender": ["Male", "Female", "Female"] * (n // 3),
            "state": ["Kwara", "Lagos", "Oyo", "Kano", "Rivers", "FCT"] * (n // 6),
            "Q1": pd.Categorical(
                [SCALE[i % 5] for i in range(n)], categories=SCALE, ordered=True
            ),
            "survey_date": pd.to_datetime(
                [f"2026-0{1 + i % 3}-{1 + i % 28:02d}" for i in range(n)]
            ),
            "age_out_of_range": [i % 11 == 0 for i in range(n)],
        }
    )
    frame.loc[frame.index[:12], "income"] = None  # give missingness something to draw
    return frame


@pytest.fixture(scope="module")
def cleaned() -> pd.DataFrame:
    df = build()
    ledger = Ledger("demo.csv")
    for step in plan_from_findings(profile(df).findings):
        if step.confidence >= 0.90:
            df = ledger.apply(df, step.op, step.params, step.rationale)
    return df


# --------------------------------------------------------------------------
# Every kind must render
# --------------------------------------------------------------------------

def test_all_chart_kinds_are_reachable(plottable):
    """A kind with a template but no rule to produce it is dead code."""
    produced = {s.kind for s in charts.suggest(plottable, limit=200)}
    templated = set(charts.CODE_TEMPLATES)
    unreachable = templated - produced
    assert not unreachable, (
        f"these kinds have templates but nothing suggests them: {sorted(unreachable)}"
    )


@pytest.mark.parametrize("kind", sorted(charts.CODE_TEMPLATES))
def test_every_kind_renders_to_a_png(plottable, kind):
    """The demo data only hits 4 of 8. This hits all of them."""
    specs = [s for s in charts.suggest(plottable, limit=200) if s.kind == kind]
    assert specs, f"nothing produced a {kind} chart from the fixture"

    figure = charts.render(specs[0], plottable)
    png = charts.to_png(figure)
    assert png.startswith(b"\x89PNG"), f"{kind} did not produce a PNG"
    assert len(png) > 1_000, f"{kind} rendered a suspiciously empty image"


@pytest.mark.parametrize("kind", sorted(charts.CODE_TEMPLATES))
def test_every_kind_carries_runnable_code(plottable, kind):
    """A chart you can't reproduce is a chart nobody should cite."""
    spec = next(s for s in charts.suggest(plottable, limit=200) if s.kind == kind)
    code = spec.code
    assert "no template" not in code, f"{kind} has no code template"
    compile(code, f"<{kind}>", "exec")  # must at least be valid Python


# --------------------------------------------------------------------------
# What must never be suggested
# --------------------------------------------------------------------------

def test_identifiers_are_never_a_chart_dimension(cleaned):
    """'Q2 by respondent_name' is a privacy breach dressed as an analysis.

    The missingness chart is the deliberate exception: it plots column names
    and blank counts, never the values inside them. Knowing your ID column has
    gaps is useful and exposes nobody.
    """
    identifiers = set(charts.classify(cleaned)["identifier"])
    assert identifiers, "fixture should contain identifiers to exclude"
    for spec in charts.suggest(cleaned, limit=200):
        if spec.kind == "missingness":
            continue
        assert not set(spec.columns) & identifiers, (
            f"{spec.title!r} plots an identifier: {set(spec.columns) & identifiers}"
        )


def test_multiselect_members_are_not_identifiers(cleaned):
    """They're superseded columns, not people. Saying 'identifier' teaches a lie."""
    roles = charts.classify(cleaned)
    members = set(roles["multiselect_member"])
    assert members, "fixture should contain collapsed multi-select members"
    assert all(m.startswith("Q7_Part_") for m in members)
    assert not members & set(roles["identifier"])


def test_multiselect_members_are_never_plotted(cleaned):
    """They're double-counted by the _count column that replaced them."""
    members = set(charts.classify(cleaned)["multiselect_member"])
    for spec in charts.suggest(cleaned, limit=200):
        assert not set(spec.columns) & members, f"{spec.title!r} re-plots {members}"


def test_ordered_scales_keep_their_order(cleaned):
    """A Likert chart sorted alphabetically is a wrong chart that looks right."""
    roles = charts.classify(cleaned)
    assert roles["ordered"], "cleaning should have restored at least one scale"
    for column in roles["ordered"]:
        assert cleaned[column].cat.ordered


def test_suggestions_are_ranked_best_first(cleaned):
    scores = [s.score for s in charts.suggest(cleaned, limit=200)]
    assert scores == sorted(scores, reverse=True)


def test_limit_is_respected(cleaned):
    assert len(charts.suggest(cleaned, limit=3)) <= 3


# --------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------

def test_empty_frame_suggests_nothing_and_does_not_raise():
    assert charts.suggest(pd.DataFrame()) == []


def test_tiny_frame_suggests_nothing():
    """Five rows is not a distribution. Drawing one implies it is."""
    tiny = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert charts.suggest(tiny) == []


def test_all_constant_frame_does_not_raise():
    constant = pd.DataFrame({"consent": ["Yes"] * 40, "site": ["A"] * 40})
    for spec in charts.suggest(constant):
        charts.render(spec, constant)


def test_dashboard_html_embeds_the_images(plottable):
    specs = charts.suggest(plottable, limit=3)
    rendered = [(s, charts.to_png(charts.render(s, plottable))) for s in specs]
    html = charts.dashboard_html(rendered, "demo.csv")
    assert html.lstrip().lower().startswith("<!doctype html")
    assert html.count("data:image/png;base64,") == len(rendered)
    for spec, _ in rendered:
        assert spec.title in html


# --------------------------------------------------------------------------
# Insights export
# --------------------------------------------------------------------------

def test_insights_markdown_reports_the_whole_session(cleaned):
    """The PDF is for handing over. This is for working with."""
    from hma.ledger import Ledger
    from hma.pipeline import build_plan
    from hma.profiling import profile as prof
    from hma.report import insights_markdown
    from scripts.make_messy_survey import build as build_messy

    raw = build_messy()
    before = prof(raw)
    plan = build_plan(before.findings, raw)
    ledger = Ledger("demo.csv")
    df = raw.copy()
    for p in plan.runnable:
        if p.confidence >= 0.90:
            df = ledger.apply(df, p.op, p.params, p.rationale)

    text = insights_markdown(
        before, prof(df), ledger,
        specs=charts.suggest(df, limit=4),
        never_impute=plan.never_impute,
        dataset_name="demo.csv",
    )

    assert text.startswith("# Data quality insights")
    assert "## What was found" in text
    assert "## What was done" in text
    assert "## What to look at" in text
    assert "## Cautions" in text
    assert "Q9_followup" in text, "skip-logic caution should name the column"
    assert "Do not impute" in text
    assert "clean.py" in text, "should point at the reproducible script"


def test_insights_survives_an_empty_session():
    """No findings, no ops, no charts — must still produce a valid document."""
    from hma.ledger import Ledger
    from hma.profiling import profile as prof
    from hma.report import insights_markdown

    clean = pd.DataFrame({"a": range(50), "b": [f"v{i%4}" for i in range(50)]})
    p = prof(clean)
    text = insights_markdown(p, p, Ledger("x.csv"), dataset_name="x.csv")
    assert "No operations were applied." in text
    assert "## Cautions" in text


def test_insights_escapes_pipes_that_would_break_the_tables():
    """A column named 'rate|pct' would silently shred the markdown table."""
    from hma.ledger import Ledger
    from hma.profiling import profile as prof
    from hma.report import insights_markdown

    df = pd.DataFrame({"a|b": [1, 2, 999] * 20})
    p = prof(df)
    text = insights_markdown(p, p, Ledger("x.csv"), dataset_name="x.csv")
    for line in text.splitlines():
        if line.startswith("| `") and "sentinel" in line:
            assert line.count("|") >= 5

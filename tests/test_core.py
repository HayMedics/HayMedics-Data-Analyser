"""Tests for the parts that would embarrass you if they broke.

The headline test is test_exported_script_reproduces_session. If that ever
fails, the ledger is lying and the project has no reason to exist.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hma import analyst, survey  # noqa: E402
from hma.agents import plan_from_findings  # noqa: E402
from hma.ledger import Ledger  # noqa: E402
from hma.profiling import profile  # noqa: E402
from scripts.make_messy_survey import build  # noqa: E402


@pytest.fixture(scope="module")
def messy() -> pd.DataFrame:
    return build()


# --------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------

def test_exported_script_reproduces_session(messy, tmp_path):
    """Run a session, export the script, run the script, compare.

    This is the whole product in one test.
    """
    raw = tmp_path / "raw.csv"
    messy.to_csv(raw, index=False)

    df = pd.read_csv(raw)
    ledger = Ledger(source_name=str(raw))
    for step in plan_from_findings(profile(df).findings):
        if step.confidence >= 0.90:
            df = ledger.apply(df, step.op, step.params, step.rationale)

    app_output = tmp_path / "from_app.csv"
    df.to_csv(app_output, index=False)

    script = tmp_path / "clean.py"
    script.write_text(ledger.to_script())

    proc = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Exported script failed:\n{proc.stderr}"

    from_script = pd.read_csv(tmp_path / "cleaned.csv")
    from_app = pd.read_csv(app_output)

    assert from_script.shape == from_app.shape
    pd.testing.assert_frame_equal(from_script, from_app, check_dtype=False)


def test_script_needs_no_hma_import(messy, tmp_path):
    """The exported script must not depend on this package."""
    raw = tmp_path / "raw.csv"
    messy.to_csv(raw, index=False)
    ledger = Ledger(source_name=str(raw))
    ledger.apply(pd.read_csv(raw), "drop_duplicates", {}, "dupes")  # populate it

    script = ledger.to_script()
    imports = [
        line for line in script.splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imports, "script should import pandas at least"
    assert all("hma" not in line for line in imports), imports
    assert "openrouter" not in script.lower()
    assert "api_key" not in script.lower()


# --------------------------------------------------------------------------
# Survey detection
# --------------------------------------------------------------------------

def test_detects_planted_pathologies(messy):
    kinds = {f.kind for f in profile(messy).findings}
    for expected in {"sentinel_missing", "multiselect_group", "pii",
                     "likert_scale", "case_variants", "structural_missing"}:
        assert expected in kinds, f"missed {expected}"


def test_pii_matches_through_underscores():
    """`\\bname\\b` does not match `respondent_name` — underscores are word chars."""
    df = pd.DataFrame({"respondent_name": ["A"], "emailAddress": ["a@b.c"], "age": [30]})
    flagged = {f.column for f in survey.detect_pii(df)}
    assert "respondent_name" in flagged
    assert "emailAddress" in flagged
    assert "age" not in flagged


def test_multiselect_blanks_are_not_called_missing(messy):
    """Blank in Q7_Part_3 means 'not ticked'. Calling it skip logic is wrong."""
    findings = profile(messy).findings
    structural = {f.column for f in findings if f.kind == "structural_missing"}
    assert not any(c.startswith("Q7_Part_") for c in structural)


def test_likert_survives_modern_string_dtype():
    """pandas 3.0 uses `str` dtype; `dtype == object` silently returns False."""
    series = pd.Series(["Agree", "Disagree", "Neutral"] * 5, name="Q1").astype("string")
    assert survey.is_texty(series)
    assert survey.detect_likert(series) is not None


def test_sentinel_not_flagged_when_it_is_most_of_the_column():
    """99 as 95% of a column is a real value, not a missing code."""
    series = pd.Series([99] * 95 + [1, 2, 3, 4, 5], name="score")
    found = survey.detect_sentinel_missing(series)
    assert found is None or found.confidence < 0.5


# --------------------------------------------------------------------------
# The registry boundary
# --------------------------------------------------------------------------

def test_unknown_op_is_rejected(messy):
    ledger = Ledger()
    with pytest.raises(KeyError):
        ledger.apply(messy, "rm_rf_slash", {"path": "/"})


def test_undo_restores_previous_state(messy):
    ledger = Ledger()
    after = ledger.apply(messy, "drop_duplicates", {}, "dupes")
    assert len(after) < len(messy)
    restored = ledger.undo()
    assert len(restored) == len(messy)
    assert len(ledger) == 0


# --------------------------------------------------------------------------
# SQL guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE data",
        "SELECT 1; DROP TABLE data",
        "INSERT INTO data VALUES (1)",
        "UPDATE data SET age = 1",
        "SELECT * FROM read_csv('/etc/passwd')",
        "select 1 -- ; drop table data\n; drop table data",
        "ATTACH '/tmp/evil.db'",
        "",
    ],
)
def test_sql_guard_blocks_writes_and_reads_of_disk(sql):
    with pytest.raises(analyst.UnsafeQuery):
        analyst.guard(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM data",
        "select gender, count(*) from data group by gender",
        "WITH t AS (SELECT * FROM data) SELECT COUNT(*) FROM t",
        "SELECT AVG(age) FROM data WHERE age IS NOT NULL;",
    ],
)
def test_sql_guard_allows_reads(sql):
    assert analyst.guard(sql)


def test_sql_actually_computes(messy):
    result = analyst.run_sql(messy, "SELECT COUNT(*) AS n FROM data")
    assert result.iloc[0]["n"] == len(messy)


# --------------------------------------------------------------------------
# Quality score
# --------------------------------------------------------------------------

def test_cleaning_improves_quality_score(messy):
    before = profile(messy)
    df = messy.copy()
    ledger = Ledger()
    for step in plan_from_findings(before.findings):
        if step.confidence >= 0.90:
            df = ledger.apply(df, step.op, step.params, step.rationale)
    df = ledger.apply(df, "drop_duplicates", {}, "dupes")
    assert profile(df).quality_score > before.quality_score


def test_perfect_frame_scores_high():
    df = pd.DataFrame({"a": range(100), "b": [f"v{i % 5}" for i in range(100)]})
    assert profile(df).quality_score >= 95


# --------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------

def test_brand_assets_exist():
    """A missing logo should degrade the look, never break the app."""
    from hma import brand
    assert brand.asset(brand.ICON), "page icon missing from assets/"
    assert brand.asset(brand.LOGO_HORIZONTAL), "PDF letterhead logo missing"
    assert brand.asset(brand.LOGO_STACKED_TAGLINE), "sidebar logo missing"
    assert brand.asset(Path("/nope/missing.png")) is None


def test_brand_colours_are_valid_hex():
    from hma import brand
    for name in ("BLUE", "ORANGE", "INDIGO", "NAVY", "SLATE", "TINT", "LINE"):
        value = getattr(brand, name)
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", value), f"{name}={value!r}"


def test_report_is_branded(messy, tmp_path):
    """The PDF must carry the logo and the HayMedics metadata."""
    from pypdf import PdfReader

    from hma import brand
    from hma.report import build_report

    before = profile(messy)
    ledger = Ledger("demo.csv")
    df = ledger.apply(messy, "drop_duplicates", {}, "dupes")

    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(build_report(before, profile(df), ledger, "demo.csv"))

    reader = PdfReader(pdf)
    assert brand.ORG in (reader.metadata.author or "")
    assert brand.PRODUCT in (reader.metadata.title or "")
    assert len(list(reader.pages[0].images)) >= 1, "letterhead logo not embedded"
    assert brand.ORG_TAGLINE in reader.pages[0].extract_text()


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------

def test_duplicates_are_detected_and_fixed(messy):
    """The score docks points for duplicates, so the plan must offer the fix."""
    before = profile(messy)
    assert before.n_duplicate_rows > 0
    assert any(f.kind == "duplicate_rows" for f in before.findings)

    df = messy.copy()
    ledger = Ledger()
    for step in plan_from_findings(before.findings):
        if step.confidence >= 0.90:
            df = ledger.apply(df, step.op, step.params, step.rationale)
    assert profile(df).n_duplicate_rows == 0, "'apply everything' left duplicates behind"


def test_dedupe_runs_after_standardising(messy):
    """Two rows aren't equal until STANDARDISE has run.

    Dedupe first and '  Lagos' vs 'lagos' are two different respondents, so
    the real duplicate survives. This asserts the stage ordering rather than
    a fixed position: dedupe is no longer last now that MISSING, OUTLIERS and
    VALIDATE follow it, but it must still come after every standardising op.
    """
    from hma.ops import REGISTRY, Stage

    plan = plan_from_findings(profile(messy).findings)
    stages = [REGISTRY[p.op].stage for p in plan]

    assert Stage.DEDUPLICATE in stages, "nothing proposes removing duplicates"
    assert stages == sorted(stages), "the plan is not in pipeline-stage order"

    last_standardise = max(
        (i for i, s in enumerate(stages) if s <= Stage.STANDARDISE), default=-1
    )
    first_dedupe = stages.index(Stage.DEDUPLICATE)
    assert first_dedupe > last_standardise, "dedupe runs before standardising"


def test_sentinels_are_recoded_before_types(messy):
    """Coerce first and to_numeric silently eats 'Prefer not to say'.

    That loses the fact that someone refused — which in survey work is a
    finding, not an absence of one.
    """
    from hma.ops import REGISTRY, Stage

    stages = [REGISTRY[p.op].stage for p in plan_from_findings(profile(messy).findings)]
    if Stage.SENTINEL in stages and Stage.TYPES in stages:
        assert max(i for i, s in enumerate(stages) if s == Stage.SENTINEL) < min(
            i for i, s in enumerate(stages) if s == Stage.TYPES
        )


def test_brand_css_does_not_leak_into_the_page():
    """The stylesheet must survive Streamlit's CommonMark pass intact.

    Streamlit renders st.markdown through a CommonMark parser. Get the HTML
    block type wrong and the CSS is dumped on screen as visible text instead
    of being applied — the whole app reads as broken, and nothing raises.
    This is the guard for that.
    """
    from markdown_it import MarkdownIt

    from hma import brand

    # Streamlit dedents markdown before rendering it; mirror that here.
    rendered = MarkdownIt("commonmark").render(textwrap.dedent(brand.css()))

    assert "<p>" not in rendered, (
        "CSS leaked into the page as paragraph text. The <style> tag must be "
        "the FIRST tag in the block — a <link> before it opens a type-6 HTML "
        "block that any blank line will terminate."
    )
    assert rendered.count("<style>") == 1
    assert rendered.count("</style>") == 1


def test_brand_css_font_import_is_first_rule():
    """@import is dropped by the browser unless it leads the stylesheet."""
    from hma import brand

    body = brand.css().split("<style>", 1)[1].strip()
    assert body.startswith("@import"), "the @import must be the first rule in the sheet"


def test_brand_css_uses_only_declared_palette():
    """No hand-typed hex codes. Every colour traces back to the logo."""
    from hma import brand

    declared = {getattr(brand, n).lower() for n in
                ("BLUE", "ORANGE", "INDIGO", "NAVY", "SLATE", "TINT",
                 "LINE", "CANVAS", "MUTED", "SUCCESS", "DANGER")}
    used = {h.lower() for h in re.findall(r"#[0-9A-Fa-f]{6}", brand.css())}
    stray = used - declared - {"#ffffff"}
    assert not stray, f"hex codes not traceable to brand.py: {stray}"


def test_pii_subtypes_are_described_accurately():
    """An ID and a name are different disclosure risks. Reporting 'suggests
    name' for an ID column is untrue and points at the wrong mitigation — an
    ID links records across files, a name identifies someone on sight.
    """
    from hma import survey

    df = pd.DataFrame({
        "respondent_id": ["R001"], "respondent_name": ["A. Bello"],
        "email": ["a@b.c"], "national_id": ["12345"], "age": [30],
    })
    detail = {f.column: f.detail.lower() for f in survey.detect_pii(df)}
    # "Column name suggests ..." contains the word name regardless, so assert
    # on the claim itself rather than on the substring.
    assert "suggests record id" in detail["respondent_id"]
    assert "suggests name" not in detail["respondent_id"], detail["respondent_id"]
    assert "suggests name" in detail["respondent_name"]
    assert "suggests contact" in detail["email"]
    assert "suggests id number" in detail["national_id"]
    assert "age" not in detail


def test_underscore_patterns_actually_match():
    """Patterns run against tokenised names, where underscores are spaces.

    Written `national_?id`, the pattern silently never fires — the string it
    sees is "national id". Every underscore pattern was dead this way.
    """
    from hma import survey

    for column in ["national_id", "patient_id", "hospital_no", "case_no",
                   "participant_id", "other_text", "e_mail"]:
        found = survey.detect_pii(pd.DataFrame({column: ["x"]}))
        assert found, f"{column} was not detected — check the separator in PII_PATTERNS"


def test_free_text_is_not_offered_a_meaningless_hash():
    """Hashing prose destroys the data and protects nobody."""
    from hma import survey

    finding = survey.detect_pii(pd.DataFrame({"free_text_comment": ["the clinic was far"]}))[0]
    assert finding.suggested_op is None
    assert "needs your eyes" in finding.detail

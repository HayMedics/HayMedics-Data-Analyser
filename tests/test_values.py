"""Value normalisation, questions and findings.

The value_synonyms tests are pinned to the real numbers from the healthcare
dashboard that exposed the gap: Gender arrived as 10 categories and Smoker? as
10, when both are binary. Every detector returned nothing, and the charts
rendered beautifully around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hma import findings, questions, survey  # noqa: E402
from hma.ledger import Ledger  # noqa: E402
from hma.pipeline import build_plan  # noqa: E402
from hma.profiling import profile  # noqa: E402
from scripts.make_dirty_healthcare import build  # noqa: E402


def _gender() -> pd.Series:
    return pd.Series(
        ["Male"] * 842 + ["f"] * 820 + ["m"] * 603 + ["Female"] * 541
        + ["Man"] * 287 + ["1"] * 283 + ["Woman"] * 266 + ["2"] * 237
        + ["U"] * 28 + ["Gender"] * 1,
        name="Gender",
    )


def _smoker() -> pd.Series:
    return pd.Series(
        ["N"] * 1019 + ["No"] * 985 + ["FALSE"] * 503 + ["0"] * 501
        + ["Yes"] * 265 + ["Y"] * 240 + ["TRUE"] * 149 + ["1"] * 129
        + ["?"] * 45 + ["Smoker?"] * 1,
        name="Smoker?",
    )


@pytest.fixture(scope="module")
def dirty() -> pd.DataFrame:
    return build()


@pytest.fixture(scope="module")
def cleaned(dirty) -> pd.DataFrame:
    df = dirty.copy()
    ledger = Ledger("dirty.csv")
    for p in build_plan(profile(dirty).findings, dirty).runnable:
        if p.confidence >= 0.90:
            df = ledger.apply(df, p.op, p.params, p.rationale)
    return df


# --------------------------------------------------------------------------
# Synonyms and binaries
# --------------------------------------------------------------------------

def test_binary_written_ten_ways_is_detected():
    """Yes/Y/1/TRUE and No/N/0/FALSE are two answers, not eight."""
    finding = survey.detect_value_synonyms(_smoker())
    assert finding is not None, "a 10-category binary went undetected"
    assert finding.suggested_op == "map_categories"
    assert set(finding.suggested_params["mapping"].values()) == {"Yes", "No"}


def test_sex_written_nine_ways_is_detected():
    finding = survey.detect_value_synonyms(_gender())
    assert finding is not None
    assert set(finding.suggested_params["mapping"].values()) == {"Male", "Female"}


@pytest.mark.parametrize(
    "column, value, expected",
    [("Gender", "1", "Male"), ("Gender", "2", "Female"),
     ("Smoker?", "1", "Yes"), ("Smoker?", "0", "No")],
)
def test_the_digit_ambiguity_is_resolved_by_column_name(column, value, expected):
    """'1' means Male in Gender and Yes in Smoker?.

    No amount of looking at the values resolves this — you have to read the
    header. This is the reason vocabularies carry a name hint.
    """
    series = _gender() if column == "Gender" else _smoker()
    mapping = survey.detect_value_synonyms(series).suggested_params["mapping"]
    assert mapping[value] == expected


def test_unrecognised_values_are_left_alone_not_guessed_at():
    """'U' and '?' aren't in the vocabulary. Assigning them would be inventing."""
    gender = survey.detect_value_synonyms(_gender()).suggested_params["mapping"]
    assert "U" not in gender
    smoker = survey.detect_value_synonyms(_smoker()).suggested_params["mapping"]
    assert "?" not in smoker


def test_already_consistent_binary_is_not_touched():
    """Two values is already consistent. Churning it adds a ledger entry and no value."""
    clean = pd.Series(["Yes"] * 50 + ["No"] * 50, name="Smoker?")
    assert survey.detect_value_synonyms(clean) is None


def test_numeric_columns_are_never_synonym_mapped():
    """A numeric column of 1s and 2s is numbers, not a coded sex variable."""
    ages = pd.Series([1, 2] * 40, name="prior_admissions")
    assert survey.detect_value_synonyms(ages) is None


def test_free_text_is_not_synonym_mapped():
    prose = pd.Series([f"comment number {i}" for i in range(40)], name="notes")
    assert survey.detect_value_synonyms(prose) is None


def test_question_mark_is_a_missing_code():
    """45 of them sat in the Smoker? column and nothing flagged them."""
    series = pd.Series(["Yes"] * 40 + ["No"] * 40 + ["?"] * 8, name="Smoker?")
    finding = survey.detect_sentinel_missing(series)
    assert finding is not None
    assert "?" in finding.suggested_params["values"]


def test_int64_min_is_a_missing_code():
    """Not a survey convention — it's a null forced into an integer column."""
    series = pd.Series([34, 41, -9223372036854775808, 55, 29] * 8, name="Age")
    finding = survey.detect_sentinel_missing(series)
    assert finding is not None
    assert -9223372036854775808 in finding.suggested_params["values"]


# --------------------------------------------------------------------------
# Header residue
# --------------------------------------------------------------------------

def test_repeated_header_row_is_detected():
    """A 'Gender' category with exactly one member, spelled 'Gender'."""
    df = pd.DataFrame({"Gender": ["Male", "Female"] * 20, "Age": ["30", "40"] * 20})
    df = pd.concat([df, pd.DataFrame([{"Gender": "Gender", "Age": "Age"}])],
                   ignore_index=True)
    finding = survey.detect_repeated_header_rows(df)
    assert finding is not None
    assert finding.suggested_op == "drop_repeated_header_rows"


def test_clean_table_has_no_header_residue():
    df = pd.DataFrame({"Gender": ["Male", "Female"] * 20, "Age": [30, 40] * 20})
    assert survey.detect_repeated_header_rows(df) is None


# --------------------------------------------------------------------------
# End to end on the healthcare export
# --------------------------------------------------------------------------

def test_cleaning_collapses_the_binaries(cleaned, dirty):
    """The failure from the dashboard, asserted."""
    assert dirty["Smoker?"].nunique() >= 8
    assert set(cleaned["Smoker?"].dropna().unique()) == {"Yes", "No"}
    assert set(cleaned["Readmitted 30d"].dropna().unique()) == {"Yes", "No"}


def test_cleaning_collapses_sex(cleaned, dirty):
    assert dirty["Gender"].nunique() >= 9
    remaining = set(cleaned["Gender"].dropna().unique())
    assert remaining <= {"Male", "Female", "U"}
    assert {"Male", "Female"} <= remaining


def test_header_row_is_gone(cleaned):
    assert "Gender" not in set(cleaned["Gender"].dropna().unique())
    assert len(cleaned) == 4_000


def test_int64_min_does_not_survive_into_age(cleaned):
    ages = pd.to_numeric(cleaned["Age"], errors="coerce").dropna()
    assert ages.min() > 0, "a sentinel survived into the age column"
    assert ages.max() < 130


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------

def test_questions_are_suggested_from_the_cleaned_schema(cleaned):
    suggested = questions.suggest_questions(cleaned)
    assert suggested
    assert all(q.text.strip().endswith("?") for q in suggested)
    assert all(q.why for q in suggested), "a suggestion with no reason is noise"


def test_questions_are_ranked_and_deduplicated(cleaned):
    suggested = questions.suggest_questions(cleaned)
    scores = [q.score for q in suggested]
    assert scores == sorted(scores, reverse=True)
    assert len({q.text.lower() for q in suggested}) == len(suggested)


def test_outcome_beats_column_order(cleaned):
    """`Readmitted 30d` sits left of `Outcome`; hint priority must still win."""
    assert questions._pick(list(cleaned.columns), questions.OUTCOME_HINTS) == "Outcome"


def test_no_questions_for_a_tiny_frame():
    assert questions.suggest_questions(pd.DataFrame({"a": [1, 2]})) == []


def test_questions_never_reference_an_identifier(cleaned):
    from hma import charts
    identifiers = set(charts.classify(cleaned)["identifier"])
    for q in questions.suggest_questions(cleaned, limit=50):
        assert not set(q.columns) & identifiers, f"{q.text} references an identifier"


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------

def test_findings_locate_the_planted_signal(cleaned):
    """The fixture gives the New Protocol arm genuinely lower mortality."""
    items = findings.analyse(cleaned)
    gaps = [f for f in items if "points higher" in f.headline]
    assert gaps, "the mortality gap between treatment arms was missed"
    assert any("Treatment Arm" in f.columns for f in gaps)


def test_every_gap_carries_its_denominator(cleaned):
    """'40% higher' is meaningless without knowing if that's 40 people or 4."""
    for f in findings.analyse(cleaned):
        if "points higher" in f.headline:
            assert "records" in f.detail
            assert f.caveat, "a group difference with no caveat invites a causal read"


def test_findings_never_claim_significance(cleaned):
    """No testing was done. Saying so would be a lie with a number attached."""
    items = findings.analyse(cleaned)
    blob = " ".join(f"{f.headline} {f.detail} {f.caveat}" for f in items).lower()
    for word in ("significant", "p <", "p-value", "proves", "causes"):
        assert word not in blob, f"findings overclaim: {word!r}"


def test_completeness_is_reported_first(cleaned):
    """Every number below it is computed on whatever survived."""
    assert "complete" in findings.analyse(cleaned)[0].headline


def test_thin_groups_are_not_reported_as_gaps():
    """Small groups make dramatic percentages out of nothing."""
    df = pd.DataFrame({
        "Treatment Arm": ["A"] * 200 + ["B"] * 4,
        "Outcome": ["Died"] * 20 + ["recovered"] * 180 + ["Died"] * 4,
    })
    gap = findings.group_gap(df, "Outcome", "Treatment Arm")
    assert gap is None, "a 4-record group should not produce a headline"


def test_recommendations_are_next_steps_not_conclusions(cleaned):
    items = findings.analyse(cleaned)
    recs = findings.recommendations(cleaned, items)
    assert recs
    assert any("clean.py" in r for r in recs), "should point at reproducibility"


def test_findings_on_an_empty_frame_do_not_raise():
    assert findings.analyse(pd.DataFrame()) == []


# --------------------------------------------------------------------------
# The date detector that manufactured sentinels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["2026-03-01", "01/03/2026", "15-Mar-2026",
                                   "2026/03/01", "3 January 2026"])
def test_real_dates_look_like_dates(value):
    assert survey._looks_like_a_date(value)


@pytest.mark.parametrize("value", ["49", "75.1", "3072.98", "37", "0",
                                   "-9223372036854775808", "110/79", "98"])
def test_numbers_do_not_look_like_dates(value):
    """pd.to_datetime('49') returns 1970-01-01 and reports success.

    It reads a bare integer as nanoseconds since the epoch, so "does it parse"
    cannot tell a date from a number. `110/79` is a blood pressure, and one
    separator is not a date.
    """
    assert not survey._looks_like_a_date(value)


def test_an_age_column_is_never_detected_as_dates():
    ages = pd.Series([str(a) for a in range(18, 90)] * 3, name="Age")
    assert survey.detect_date_like(ages) is None


def test_a_real_date_column_is_still_detected():
    dates = pd.Series(
        ["2026-03-01", "01/03/2026", "2026-02-11", "11-02-2026"] * 10,
        name="Admit Date",
    )
    finding = survey.detect_date_like(dates)
    assert finding is not None
    assert finding.suggested_op == "coerce_datetime"


def test_no_column_is_coerced_two_ways(dirty):
    """coerce_datetime then coerce_numeric on one column regenerates sentinels.

    NaT's internal value IS int64-min, so parsing ages as dates and converting
    back manufactures the exact missing-data code the SENTINEL stage removed.
    """
    from hma.pipeline import build_plan as plan_of

    plan = plan_of(profile(dirty).findings, dirty)
    seen: dict[str, set[str]] = {}
    for p in plan.runnable:
        column = str(p.params.get("column", ""))
        if column and p.op in {"coerce_datetime", "coerce_numeric"}:
            seen.setdefault(column, set()).add(p.op)
    for column, ops in seen.items():
        assert len(ops) == 1, f"{column} is coerced two contradictory ways: {ops}"


def test_contradiction_guard_blocks_the_loser():
    from hma.pipeline import Proposal, _block_contradictions

    proposals = [
        Proposal(op="coerce_datetime", params={"column": "Age"}, rationale="", confidence=0.86),
        Proposal(op="coerce_numeric", params={"column": "Age"}, rationale="", confidence=0.95),
    ]
    _block_contradictions(proposals)
    assert proposals[0].blocked is not None
    assert proposals[1].blocked is None

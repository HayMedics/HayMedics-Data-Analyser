"""The pipeline: stage ordering and the refusals.

The ordering here isn't taste. Each rule exists because breaking it corrupts
something specific and silently — the data still looks fine, the numbers are
just wrong. These tests pin the sequence so a future refactor can't quietly
reorder it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hma.ops import REGISTRY, Stage  # noqa: E402
from hma.pipeline import Proposal, build_plan, check_order, guard, order_key  # noqa: E402
from hma.profiling import profile  # noqa: E402
from scripts.make_messy_survey import build  # noqa: E402


@pytest.fixture(scope="module")
def messy() -> pd.DataFrame:
    return build()


@pytest.fixture(scope="module")
def plan(messy):
    return build_plan(profile(messy).findings, messy)


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------

def test_plan_is_in_stage_order(plan):
    stages = [REGISTRY[p.op].stage for p in plan.proposals]
    assert stages == sorted(stages)


def test_stage_sequence_matches_the_conventional_method():
    """Structure, sentinels, types, standardise, dedupe, missing, outliers...

    Written out so a reordering shows up as a failing test rather than as a
    subtly wrong dataset six months from now.
    """
    order = [s.name for s in sorted(Stage)]
    assert order.index("STRUCTURE") < order.index("SENTINEL")
    assert order.index("SENTINEL") < order.index("TYPES"), (
        "coerce before recoding and to_numeric eats 'Prefer not to say'"
    )
    assert order.index("TYPES") < order.index("STANDARDISE")
    assert order.index("STANDARDISE") < order.index("DEDUPLICATE"), (
        "'  Lagos' and 'lagos' are the same respondent — normalise first"
    )
    assert order.index("DEDUPLICATE") < order.index("MISSING"), (
        "imputing before dedupe lets duplicate rows drag the median"
    )
    assert order.index("MISSING") < order.index("OUTLIERS")
    assert order.index("OUTLIERS") < order.index("VALIDATE")


def test_every_stage_explains_itself():
    """A stage that can't say why it's there can't be argued with."""
    for stage in Stage:
        assert stage.label and stage.why, f"{stage.name} has no explanation"
        assert len(stage.why) > 25, f"{stage.name}'s reason is too thin to be useful"


def test_check_order_accepts_a_correct_sequence():
    assert check_order([
        "collapse_multiselect", "recode_sentinel_missing",
        "coerce_numeric", "strip_whitespace", "drop_duplicates", "fill_missing",
    ]) == []


def test_check_order_catches_dedupe_before_standardise():
    violations = check_order(["drop_duplicates", "strip_whitespace"])
    assert violations
    assert "strip_whitespace" in violations[0]


def test_check_order_catches_types_before_sentinels():
    violations = check_order(["coerce_numeric", "recode_sentinel_missing"])
    assert violations, "recoding 999 after to_numeric is too late — it's already averaged in"


def test_check_order_ignores_unknown_ops():
    assert check_order(["not_a_real_op"]) == []


def test_order_key_is_stable_for_equal_stages():
    a = Proposal(op="strip_whitespace", params={"column": "a"}, rationale="", confidence=0.9)
    b = Proposal(op="map_categories", params={"column": "b", "mapping": {}}, rationale="", confidence=0.9)
    assert order_key(a) is not None and order_key(b) is not None


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

def test_guard_blocks_imputing_skip_logic():
    """The single most common way a survey analysis goes quietly wrong."""
    proposal = Proposal(
        op="fill_missing",
        params={"column": "Q9_followup", "method": "mode"},
        rationale="fill the gaps",
        confidence=0.99,
    )
    reason = guard(proposal, never_impute={"Q9_followup"})
    assert reason and "never asked" in reason


def test_guard_blocks_imputing_multiselect_members():
    reason = guard(
        Proposal(op="fill_missing", params={"column": "Q7_Part_3", "method": "mode"},
                 rationale="", confidence=0.99),
        never_impute={"Q7_Part_3"},
    )
    assert reason


def test_guard_allows_imputing_an_ordinary_column():
    assert guard(
        Proposal(op="fill_missing", params={"column": "age", "method": "median"},
                 rationale="", confidence=0.9),
        never_impute={"Q9_followup"},
    ) is None


def test_guard_rejects_nonsense_clip_bounds():
    assert guard(
        Proposal(op="clip_outliers", params={"column": "age", "lower": 120, "upper": 0},
                 rationale="", confidence=0.9),
        never_impute=set(),
    )


def test_guard_binds_our_own_plans_too(messy, plan):
    """A rule that only constrains the LLM is a rule we don't believe.

    The deterministic planner must not be proposing anything its own guard
    would reject.
    """
    for proposal in plan.proposals:
        assert guard(proposal, plan.never_impute) is None, (
            f"our own planner proposed something the guard rejects: {proposal.op}"
        )


def test_skip_logic_columns_are_marked_never_impute(plan):
    assert "Q9_followup" in plan.never_impute
    assert any(c.startswith("Q7_Part_") for c in plan.never_impute)


def test_plan_never_imputes_skip_logic_columns(plan):
    for proposal in plan.proposals:
        if proposal.op == "fill_missing":
            assert proposal.params["column"] not in plan.never_impute


def test_the_guard_rule_exists_in_exactly_one_place():
    """The LLM path once had its own copy of never_impute, with the kinds
    hardcoded. Adding a kind to pipeline.py would have silently left the model
    unguarded. A safety rule with two copies is a safety rule that will drift.

    The prompt is allowed to name the rule — that's how the model is told
    about it. What's banned is re-deriving it in code.
    """
    import ast

    source = (ROOT / "src" / "hma" / "agents.py").read_text()
    assert "never_impute_columns(" in source, (
        "agents.py must call the shared function, not re-derive the rule"
    )

    # Strip every string literal, so the prompt text doesn't trip this.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    code_only = ast.unparse(tree)

    assert "structural_missing" not in code_only, (
        "agents.py re-derives the protected kinds in code — call "
        "never_impute_columns() so the rule lives in one place"
    )


def test_both_planners_protect_the_same_columns(messy):
    """Deterministic and LLM paths must agree on what's untouchable."""
    from hma.pipeline import never_impute_columns

    findings = profile(messy).findings
    assert build_plan(findings, messy).never_impute == never_impute_columns(findings)


def test_never_impute_survives_a_new_protected_kind():
    """Adding a kind to NEVER_IMPUTE_KINDS must protect it everywhere at once."""
    from hma.pipeline import NEVER_IMPUTE_KINDS, never_impute_columns
    from hma.survey import Finding

    assert "structural_missing" in NEVER_IMPUTE_KINDS
    protected = never_impute_columns([
        Finding(kind="structural_missing", column="Q9", confidence=0.9, detail=""),
        Finding(kind="multiselect_group", column="Q7", confidence=0.9, detail="",
                suggested_params={"members": ["Q7_Part_1", "Q7_Part_2"]}),
    ])
    assert protected == {"Q9", "Q7", "Q7_Part_1", "Q7_Part_2"}

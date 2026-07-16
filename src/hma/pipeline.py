"""The cleaning pipeline.

This module owns one idea: **the order is the method**.

Anyone can list the conventional cleaning steps. The value is in enforcing
them, because every ordering mistake is silent. Nothing raises when you
average a column that still contains 999. Nothing raises when you deduplicate
before you normalise casing. You just get a wrong answer that looks fine.

So the order isn't a comment or a convention here. `Stage` is attached to every
operation in the registry, the planner sorts by it, and `check_order()` will
tell you the moment a ledger violates it.

The conventional sequence, and what each step is protecting:

    1  Fix structure              tidy shape: one row per observation
    2  Recode disguised missing   999 / "Prefer not to say" -> NaN
    3  Fix types                  text -> number / date, restore ordinality
    4  Standardise values         whitespace, casing, category spellings
    5  Remove duplicates          only meaningful once 4 has run
    6  Handle missing             impute, with skip-logic held back
    7  Handle outliers            range and domain rules
    8  Validate                   cross-field logic
    9  De-identify                last, before export

Transformation (deriving, aggregating, encoding, scaling) deliberately lives
outside this module. Those are modelling decisions and they belong to the
analyst, not to a cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import settings
from .ops import REGISTRY, Stage
from .survey import Finding

# Columns flagged as skip logic must never reach an imputer. This is the one
# rule in the whole project worth being dogmatic about: filling a skip-logic
# column manufactures opinions for people who were never asked the question.
NEVER_IMPUTE_KINDS = {"structural_missing", "multiselect_group"}


@dataclass
class Proposal:
    """One suggested operation, and everything needed to judge it."""

    op: str
    params: dict
    rationale: str
    confidence: float = 0.0
    blocked: str | None = None  # set when a guard refuses it

    @property
    def stage(self) -> Stage:
        return REGISTRY[self.op].stage

    @property
    def target(self) -> str:
        for key in ("column", "stem", "start_column"):
            if key in self.params:
                return str(self.params[key])
        if "columns" in self.params:
            return ", ".join(map(str, self.params["columns"]))
        return "<table>"

    @property
    def is_safe_to_auto_apply(self) -> bool:
        return self.blocked is None and self.confidence >= settings.auto_apply_threshold


@dataclass
class Plan:
    """An ordered cleaning plan, grouped into the conventional stages."""

    proposals: list[Proposal] = field(default_factory=list)
    never_impute: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.proposals)

    def __iter__(self):
        return iter(self.proposals)

    @property
    def runnable(self) -> list[Proposal]:
        return [p for p in self.proposals if p.blocked is None]

    @property
    def blocked(self) -> list[Proposal]:
        return [p for p in self.proposals if p.blocked is not None]

    def by_stage(self) -> dict[Stage, list[Proposal]]:
        """Proposals grouped by stage, in execution order, empty stages omitted."""
        grouped: dict[Stage, list[Proposal]] = {}
        for proposal in self.proposals:
            grouped.setdefault(proposal.stage, []).append(proposal)
        return {stage: grouped[stage] for stage in Stage if stage in grouped}


def never_impute_columns(findings: list[Finding]) -> set[str]:
    """Columns where filling a blank would invent an answer.

    This lives in exactly one place on purpose. The deterministic planner and
    the LLM planner both call it, so there is no way to protect a column in
    one path and forget it in the other — a safety rule with two copies is a
    safety rule that will drift, and the drift is invisible until someone
    publishes an imputed skip-logic column.
    """
    structural = {f.column for f in findings if f.kind in NEVER_IMPUTE_KINDS}
    members = {
        member
        for f in findings
        if f.kind == "multiselect_group"
        for member in f.suggested_params.get("members", [])
    }
    return structural | members


def order_key(proposal: Proposal) -> tuple:
    """Sort by stage first, then by confidence within the stage.

    Stage dominates. A 99%-confidence imputation still runs after a
    60%-confidence sentinel recode, because doing it the other way round
    averages 999 into someone's real age.
    """
    return (proposal.stage.value, -proposal.confidence, proposal.op)


def build_plan(findings: list[Finding], df: pd.DataFrame | None = None) -> Plan:
    """Turn deterministic findings into a correctly ordered plan.

    No LLM, no tokens. This is the plan you want most of the time — it's
    reproducible, it's free, and it can't hallucinate a column name.
    """
    proposals: list[Proposal] = []
    never_impute = never_impute_columns(findings)

    for finding in findings:
        op = finding.suggested_op
        if not op or op not in REGISTRY:
            continue

        proposal = Proposal(
            op=op,
            params=dict(finding.suggested_params),
            rationale=finding.detail,
            confidence=finding.confidence,
        )
        proposal.blocked = guard(proposal, never_impute)
        proposals.append(proposal)

    proposals.sort(key=order_key)
    _block_contradictions(proposals)
    return Plan(proposals, never_impute=never_impute)


# Ops that cannot both be right about the same column.
CONTRADICTORY: tuple[frozenset[str], ...] = (
    frozenset({"coerce_datetime", "coerce_numeric"}),
    frozenset({"deidentify", "drop_column"}),
)


def _block_contradictions(proposals: list[Proposal]) -> None:
    """Stop a column being coerced two incompatible ways.

    `Age` was once proposed for both coerce_datetime and coerce_numeric. Run in
    stage order that reads: parse ages as dates, then convert the dates back to
    numbers — and NaT's internal value is int64-min, so every blank age came
    back as -9223372036854775808. The sentinels weren't surviving the clean,
    they were being manufactured by it.

    Lower confidence loses. Both stay visible, so the ledger shows the choice
    rather than hiding it.
    """
    for pair in CONTRADICTORY:
        by_column: dict[str, list[Proposal]] = {}
        for proposal in proposals:
            if proposal.op in pair and proposal.blocked is None:
                column = str(proposal.params.get("column", ""))
                if column:
                    by_column.setdefault(column, []).append(proposal)

        for column, clashing in by_column.items():
            if len({p.op for p in clashing}) < 2:
                continue
            winner = max(clashing, key=lambda p: p.confidence)
            for loser in clashing:
                if loser is winner:
                    continue
                loser.blocked = (
                    f"{loser.op} contradicts {winner.op} on {column}, which is "
                    f"more confident ({winner.confidence:.0%} vs "
                    f"{loser.confidence:.0%}). A column cannot be both."
                )


def guard(proposal: Proposal, never_impute: set[str]) -> str | None:
    """Refuse proposals that are wrong regardless of who suggested them.

    Applies to the LLM's plans and to our own deterministic ones equally. A
    rule that only constrains the model is a rule you don't actually believe.
    """
    if proposal.op == "fill_missing":
        column = str(proposal.params.get("column", ""))
        if column in never_impute:
            return (
                f"{column} is skip-logic or multi-select. Filling it would "
                f"invent answers for people who were never asked."
            )

    if proposal.op == "clip_outliers":
        lower, upper = proposal.params.get("lower"), proposal.params.get("upper")
        if lower is None or upper is None or lower >= upper:
            return "clip_outliers needs a lower bound below its upper bound."

    return None


def check_order(op_names: list[str]) -> list[str]:
    """Report any place a sequence of ops runs out of conventional order.

    Returns human-readable violations, empty list when clean. The ledger runs
    this so a hand-applied session can't quietly end up in the wrong order and
    still claim to have followed the method.
    """
    violations: list[str] = []
    highest = Stage.STRUCTURE
    highest_op = None

    for name in op_names:
        if name not in REGISTRY:
            continue
        stage = REGISTRY[name].stage
        if stage < highest:
            violations.append(
                f"{name} ({stage.label}) ran after {highest_op} ({highest.label}). "
                f"{stage.why}"
            )
        else:
            highest, highest_op = stage, name

    return violations

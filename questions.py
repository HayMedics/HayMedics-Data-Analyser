"""Questions worth asking, read off the cleaned schema.

A blank box that says "ask me anything" is the least helpful thing a tool can
show someone. You have to already know what's in the data to know what to type,
and if you knew that you wouldn't need the tool.

So this reads the roles the columns actually have and proposes questions the
data can genuinely answer. Deterministic — no model, no tokens. The LLM is only
needed to turn a chosen question into SQL, and if it's unavailable you can still
read the list and learn what shape of question this dataset supports.

Nothing here is asked automatically. They're offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import charts

# Column-name fragments that usually mark the thing a study is about.
OUTCOME_HINTS = (
    "outcome", "died", "death", "mortality", "survived", "readmit", "relapse",
    "complication", "recovered", "result", "status", "cured", "success",
)
EXPOSURE_HINTS = (
    "arm", "treatment", "group", "intervention", "protocol", "regimen",
    "exposure", "cohort", "allocation",
)
COST_HINTS = ("bill", "cost", "charge", "price", "fee", "amount", "expense", "spend")


@dataclass
class Question:
    """One question, and why it's worth the ask."""

    text: str
    why: str
    columns: list[str] = field(default_factory=list)
    score: float = 0.0
    category: str = "explore"


def _matches(name: str, hints: tuple[str, ...]) -> bool:
    lowered = str(name).lower()
    return any(h in lowered for h in hints)


def _pick(columns: list[str], hints: tuple[str, ...]) -> str | None:
    """First column matching the highest-priority hint.

    Iterates hints outermost, not columns. Scanning columns first means the
    answer depends on the file's column order: `Readmitted 30d` would be
    chosen as the outcome over `Outcome` purely because it sits to the left.
    The hint tuples are written in priority order, so honour that instead.
    """
    for hint in hints:
        for column in columns:
            if hint in str(column).lower():
                return column
    return None


def suggest_questions(df: pd.DataFrame, limit: int = 10) -> list[Question]:
    """Questions this dataset can actually answer, best first.

    Built from the CLEANED frame on purpose. Suggest "average age by ward"
    while age still contains 999 and the tool is inviting you to compute a
    wrong number and believe it.
    """
    if df.empty or len(df) < 10:
        return []

    roles = charts.classify(df)
    categorical = [c for c in roles["categorical"] if 2 <= df[c].nunique(dropna=True) <= 12]
    numeric = roles["numeric"]
    ordered = roles["ordered"]
    datetime_cols = roles["datetime"]

    outcome = _pick(categorical, OUTCOME_HINTS)
    exposure = _pick(categorical, EXPOSURE_HINTS)
    cost = _pick(numeric, COST_HINTS)

    questions: list[Question] = []

    # The comparison the study was probably designed to make.
    if outcome and exposure:
        questions.append(Question(
            text=f"What share of each {exposure} had each {outcome}?",
            why=(
                "This is the comparison the dataset exists to support. Ask for "
                "row percentages, not counts — if the groups are different sizes, "
                "raw counts will mislead you."
            ),
            columns=[exposure, outcome],
            score=1.0,
            category="headline",
        ))

    if outcome:
        for group in [c for c in categorical if c != outcome][:2]:
            questions.append(Question(
                text=f"What share of each {group} had each {outcome}?",
                why=(
                    f"Breaks the outcome down by {group}. A gap here is worth "
                    f"reporting — but it isn't proof of cause, because groups "
                    f"differ in more ways than one."
                ),
                columns=[group, outcome],
                score=0.9,
                category="comparison",
            ))
        questions.append(Question(
            text=f"How many records fall into each {outcome}, and how many are blank?",
            why=(
                f"Before comparing anything, find out how complete {outcome} is. "
                f"A comparison built on a half-empty outcome column is a "
                f"comparison of who answered, not of what happened."
            ),
            columns=[outcome],
            score=0.95,
            category="sanity",
        ))

    if cost and (exposure or categorical):
        group = exposure or categorical[0]
        questions.append(Question(
            text=f"What is the median {cost} by {group}?",
            why=(
                "Median, not mean — cost distributions have long tails, and a "
                "handful of expensive cases will drag an average somewhere no "
                "actual patient sits."
            ),
            columns=[cost, group],
            score=0.85,
            category="headline",
        ))

    for scale in ordered[:2]:
        questions.append(Question(
            text=f"What is the median {scale}, overall and by {categorical[0]}?"
                 if categorical else f"What is the median {scale}?",
            why=(
                f"{scale} is an ordered scale, so median and quartiles are the "
                f"honest summary. A mean of an ordinal assumes the gap between "
                f"'agree' and 'strongly agree' equals the gap between 'neutral' "
                f"and 'agree', which nobody has established."
            ),
            columns=[scale] + categorical[:1],
            score=0.8,
            category="comparison",
        ))

    for number in numeric[:2]:
        if number == cost:
            continue
        if categorical:
            questions.append(Question(
                text=f"What is the mean and median {number} by {categorical[0]}?",
                why=(
                    "Ask for both. Where the mean and median diverge, the "
                    "distribution is skewed and the mean is the wrong summary."
                ),
                columns=[number, categorical[0]],
                score=0.75,
                category="comparison",
            ))
        questions.append(Question(
            text=f"What are the minimum, maximum and quartiles of {number}?",
            why=(
                "The fastest way to catch a cleaning miss. An impossible "
                "minimum or maximum here means a missing-data code survived "
                "and is now sitting inside your averages."
            ),
            columns=[number],
            score=0.7,
            category="sanity",
        ))

    if datetime_cols and categorical:
        questions.append(Question(
            text=f"How many records per month, by {categorical[0]}?",
            why=(
                "Shows collection over time. A gap or a spike usually means a "
                "site stopped reporting or a batch was entered at once — both "
                "change how you read everything else."
            ),
            columns=[datetime_cols[0], categorical[0]],
            score=0.7,
            category="sanity",
        ))

    for column in categorical[:3]:
        questions.append(Question(
            text=f"How many records in each {column}?",
            why=(
                "Check the balance. A category holding almost everyone gives you "
                "no power to compare, and a tiny one won't survive being split "
                "further."
            ),
            columns=[column],
            score=0.6,
            category="explore",
        ))

    seen: set[str] = set()
    unique: list[Question] = []
    for question in sorted(questions, key=lambda q: -q.score):
        if question.text.lower() in seen:
            continue
        seen.add(question.text.lower())
        unique.append(question)
    return unique[:limit]

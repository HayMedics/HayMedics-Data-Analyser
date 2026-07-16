"""What the data says.

Everything else in this package reports on the data's condition — what was
wrong, what was changed. That report is for you and your reviewer. It is not
what you hand a director, because it answers a question they didn't ask.

This module computes findings: the numbers themselves, ranked by whether a
decision might turn on them.

Two rules it holds to.

**Descriptive, not inferential.** No p-values, no significance claims. A gap
between two groups gets reported as a gap, with both denominators visible.
Whether it's causal depends on how people ended up in those groups, and that's
a question about the study design, not about this table. A tool that prints
p < 0.05 next to an observational cross-tab is teaching its user to overclaim.

**Every finding carries its denominator.** "40% higher" means nothing without
knowing whether that's 40 patients or 4. Small groups produce dramatic
percentages and no information, so anything computed on a thin group says so
in the same breath.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import charts

# Below this, a percentage is noise dressed as a number.
MIN_GROUP = 30
# A gap smaller than this is not worth a decision-maker's attention.
NOTABLE_GAP_PP = 8.0

OUTCOME_HINTS = (
    "outcome", "died", "death", "mortality", "survived", "readmit", "relapse",
    "complication", "recovered", "result", "status",
)
EXPOSURE_HINTS = (
    "arm", "treatment", "group", "intervention", "protocol", "regimen",
    "exposure", "cohort", "allocation",
)
BAD_OUTCOME_VALUES = (
    "died", "death", "dead", "deceased", "yes", "true", "1", "relapsed",
    "readmitted", "complication", "failed", "worsened",
)


@dataclass
class Finding:
    """One thing the data says, with the numbers behind it."""

    headline: str
    detail: str
    table: pd.DataFrame | None = None
    columns: list[str] = field(default_factory=list)
    importance: float = 0.0
    caveat: str = ""


def _norm(value: object) -> str:
    return str(value).strip().lower()


def _pick(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        for column in columns:
            if hint in str(column).lower():
                return column
    return None


def _bad_value(series: pd.Series) -> object | None:
    """Which level of this outcome is the one nobody wants."""
    for value in series.dropna().unique():
        if _norm(value) in BAD_OUTCOME_VALUES:
            return value
    return None


def completeness(df: pd.DataFrame) -> Finding:
    """How much of the table is actually there.

    First, always. Every number below it is computed on whatever survived, and
    a reader who doesn't know that will read them as if they describe everyone.
    """
    total = df.size
    missing = int(df.isna().sum().sum())
    worst = df.isna().mean().sort_values(ascending=False)
    thin = worst[worst > 0.2]

    detail = (
        f"{len(df):,} records across {len(df.columns)} fields. "
        f"{(1 - missing / max(total, 1)):.1%} of cells are populated."
    )
    if len(thin):
        listed = ", ".join(f"{c} ({p:.0%} blank)" for c, p in thin.head(4).items())
        detail += (
            f" Fields with substantial gaps: {listed}. Any figure below that "
            f"uses these describes only the records where they were recorded."
        )

    return Finding(
        headline=f"{len(df):,} records, {(1 - missing / max(total, 1)):.0%} complete",
        detail=detail,
        table=(
            thin.head(8).mul(100).round(1).rename("% blank").to_frame().reset_index(
                names="Field"
            )
            if len(thin) else None
        ),
        importance=1.0,
    )


def outcome_split(df: pd.DataFrame, outcome: str) -> Finding | None:
    """The headline rate. What happened, to how many."""
    counts = df[outcome].value_counts(dropna=False)
    known = df[outcome].notna().sum()
    if known < MIN_GROUP:
        return None

    table = (
        df[outcome].value_counts(dropna=True).rename("Records").to_frame()
        .assign(**{"% of known": lambda t: (t["Records"] / known * 100).round(1)})
        .reset_index(names=outcome)
    )
    top = table.iloc[0]
    blank = int(counts.get(pd.NA, 0)) + int(df[outcome].isna().sum())

    detail = (
        f"Of {known:,} records where {outcome} was recorded, "
        f"{top['% of known']:.1f}% are '{top[outcome]}' ({int(top['Records']):,})."
    )
    if blank:
        detail += (
            f" {blank:,} records ({blank / len(df):.1%}) have no {outcome} at all "
            f"and are excluded from every percentage here."
        )

    return Finding(
        headline=f"{outcome}: {top['% of known']:.0f}% {top[outcome]}",
        detail=detail,
        table=table,
        columns=[outcome],
        importance=0.95,
    )


def group_gap(df: pd.DataFrame, outcome: str, group: str) -> Finding | None:
    """Does the outcome rate differ across this group?

    Reports the gap in percentage points, with both denominators. Deliberately
    stops short of claiming why.
    """
    bad = _bad_value(df[outcome])
    if bad is None:
        return None

    subset = df[[group, outcome]].dropna()
    if len(subset) < MIN_GROUP * 2:
        return None

    rates = (
        subset.assign(_bad=lambda t: t[outcome].map(_norm) == _norm(bad))
        .groupby(group, observed=True)
        .agg(Records=("_bad", "size"), Rate=("_bad", "mean"))
    )
    rates = rates[rates["Records"] >= MIN_GROUP]
    if len(rates) < 2:
        return None

    rates["Rate"] = (rates["Rate"] * 100).round(1)
    rates = rates.sort_values("Rate", ascending=False)
    high, low = rates.iloc[0], rates.iloc[-1]
    gap = float(high["Rate"] - low["Rate"])
    if gap < NOTABLE_GAP_PP:
        return None

    return Finding(
        headline=(
            f"{bad} rate is {gap:.0f} points higher in "
            f"{high.name} than {low.name}"
        ),
        detail=(
            f"Among {group} = {high.name}, {high['Rate']:.1f}% of "
            f"{int(high['Records']):,} records ended in '{bad}'. Among "
            f"{low.name} it is {low['Rate']:.1f}% of {int(low['Records']):,}. "
            f"That is a gap of {gap:.1f} percentage points."
        ),
        table=rates.reset_index().rename(columns={"Rate": f"% {bad}"}),
        columns=[group, outcome],
        importance=0.9 + min(gap / 200, 0.09),
        caveat=(
            f"This is a difference, not a cause. {group} groups may differ in "
            f"age, severity or anything else that was never measured here. "
            f"Treat it as a question worth designing a study around, not as an "
            f"answer."
        ),
    )


def concentration(df: pd.DataFrame, column: str) -> Finding | None:
    """One category holding almost everyone is a finding about the data."""
    counts = df[column].value_counts(dropna=True)
    if counts.empty or counts.sum() < MIN_GROUP:
        return None

    share = counts.iloc[0] / counts.sum()
    if share < 0.6 or len(counts) < 2:
        return None

    return Finding(
        headline=f"{share:.0%} of records share one {column}: {counts.index[0]}",
        detail=(
            f"'{counts.index[0]}' accounts for {counts.iloc[0]:,} of "
            f"{counts.sum():,} records. Any comparison across {column} is really "
            f"a comparison between that group and a much smaller remainder."
        ),
        table=counts.head(6).rename("Records").to_frame().reset_index(names=column),
        columns=[column],
        importance=0.7,
    )


def numeric_summary(df: pd.DataFrame, column: str) -> Finding | None:
    """Median and spread, plus a flag when the mean is the wrong summary."""
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < MIN_GROUP:
        return None

    median, mean = float(series.median()), float(series.mean())
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    skewed = median != 0 and abs(mean - median) / max(abs(median), 1e-9) > 0.15

    detail = (
        f"Median {median:,.1f} (IQR {q1:,.1f}–{q3:,.1f}), across {len(series):,} "
        f"records where {column} was recorded."
    )
    if skewed:
        detail += (
            f" The mean is {mean:,.1f} — well away from the median, so the "
            f"distribution has a tail. Report the median; the mean here sits "
            f"where few records actually are."
        )

    return Finding(
        headline=f"Median {column}: {median:,.1f}",
        detail=detail,
        columns=[column],
        importance=0.65 + (0.1 if skewed else 0),
    )


def analyse(df: pd.DataFrame, limit: int = 12) -> list[Finding]:
    """Everything the data says, ranked by whether a decision turns on it."""
    if df.empty or len(df) < MIN_GROUP:
        return []

    roles = charts.classify(df)
    categorical = [
        c for c in roles["categorical"] + roles["flag"]
        if 2 <= df[c].nunique(dropna=True) <= 12
    ]
    numeric = roles["numeric"]

    findings = [completeness(df)]

    outcome = _pick(categorical, OUTCOME_HINTS)
    exposure = _pick([c for c in categorical if c != outcome], EXPOSURE_HINTS)

    if outcome:
        split = outcome_split(df, outcome)
        if split:
            findings.append(split)

        ordered_groups = ([exposure] if exposure else []) + [
            c for c in categorical if c not in {outcome, exposure}
        ]
        for group in ordered_groups[:5]:
            gap = group_gap(df, outcome, group)
            if gap:
                findings.append(gap)

    for column in categorical[:4]:
        conc = concentration(df, column)
        if conc:
            findings.append(conc)

    for column in numeric[:3]:
        summary = numeric_summary(df, column)
        if summary:
            findings.append(summary)

    return sorted(findings, key=lambda f: -f.importance)[:limit]


def recommendations(df: pd.DataFrame, findings: list[Finding]) -> list[str]:
    """What to do about it — phrased as next steps, not conclusions."""
    out: list[str] = []

    gaps = [f for f in findings if "points higher" in f.headline]
    if gaps:
        top = gaps[0]
        out.append(
            f"**Investigate the gap.** {top.headline}. Before acting on it, check "
            f"whether the groups are comparable on the things that matter — this "
            f"data can show the difference but cannot explain it."
        )

    blanks = df.isna().mean()
    thin = blanks[blanks > 0.3]
    if len(thin):
        out.append(
            f"**Fix collection at source for {', '.join(list(thin.index)[:3])}.** "
            f"Each is over 30% blank, so every figure using them describes only "
            f"the records where someone filled them in. That is rarely a random "
            f"subset of people."
        )

    small = [
        c for c in df.columns
        if df[c].dtype == object and 1 < df[c].nunique(dropna=True) <= 12
        and (df[c].value_counts().min() < MIN_GROUP)
    ]
    if small:
        out.append(
            f"**Don't split {small[0]} any further.** Its smallest category is "
            f"under {MIN_GROUP} records; breaking it down again produces "
            f"percentages that move several points when one record changes."
        )

    out.append(
        "**Attach the cleaning script to whatever you submit.** The figures here "
        "come from a cleaned copy, and `clean.py` regenerates that copy from the "
        "raw export. Without it, nobody reading this can check the numbers."
    )
    return out

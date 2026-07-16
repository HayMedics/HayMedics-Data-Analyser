"""Deterministic profiling. Zero LLM calls, zero tokens, zero cost.

Every number in here is computed from the data. Nothing is inferred by a
language model, which means profiling a 2 GB file costs exactly the same as
profiling a 2 KB one: nothing.

The LLM only gets involved later, and only to reason about *this* summary —
never about raw rows. That keeps the prompt small and the bill flat regardless
of dataset size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import survey


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    n_unique_with_na: int
    sample: list

    @property
    def is_constant(self) -> bool:
        """One value in every row, counting blank as a value.

        Counting with dropna=True instead would call a ticked/not-ticked
        multi-select column constant and dock the quality score for it.
        """
        return self.n_unique_with_na <= 1

    @property
    def is_free_text(self) -> bool:
        return self.dtype == "object" and self.n_unique > 50


@dataclass
class Profile:
    n_rows: int
    n_cols: int
    memory_mb: float
    n_duplicate_rows: int
    columns: list[ColumnProfile]
    findings: list[survey.Finding] = field(default_factory=list)

    @property
    def quality_score(self) -> int:
        """A 0-100 score. Deliberately simple and explainable.

        No model, no magic — just four penalties anyone can audit and argue
        with. A score you can't explain is a score nobody should trust.
        """
        if self.n_rows == 0 or self.n_cols == 0:
            return 0

        total_cells = self.n_rows * self.n_cols
        missing = sum(c.n_missing for c in self.columns)

        missing_penalty = (missing / total_cells) * 40
        dup_penalty = (self.n_duplicate_rows / self.n_rows) * 20
        constant_penalty = (
            sum(1 for c in self.columns if c.is_constant) / self.n_cols
        ) * 15
        sentinel_penalty = min(
            sum(1 for f in self.findings if f.kind == "sentinel_missing") * 3, 25
        )

        score = 100 - missing_penalty - dup_penalty - constant_penalty - sentinel_penalty
        return max(0, min(100, round(score)))

    @property
    def score_breakdown(self) -> dict[str, str]:
        total_cells = max(self.n_rows * self.n_cols, 1)
        missing = sum(c.n_missing for c in self.columns)
        return {
            "Missing cells": f"{missing:,} of {total_cells:,} ({missing / total_cells:.1%})",
            "Duplicate rows": f"{self.n_duplicate_rows:,}",
            "Constant columns": f"{sum(1 for c in self.columns if c.is_constant)}",
            "Sentinel-code columns": f"{sum(1 for f in self.findings if f.kind == 'sentinel_missing')}",
        }

    def to_prompt(self, max_columns: int = 40) -> str:
        """Compact text summary for the LLM. This — not the data — is the context."""
        lines = [
            f"Dataset: {self.n_rows:,} rows x {self.n_cols} columns",
            f"Quality score: {self.quality_score}/100",
            f"Duplicate rows: {self.n_duplicate_rows:,}",
            "",
            "COLUMNS:",
        ]
        for col in self.columns[:max_columns]:
            sample = ", ".join(str(s)[:24] for s in col.sample[:3])
            lines.append(
                f"- {col.name} | {col.dtype} | {col.pct_missing:.0%} missing "
                f"| {col.n_unique} unique | e.g. {sample}"
            )
        if self.n_cols > max_columns:
            lines.append(f"... and {self.n_cols - max_columns} more columns")

        if self.findings:
            lines += ["", "SURVEY FINDINGS (detected deterministically):"]
            lines += [f"- {f}" for f in self.findings[:25]]
        return "\n".join(lines)


def profile(df: pd.DataFrame, run_survey_scan: bool = True) -> Profile:
    columns = []
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                n_missing=n_missing,
                pct_missing=n_missing / len(df) if len(df) else 0.0,
                n_unique=int(series.nunique(dropna=True)),
                n_unique_with_na=int(series.nunique(dropna=False)),
                sample=series.dropna().unique()[:5].tolist(),
            )
        )

    try:
        n_dupes = int(df.duplicated().sum())
    except TypeError:
        # Unhashable cells (lists, dicts) — rare but real in JSON exports.
        n_dupes = 0

    return Profile(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
        n_duplicate_rows=n_dupes,
        columns=columns,
        findings=survey.scan(df) if run_survey_scan else [],
    )

"""The compute layer: DuckDB.

Most tools in this space hold the whole frame in pandas, which is why they all
list "larger dataset support" as future work. DuckDB reads Parquet/CSV larger
than RAM, so starting here means never hitting that wall.

The LLM writes SQL. It does not compute. It never sees a single data row —
only the schema. Then this module runs the SQL and the *result* is the answer.
That's what "computed from your data, not guessed" has to mean to be true.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

# Anything that isn't a read. DuckDB has no permissions model here, so the
# guard is ours to enforce.
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|"
    r"install|load|pragma|set|call|read_csv|read_parquet|read_json)\b",
    re.IGNORECASE,
)


class UnsafeQuery(ValueError):
    """The generated SQL tried to do something other than read."""


def guard(sql: str) -> str:
    """Reject anything that isn't a single read-only statement."""
    cleaned = re.sub(r"--[^\n]*", " ", sql)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL).strip().rstrip(";")

    if not cleaned:
        raise UnsafeQuery("Empty query.")
    if ";" in cleaned:
        raise UnsafeQuery("Multiple statements are not allowed.")
    if not re.match(r"^\s*(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeQuery("Only SELECT / WITH queries are allowed.")

    hit = FORBIDDEN.search(cleaned)
    if hit:
        raise UnsafeQuery(f"Disallowed keyword: {hit.group(0)}")
    return cleaned


def run_sql(df: pd.DataFrame, sql: str, limit: int = 5_000) -> pd.DataFrame:
    """Execute read-only SQL against the frame, exposed as table `data`."""
    safe = guard(sql)
    con = duckdb.connect(":memory:")
    try:
        con.register("data", df)
        result = con.execute(safe).df()
    finally:
        con.close()

    if len(result) > limit:
        result = result.head(limit)
    return result


def schema_for_prompt(df: pd.DataFrame, max_columns: int = 60) -> str:
    """Describe the table to the LLM. Column names and types only — no rows."""
    lines = ["Table `data` columns:"]
    for name in list(df.columns)[:max_columns]:
        series = df[name]
        note = ""
        if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
            note = f"  (ordered: {' < '.join(map(str, series.cat.categories))})"
        elif series.dtype == object and series.nunique(dropna=True) <= 12:
            values = ", ".join(str(v)[:20] for v in series.dropna().unique()[:12])
            note = f"  (values: {values})"
        lines.append(f'  "{name}" {series.dtype}{note}')

    if len(df.columns) > max_columns:
        lines.append(f"  ... and {len(df.columns) - max_columns} more columns")
    return "\n".join(lines)


def suggest_chart(result: pd.DataFrame) -> str | None:
    """Pick a chart type from the result's shape. No LLM needed for this."""
    if result.empty or len(result.columns) < 2 or len(result) > 50:
        return None

    first, second = result.columns[0], result.columns[1]
    x_is_label = result[first].dtype == object or isinstance(
        result[first].dtype, pd.CategoricalDtype
    )
    y_is_number = pd.api.types.is_numeric_dtype(result[second])

    if x_is_label and y_is_number:
        return "bar"
    if pd.api.types.is_datetime64_any_dtype(result[first]) and y_is_number:
        return "line"
    if pd.api.types.is_numeric_dtype(result[first]) and y_is_number:
        return "scatter"
    return None

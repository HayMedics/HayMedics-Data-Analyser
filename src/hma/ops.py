"""The operation registry.

This is the security boundary and the honesty boundary at once.

The LLM never writes or executes code. It emits `{"op": "...", "params": {...}}`
and nothing else. If the op isn't in this registry, it doesn't run. That means
a prompt injection in a CSV cell can, at absolute worst, propose a bad *fill
strategy* — it cannot reach the filesystem.

Every Operation carries three things:
  apply  -> runs it on a DataFrame, returns (new_df, stats)
  code   -> renders it as a line of plain pandas, for the exported script
  schema -> describes its params, which becomes the LLM's tool spec

Because `code` lives next to `apply`, the exported script can't drift from
what actually ran. That's what makes the ledger reproducible rather than
merely readable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

Stats = dict[str, object]


class Stage(IntEnum):
    """The conventional cleaning order. Lower numbers run first.

    Every number here exists because running that step later corrupts
    something specific. This is the sequence itself, encoded — not a
    preference, and not a list of ops someone happened to think of first.

    STRUCTURE   fix the shape before anything reads a cell. A question-text
                row sitting above the data means every step below is looking
                at the wrong values.

    SENTINEL    999 -> NaN *before* types. Coerce first and "Prefer not to
                say" is silently eaten by to_numeric — you lose the fact that
                someone refused, which in survey work is itself a finding.

    TYPES       text -> number/date, and restore ordinality to scales. Only
                safe once the disguised missing values are already gone.

    STANDARDISE whitespace, casing, category spellings. Must precede DEDUPE.

    DEDUPLICATE two rows are not equal until STANDARDISE has run. "  Lagos"
                and "lagos" are the same place and the same respondent.

    MISSING     impute only once you know what is genuinely missing rather
                than merely disguised. Never on skip-logic columns.

    OUTLIERS    a range check is meaningless until the column is numeric.

    VALIDATE    cross-field logic, once every field is individually sound.

    PRIVACY     de-identify last. Hash or drop an identifier earlier and you
                lose the column DEDUPLICATE and VALIDATE needed.
    """

    STRUCTURE = 1
    SENTINEL = 2
    TYPES = 3
    STANDARDISE = 4
    DEDUPLICATE = 5
    MISSING = 6
    OUTLIERS = 7
    VALIDATE = 8
    PRIVACY = 9

    @property
    def label(self) -> str:
        return {
            Stage.STRUCTURE: "Fix structure",
            Stage.SENTINEL: "Recode disguised missing",
            Stage.TYPES: "Fix types",
            Stage.STANDARDISE: "Standardise values",
            Stage.DEDUPLICATE: "Remove duplicates",
            Stage.MISSING: "Handle missing",
            Stage.OUTLIERS: "Handle outliers",
            Stage.VALIDATE: "Validate",
            Stage.PRIVACY: "De-identify",
        }[self]

    @property
    def why(self) -> str:
        return {
            Stage.STRUCTURE: "Everything below reads the wrong cells until the shape is right.",
            Stage.SENTINEL: "Before types, or to_numeric silently eats every refusal.",
            Stage.TYPES: "Safe only once disguised missing values are already NaN.",
            Stage.STANDARDISE: "Before dedupe, or ' Lagos' and 'lagos' survive as two rows.",
            Stage.DEDUPLICATE: "Rows aren't comparable until their values are normalised.",
            Stage.MISSING: "Impute only what is genuinely missing, never skip logic.",
            Stage.OUTLIERS: "A range check needs a numeric column to check against.",
            Stage.VALIDATE: "Cross-field logic, once each field is individually sound.",
            Stage.PRIVACY: "Last: dedupe and validation need the identifiers.",
        }[self]


@dataclass(frozen=True)
class Operation:
    name: str
    apply: Callable[..., tuple[pd.DataFrame, Stats]]
    code: Callable[..., str]
    doc: str
    schema: dict[str, str]
    stage: Stage


REGISTRY: dict[str, Operation] = {}


def operation(
    name: str,
    doc: str,
    schema: dict[str, str],
    code: Callable[..., str],
    stage: Stage,
):
    """Register an op: runner, renderer, schema and stage, together.

    `stage` is mandatory. An operation that doesn't know where it belongs in
    the sequence is an operation nobody can order correctly.
    """

    def decorator(fn):
        REGISTRY[name] = Operation(
            name=name, apply=fn, code=code, doc=doc, schema=schema, stage=stage
        )
        return fn

    return decorator


def _q(value: object) -> str:
    """Render a Python literal for the exported script."""
    return repr(value)


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------

@operation(
    "promote_header_row",
    "Drop question-text rows sitting above the real data.",
    {"drop_rows": "int — how many rows to discard from the top"},
    stage=Stage.STRUCTURE,
    # This op drops rows. That is all it does.
    #
    # It deliberately does NOT re-infer dtypes afterwards, even though the
    # columns are still text at this point. Type conversion belongs to the
    # TYPES stage, where each coercion gets its own ledger entry and its own
    # line in the exported script. Silent inference inside a STRUCTURE op is
    # a change nobody can see, which is the thing this project exists to stop.
    code=lambda drop_rows=1, **_: f"df = df.iloc[{drop_rows}:].reset_index(drop=True)",
)
def promote_header_row(df: pd.DataFrame, drop_rows: int = 1) -> tuple[pd.DataFrame, Stats]:
    out = df.iloc[drop_rows:].reset_index(drop=True)
    return out, {"rows_affected": drop_rows}


@operation(
    "drop_duplicates", "Remove exactly duplicated rows.", {},
    stage=Stage.DEDUPLICATE,
    code=lambda **_: "df = df.drop_duplicates().reset_index(drop=True)",
)
def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, Stats]:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, {"rows_affected": before - len(out)}


@operation(
    "drop_repeated_header_rows",
    "Remove rows that are a copy of the header line.",
    {},
    stage=Stage.STRUCTURE,
    # The renderer and the runner must agree exactly, so the comparison logic
    # is written once here and once identically below. tests/test_drift.py
    # executes both and diffs the frames.
    code=lambda **_: (
        "_names = [str(c).strip().lower() for c in df.columns]\n"
        "_looks_like_header = df.apply(\n"
        "    lambda r: sum(str(v).strip().lower() == n\n"
        "                  for v, n in zip(r, _names)) >= max(2, len(df.columns) * 0.5),\n"
        "    axis=1,\n"
        ")\n"
        "df = df[~_looks_like_header].reset_index(drop=True)"
    ),
)
def drop_repeated_header_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, Stats]:
    names = [str(c).strip().lower() for c in df.columns]
    looks_like_header = df.apply(
        lambda r: sum(
            str(v).strip().lower() == n for v, n in zip(r, names, strict=False)
        ) >= max(2, len(df.columns) * 0.5),
        axis=1,
    )
    out = df[~looks_like_header].reset_index(drop=True)
    return out, {"rows_affected": int(looks_like_header.sum())}


@operation(
    "drop_empty_rows", "Remove rows where every field is blank.", {},
    stage=Stage.STRUCTURE,
    code=lambda **_: "df = df.dropna(how='all').reset_index(drop=True)",
)
def drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, Stats]:
    before = len(df)
    out = df.dropna(how="all").reset_index(drop=True)
    return out, {"rows_affected": before - len(out)}


@operation(
    "drop_column",
    "Delete a column entirely.",
    {"column": "str — column to drop"},
    stage=Stage.STRUCTURE,
    code=lambda column, **_: f"df = df.drop(columns=[{_q(column)}])",
)
def drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, Stats]:
    out = df.drop(columns=[column])
    return out, {"rows_affected": len(df)}


@operation(
    "rename_column",
    "Rename a column.",
    {"column": "str — current name", "new_name": "str — new name"},
    stage=Stage.STRUCTURE,
    code=lambda column, new_name, **_: f"df = df.rename(columns={{{_q(column)}: {_q(new_name)}}})",
)
def rename_column(df: pd.DataFrame, column: str, new_name: str) -> tuple[pd.DataFrame, Stats]:
    return df.rename(columns={column: new_name}), {"rows_affected": 0}


# --------------------------------------------------------------------------
# Survey-specific
# --------------------------------------------------------------------------

@operation(
    "recode_sentinel_missing",
    "Turn missing-data codes (999, 'Prefer not to say') into real NaN.",
    {"column": "str — column to fix", "values": "list — the codes meaning 'missing'"},
    stage=Stage.SENTINEL,
    # Mirrors the runner exactly, dtype restoration included. Renderer and
    # runner must agree or the exported script stops reproducing the session.
    code=lambda column, values, **_: (
        f"_recoded = df[{_q(column)}].replace({_q(list(values))}, pd.NA)\n"
        f"_numeric = pd.to_numeric(_recoded, errors='coerce')\n"
        f"# Putting pd.NA into an int column turns the whole column to object,\n"
        f"# so `age` silently stops being a number. Restore it when nothing is lost.\n"
        f"df[{_q(column)}] = (\n"
        f"    _numeric if _numeric.notna().sum() == _recoded.notna().sum() else _recoded\n"
        f")"
    ),
)
def recode_sentinel_missing(df: pd.DataFrame, column: str, values: list) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    affected = int(out[column].isin(values).sum())

    recoded = out[column].replace(list(values), pd.NA)

    # Assigning pd.NA into an int64 column promotes it to object. That quietly
    # un-numbers the column: is_numeric_dtype goes False, so every chart calls
    # it free text, DuckDB reads it as VARCHAR, and sorting goes lexicographic.
    # If everything left is still a number, put the numeric dtype back.
    numeric = pd.to_numeric(recoded, errors="coerce")
    lossless = numeric.notna().sum() == recoded.notna().sum()
    out[column] = numeric if lossless else recoded

    return out, {"rows_affected": affected, "kept_numeric": bool(lossless)}


@operation(
    "order_likert",
    "Restore ordinality to a Likert column so median and sort behave.",
    {"column": "str — the Likert column", "order": "list — levels, lowest to highest"},
    # This MUST mirror the runner below, normalisation included. If the
    # renderer takes a shortcut the runner doesn't, the exported script stops
    # reproducing the session and the whole ledger becomes a lie.
    stage=Stage.TYPES,
    code=lambda column, order, **_: (
        f"_order = {_q(list(order))}\n"
        f"_lookup = {{str(level).strip().lower(): level for level in _order}}\n"
        f"df[{_q(column)}] = pd.Categorical(\n"
        f"    df[{_q(column)}].map(\n"
        f"        lambda v: _lookup.get(str(v).strip().lower()) if pd.notna(v) else None\n"
        f"    ),\n"
        f"    categories=_order,\n"
        f"    ordered=True,\n"
        f")"
    ),
)
def order_likert(df: pd.DataFrame, column: str, order: list) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    lookup = {str(level).strip().lower(): level for level in order}
    normalised = out[column].map(
        lambda v: lookup.get(str(v).strip().lower()) if pd.notna(v) else None
    )
    out[column] = pd.Categorical(normalised, categories=list(order), ordered=True)
    return out, {"rows_affected": int(out[column].notna().sum())}


@operation(
    "collapse_multiselect",
    "Fold exploded multi-select columns back into a count plus a joined list.",
    {"stem": "str — the question stem, e.g. Q7", "members": "list — the part columns"},
    stage=Stage.STRUCTURE,
    code=lambda stem, members, **_: (
    f"_members = {_q(list(members))}\n"
    f"df[{_q(stem + '_count')}] = df[_members].notna().sum(axis=1)\n"
    f"df[{_q(stem + '_selected')}] = df[_members].apply(\n"
    f"    lambda r: ', '.join(str(v) for v in r.dropna()), axis=1\n"
    f")"
),
)
def collapse_multiselect(df: pd.DataFrame, stem: str, members: list) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    present = [m for m in members if m in out.columns]
    out[f"{stem}_count"] = out[present].notna().sum(axis=1)
    out[f"{stem}_selected"] = out[present].apply(
        lambda r: ", ".join(str(v) for v in r.dropna()), axis=1
    )
    return out, {"rows_affected": len(out), "columns_folded": len(present)}


@operation(
    "deidentify",
    "Hash or drop an identifying column. Hashing keeps linkage, drops identity.",
    {"column": "str — the identifying column", "method": "str — 'hash' or 'drop'"},
    stage=Stage.PRIVACY,
    code=lambda column, method="hash", **_: (
    f"df[{_q(column)}] = df[{_q(column)}].map(\n"
    f"    lambda v: hashlib.sha256(str(v).encode()).hexdigest()[:12] if pd.notna(v) else v\n"
    f")" if method == "hash" else f"df = df.drop(columns=[{_q(column)}])"
),
)
def deidentify(df: pd.DataFrame, column: str, method: str = "hash") -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    if method == "drop":
        return out.drop(columns=[column]), {"rows_affected": len(out)}
    affected = int(out[column].notna().sum())
    out[column] = out[column].map(
        lambda v: hashlib.sha256(str(v).encode()).hexdigest()[:12] if pd.notna(v) else v
    )
    return out, {"rows_affected": affected}


# --------------------------------------------------------------------------
# Types and text
# --------------------------------------------------------------------------

@operation(
    "coerce_numeric",
    "Convert a column to numbers; unparseable values become NaN.",
    {"column": "str — column to convert"},
    stage=Stage.TYPES,
    code=lambda column, **_: f"df[{_q(column)}] = pd.to_numeric(df[{_q(column)}], errors='coerce')",
)
def coerce_numeric(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    before = out[column].notna().sum()
    out[column] = pd.to_numeric(out[column], errors="coerce")
    lost = int(before - out[column].notna().sum())
    return out, {"rows_affected": int(out[column].notna().sum()), "coerced_to_nan": lost}


@operation(
    "coerce_datetime",
    "Parse a column as dates; unparseable values become NaT.",
    {"column": "str — column to convert", "dayfirst": "bool — read 03/04 as 3 April, not 4 March"},
    stage=Stage.TYPES,
    # Must mirror the runner exactly, format= and all. Drop `format='mixed'`
    # here and the exported script silently NaNs every date whose layout isn't
    # the first one pandas guesses — which is most of them in survey exports.
    code=lambda column, dayfirst=False, **_: (
        f"df[{_q(column)}] = pd.to_datetime(\n"
        f"    df[{_q(column)}], errors='coerce', format='mixed', dayfirst={dayfirst}\n"
        f")"
    ),
)
def coerce_datetime(df: pd.DataFrame, column: str, dayfirst: bool = False) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    before = out[column].notna().sum()
    out[column] = pd.to_datetime(
        out[column], errors="coerce", format="mixed", dayfirst=dayfirst
    )
    parsed = int(out[column].notna().sum())
    return out, {"rows_affected": parsed, "coerced_to_nat": int(before - parsed)}


@operation(
    "strip_whitespace",
    "Trim leading and trailing spaces from a text column.",
    {"column": "str — column to trim"},
    stage=Stage.STANDARDISE,
    code=lambda column, **_: f"df[{_q(column)}] = df[{_q(column)}].str.strip()",
)
def strip_whitespace(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    original = out[column].copy()
    out[column] = out[column].astype("string").str.strip()
    affected = int((original.astype("string") != out[column]).sum())
    return out, {"rows_affected": affected}


@operation(
    "map_categories",
    "Merge spelling variants into one label ('Male'/'male'/'M' -> 'Male').",
    {"column": "str — column to normalise", "mapping": "dict — {old: new}"},
    stage=Stage.STANDARDISE,
    code=lambda column, mapping, **_: f"df[{_q(column)}] = df[{_q(column)}].replace({_q(dict(mapping))})",
)
def map_categories(df: pd.DataFrame, column: str, mapping: dict) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    affected = int(out[column].isin(list(mapping)).sum())
    out[column] = out[column].replace(dict(mapping))
    return out, {"rows_affected": affected}


# --------------------------------------------------------------------------
# Imputation — deliberately last, deliberately narrow
# --------------------------------------------------------------------------

@operation(
    "fill_missing",
    "Impute blanks. Never use on skip-logic columns — it invents respondents.",
    {
        "column": "str — column to fill",
        "method": "str — 'median', 'mean', 'mode' or 'constant'",
        "value": "any — the value, when method is 'constant'",
    },
    stage=Stage.MISSING,
    code=lambda column, method, value=None, **_: (
    f"df[{_q(column)}] = df[{_q(column)}].fillna({_q(value)})" if method == "constant"
    else f"df[{_q(column)}] = df[{_q(column)}].fillna(df[{_q(column)}].{method}())"
    if method in {"median", "mean"}
    else f"df[{_q(column)}] = df[{_q(column)}].fillna(df[{_q(column)}].mode()[0])"
),
)
def fill_missing(df: pd.DataFrame, column: str, method: str, value=None) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    affected = int(out[column].isna().sum())

    if method == "constant":
        filler = value
    elif method == "median":
        filler = out[column].median()
    elif method == "mean":
        filler = out[column].mean()
    elif method == "mode":
        modes = out[column].mode()
        filler = modes[0] if len(modes) else None
    else:
        raise ValueError(f"Unknown fill method: {method}")

    out[column] = out[column].fillna(filler)
    return out, {"rows_affected": affected, "filled_with": filler}


@operation(
    "clip_outliers",
    "Cap values into a plausible range (age 0-120, etc).",
    {"column": "str — column to clip", "lower": "float — minimum", "upper": "float — maximum"},
    stage=Stage.OUTLIERS,
    code=lambda column, lower, upper, **_: f"df[{_q(column)}] = df[{_q(column)}].clip({lower}, {upper})",
)
def clip_outliers(df: pd.DataFrame, column: str, lower: float, upper: float) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    affected = int(((out[column] < lower) | (out[column] > upper)).sum())
    out[column] = out[column].clip(lower, upper)
    return out, {"rows_affected": affected}

@operation(
    "drop_constant_columns",
    "Remove columns holding the same value in every row.",
    {"columns": "list — the constant columns to drop"},
    stage=Stage.STRUCTURE,
    code=lambda columns, **_: f"df = df.drop(columns={_q(list(columns))})",
)
def drop_constant_columns(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, Stats]:
    present = [c for c in columns if c in df.columns]
    return df.drop(columns=present), {"rows_affected": len(df), "columns_dropped": len(present)}


# --------------------------------------------------------------------------
# Validation — flags, never deletes
#
# A validation step that silently drops rows is a validation step that hides
# your data-collection problems. These add a boolean column so the bad rows
# stay visible and countable, and you decide what to do about them.
# --------------------------------------------------------------------------

@operation(
    "flag_out_of_range",
    "Mark values outside a plausible range without changing them.",
    {"column": "str — column to check", "lower": "float — minimum", "upper": "float — maximum"},
    stage=Stage.VALIDATE,
    code=lambda column, lower, upper, **_: (
        f"df[{_q(str(column) + '_out_of_range')}] = (\n"
        f"    df[{_q(column)}].notna()\n"
        f"    & ((df[{_q(column)}] < {lower}) | (df[{_q(column)}] > {upper}))\n"
        f")"
    ),
)
def flag_out_of_range(df: pd.DataFrame, column: str, lower: float, upper: float) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    flag = out[column].notna() & ((out[column] < lower) | (out[column] > upper))
    out[f"{column}_out_of_range"] = flag
    return out, {"rows_affected": int(flag.sum())}


@operation(
    "flag_date_order",
    "Mark rows where an end date falls before its start date.",
    {"start_column": "str — the earlier date", "end_column": "str — the later date"},
    stage=Stage.VALIDATE,
    code=lambda start_column, end_column, **_: (
        f"df[{_q(str(end_column) + '_before_' + str(start_column))}] = (\n"
        f"    df[{_q(start_column)}].notna()\n"
        f"    & df[{_q(end_column)}].notna()\n"
        f"    & (df[{_q(end_column)}] < df[{_q(start_column)}])\n"
        f")"
    ),
)
def flag_date_order(df: pd.DataFrame, start_column: str, end_column: str) -> tuple[pd.DataFrame, Stats]:
    out = df.copy()
    flag = (
        out[start_column].notna()
        & out[end_column].notna()
        & (out[end_column] < out[start_column])
    )
    out[f"{end_column}_before_{start_column}"] = flag
    return out, {"rows_affected": int(flag.sum())}


# --------------------------------------------------------------------------

def llm_tool_spec() -> str:
    """Render the registry as a prompt fragment, grouped by stage.

    Grouping matters: a flat list invites the model to propose fill_missing
    before recode_sentinel_missing. Showing the sequence teaches the order for
    free, and the planner enforces it afterwards regardless.
    """
    lines = []
    for stage in Stage:
        in_stage = [o for o in REGISTRY.values() if o.stage is stage]
        if not in_stage:
            continue
        lines.append(f"\nSTAGE {stage.value} — {stage.label}  ({stage.why})")
        for op in sorted(in_stage, key=lambda o: o.name):
            params = ", ".join(f"{k} ({v})" for k, v in op.schema.items()) or "no parameters"
            lines.append(f"  - {op.name}: {op.doc}\n      params: {params}")
    return "\n".join(lines)


def stage_of(op_name: str) -> Stage:
    """Which stage an op belongs to. KeyError for unknown ops, deliberately."""
    return REGISTRY[op_name].stage

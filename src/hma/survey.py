"""Survey-specific intelligence.

This is what makes HayMedics Data Analyser a *research* tool instead of a generic cleaner.
Everything here is deterministic — no LLM, no tokens, no guessing.

The rule: this module SUGGESTS, it never APPLIES. A sentinel value of 99
might be a real answer ("99 years old"). So detection produces candidates
with a confidence score, and a human or the cleaning agent decides.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

# Numeric codes conventionally used for "no answer" in survey instruments.
# 77/88/99 are common in health surveys (DHS, BRFSS) but also plausible real
# values, so they carry lower confidence.
SENTINEL_NUMERIC: dict[float, float] = {
    # int64 minimum. Not a survey convention — it's what a null turns into when
    # a pipeline forces a column to integer. Visible in the healthcare export
    # as an "age" of -9223372036854775808.
    -9223372036854775808: 0.99,
    2147483647: 0.90, -2147483648: 0.95,   # int32 limits, same story
    -999: 0.95, -99: 0.95, -9: 0.75, -1: 0.55,
    999: 0.90, 9999: 0.90, 99999: 0.90,
    999999: 0.90, 98: 0.50, 99: 0.60, 88: 0.55, 77: 0.50,
}

SENTINEL_TEXT: dict[str, float] = {
    "": 0.99, "-": 0.90, "--": 0.90, ".": 0.85, "n/a": 0.95, "na": 0.85,
    "?": 0.95, "??": 0.95, "???": 0.95, "unspecified": 0.90, "not stated": 0.95,
    "not recorded": 0.95, "blank": 0.90, "tbd": 0.85, "pending": 0.60,
    "n.a.": 0.95, "none": 0.70, "null": 0.95, "nil": 0.85, "nan": 0.99,
    "missing": 0.95, "unknown": 0.80, "not applicable": 0.95,
    "not answered": 0.95, "no answer": 0.95, "no response": 0.95,
    "don't know": 0.90, "dont know": 0.90, "do not know": 0.90, "dk": 0.75,
    "prefer not to say": 0.95, "prefer not to answer": 0.95,
    "refused": 0.95, "declined": 0.85, "no opinion": 0.75,
    "not sure": 0.70, "undecided": 0.60, "other": 0.20,
}

# Ordered Likert scales, lowest -> highest. Used to restore ordinality that
# CSV round-tripping destroys (pandas reads them as unordered strings, which
# silently breaks every median, sort, and ordinal regression downstream).
LIKERT_SCALES: dict[str, list[str]] = {
    "agreement_5": ["strongly disagree", "disagree", "neutral", "agree",
                    "strongly agree"],
    "agreement_7": ["strongly disagree", "disagree", "somewhat disagree",
                    "neutral", "somewhat agree", "agree", "strongly agree"],
    "frequency_5": ["never", "rarely", "sometimes", "often", "always"],
    "frequency_6": ["never", "rarely", "monthly", "weekly", "daily",
                    "multiple times a day"],
    "satisfaction_5": ["very dissatisfied", "dissatisfied", "neutral",
                       "satisfied", "very satisfied"],
    "quality_5": ["very poor", "poor", "fair", "good", "excellent"],
    "importance_5": ["not at all important", "slightly important",
                     "moderately important", "very important",
                     "extremely important"],
    "likelihood_5": ["very unlikely", "unlikely", "neutral", "likely",
                     "very likely"],
    "extent_5": ["not at all", "a little", "somewhat", "a lot",
                 "a great deal"],
    "yesno": ["no", "yes"],
}

# Column-name fragments that suggest direct or indirect identifiers.
#
# These are matched against _name_tokens() output, which has already turned
# underscores and camelCase into spaces. So the separators here must be
# `[ _]?`, not `_?` — written as `national_?id` this never matches anything,
# because by the time the pattern runs the column is the string "national id".
#
# Order matters: dict order is match order and the first hit wins. record_id
# comes before name so `respondent_id` is reported as an ID rather than as
# "suggests name", which is both untrue and the wrong disclosure risk — an ID
# links records across files, a name identifies a person on sight.
PII_PATTERNS: dict[str, str] = {
    "record_id": r"\b(patient[ _]?id|hospital[ _]?no|mrn|folder|case[ _]?no|"
                 r"respondent[ _]?id|participant[ _]?id|subject[ _]?id|serial|uuid)\b",
    "id_number": r"\b(nin|ssn|nhs|passport|licen[cs]e|national[ _]?id)\b",
    "name": r"\b(name|surname|firstname|lastname|fullname|respondent)\b",
    "contact": r"\b(email|e[ _-]?mail|phone|mobile|tel|telephone|whatsapp)\b",
    "address": r"\b(address|street|postcode|zip|gps|lat|lon|longitude|latitude)\b",
    "free_text": r"\b(comment|feedback|remark|explain|describe|other[ _]?text)\b",
}

# Multi-select questions get exploded into one binary column per option.
# These are the naming conventions the big survey platforms emit.
MULTISELECT_PATTERNS: list[str] = [
    r"^(?P<stem>Q\d+)_Part_(?P<part>\d+)$",        # Kaggle survey
    r"^(?P<stem>Q\d+[a-z]?)_(?P<part>\d+)$",       # generic numbered
    r"^(?P<stem>.+?)/(?P<part>.+)$",               # KoBoToolbox / ODK
    r"^(?P<stem>.+?)___(?P<part>\d+)$",            # REDCap checkbox
]


@dataclass
class Finding:
    """One thing we noticed. Not a change — a suggestion."""

    kind: str
    column: str
    confidence: float
    detail: str
    suggested_op: str | None = None
    suggested_params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.confidence:.0%}] {self.kind} on {self.column}: {self.detail}"


def _norm(value: object) -> str:
    return str(value).strip().lower()


def is_texty(series: pd.Series) -> bool:
    """True for text columns.

    pandas 3.0 gives string columns the `str` dtype, not `object`. Checking
    `dtype == object` silently returns False on modern pandas, which quietly
    disables every text detector. Check the semantic type instead.
    """
    return (
        pd.api.types.is_string_dtype(series)
        or series.dtype == object
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _name_tokens(column: object) -> str:
    """Split a column name so word boundaries actually work.

    `_` is a word character, so `\bname\b` never matches `respondent_name`.
    Underscores and camelCase both become spaces first.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column))
    return re.sub(r"[_\-./]+", " ", text).strip().lower()


def detect_two_row_header(df: pd.DataFrame) -> Finding | None:
    """Catch the classic survey export where row 0 is question text.

    Kaggle's ML & DS Survey does exactly this: the header is the variable
    code (Q1, Q2...) and the first data row is the full question wording.
    Read it naively and every column becomes `object` dtype, so every
    numeric column silently turns into strings.
    """
    if df.empty:
        return None

    first = df.iloc[0]
    long_cells = sum(1 for v in first if isinstance(v, str) and len(v) > 40)
    ratio = long_cells / max(len(first), 1)

    if ratio < 0.3:
        return None

    # Corroborate: would dtypes improve if we dropped row 0?
    rescued = 0
    for col in df.columns:
        rest = df[col].iloc[1:]
        if df[col].dtype == object and pd.to_numeric(rest, errors="coerce").notna().mean() > 0.8:
            rescued += 1

    confidence = min(0.60 + ratio * 0.3 + (rescued / max(len(df.columns), 1)) * 0.2, 0.97)
    return Finding(
        kind="two_row_header",
        column="<table>",
        confidence=confidence,
        detail=(
            f"Row 0 looks like question text, not data "
            f"({long_cells}/{len(first)} cells are long prose; "
            f"{rescued} columns would become numeric if dropped)."
        ),
        suggested_op="promote_header_row",
        suggested_params={"drop_rows": 1},
    )


def detect_sentinel_missing(series: pd.Series, threshold: float = 0.5) -> Finding | None:
    """Find values that mean 'missing' but aren't NaN.

    Handles the string form too. A column read as ["25", "999", "Prefer not
    to say"] is text, so the numeric 999 arrives as the string "999" and a
    plain numeric lookup misses it entirely. That column is exactly the one
    that most needs catching, because it's about to be coerced to numeric and
    have its refusals silently deleted.
    """
    if series.empty:
        return None

    hits: dict[object, float] = {}
    counts = series.value_counts(dropna=True)

    for value, count in counts.items():
        conf = None

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            conf = SENTINEL_NUMERIC.get(float(value))
        elif isinstance(value, str):
            text = _norm(value)
            conf = SENTINEL_TEXT.get(text)
            if conf is None:
                # "999" living in a text column — same code, wearing a coat.
                try:
                    conf = SENTINEL_NUMERIC.get(float(text))
                except ValueError:
                    conf = None

        # A sentinel is an outlier, not the bulk of a column.
        if conf and count / len(series) > 0.5:
            conf *= 0.4

        if conf and conf >= threshold:
            hits[value] = conf

    if not hits:
        return None

    affected = int(sum(counts[v] for v in hits))
    return Finding(
        kind="sentinel_missing",
        column=str(series.name),
        confidence=max(hits.values()),
        detail=(
            f"{affected} values ({affected / len(series):.1%}) look like "
            f"missing-data codes: {sorted(hits, key=str)[:5]}"
        ),
        suggested_op="recode_sentinel_missing",
        suggested_params={"column": str(series.name), "values": list(hits)},
    )


def detect_likert(series: pd.Series) -> Finding | None:
    """Spot an ordinal scale that pandas has flattened into unordered strings."""
    if not is_texty(series):
        return None

    values = {_norm(v) for v in series.dropna().unique()}
    if not 2 <= len(values) <= 8:
        return None

    best_name, best_overlap = None, 0.0
    for name, scale in LIKERT_SCALES.items():
        overlap = len(values & set(scale)) / len(values)
        if overlap > best_overlap:
            best_name, best_overlap = name, overlap

    if best_overlap < 0.75 or best_name is None:
        return None

    scale = LIKERT_SCALES[best_name]
    present = [level for level in scale if level in values]
    return Finding(
        kind="likert_scale",
        column=str(series.name),
        confidence=round(min(best_overlap, 0.95), 2),
        detail=(
            f"Matches the '{best_name}' scale. Currently unordered — "
            f"median and sort will be wrong until it's made ordinal."
        ),
        suggested_op="order_likert",
        suggested_params={"column": str(series.name), "order": present},
    )


@dataclass(frozen=True)
class Vocabulary:
    """A set of ways people write the same small number of answers.

    `hint` is a regex on the column NAME, and it exists to break a genuine
    ambiguity: "1" means Yes in `Smoker?` and Male in `Gender`. No amount of
    looking at the values alone resolves that — you have to read the header.
    """

    name: str
    canonical: dict[str, frozenset[str]]
    hint: str = ""


VOCABULARIES: tuple[Vocabulary, ...] = (
    Vocabulary(
        name="yes_no",
        canonical={
            "Yes": frozenset({"yes", "y", "true", "t", "1", "1.0", "positive",
                              "present", "affirmative"}),
            "No": frozenset({"no", "n", "false", "f", "0", "0.0", "negative",
                             "absent", "nil"}),
        },
        hint=r"\b(smoker|smoking|readmit\w*|died|death|consent|eligible|pregnan\w*|"
             r"diabet\w*|hypertens\w*|insured|referred|complicat\w*|"
             r"is|has|was|had|any|ever)\b",
    ),
    Vocabulary(
        name="sex",
        canonical={
            "Male": frozenset({"male", "m", "man", "boy", "1", "1.0"}),
            "Female": frozenset({"female", "f", "woman", "girl", "2", "2.0"}),
        },
        hint=r"\b(sex|gender)\b",
    ),
)


def detect_value_synonyms(series: pd.Series) -> Finding | None:
    """Collapse different spellings of the same answer into one label.

    `detect_case_variants` handles 'Lagos' vs 'lagos' — values that are
    already identical once you lower them. This handles the harder half:
    'Male', 'm', 'Man' and '1' are four different strings that mean one thing,
    and no amount of normalising whitespace will merge them.

    Left alone, a binary column arrives as ten categories, every cross-tab
    splits into ten thin columns, and a chi-square on it is meaningless.

    Values the vocabulary doesn't recognise are deliberately untouched. '?'
    and 'U' are somebody else's job — guessing at them here would be the
    tool inventing answers, which is the one thing it must never do.
    """
    if not is_texty(series):
        return None

    originals = [v for v in series.dropna().unique()]
    values = {_norm(v) for v in originals}

    # Two values are already consistent; there's nothing to collapse. Above a
    # dozen and this isn't a small categorical, it's free text.
    if not 3 <= len(values) <= 14:
        return None

    column_name = _name_tokens(series.name)
    best: tuple[float, Vocabulary, dict[str, str]] | None = None

    for vocab in VOCABULARIES:
        covered: dict[str, str] = {}
        for canonical, aliases in vocab.canonical.items():
            for value in values:
                if value in aliases:
                    covered[value] = canonical

        if not covered:
            continue
        coverage = len(covered) / len(values)
        if coverage < 0.6 or len(set(covered.values())) != 2:
            continue

        score = coverage + (0.3 if vocab.hint and re.search(vocab.hint, column_name) else 0.0)
        if best is None or score > best[0]:
            best = (score, vocab, covered)

    if best is None:
        return None

    _, vocab, covered = best

    # Map from the ORIGINAL values, not the normalised ones — replace() will
    # be matching against what's actually in the column. Identity mappings are
    # dropped so the ledger reports the real number of rows changed.
    mapping = {
        original: covered[_norm(original)]
        for original in originals
        if _norm(original) in covered and str(original) != covered[_norm(original)]
    }
    if not mapping:
        return None

    affected = int(series.isin(list(mapping)).sum())
    canonicals = sorted(set(covered.values()))
    unmatched = sorted(values - set(covered))

    detail = (
        f"{len(values)} distinct values collapse to {len(canonicals)} "
        f"({' / '.join(canonicals)}) — this is a {vocab.name.replace('_', '/')} "
        f"variable written {len(values)} different ways ({affected:,} rows affected). "
        f"Until they're merged every breakdown splits into {len(values)} thin "
        f"columns instead of {len(canonicals)}."
    )
    if unmatched:
        detail += (
            f" Left alone: {unmatched[:4]} — not recognised, so not guessed at."
        )

    return Finding(
        kind="value_synonyms",
        column=str(series.name),
        confidence=0.93 if len(unmatched) <= 2 else 0.85,
        detail=detail,
        suggested_op="map_categories",
        suggested_params={"column": str(series.name), "mapping": mapping},
    )


def detect_repeated_header_rows(df: pd.DataFrame) -> Finding | None:
    """Find rows that are actually a copy of the header.

    Your dashboard showed a `Gender` category with exactly one member, spelled
    "Gender", and a `Smoker?` category spelled "Smoker?". That's the header
    line pasted back in as a record — it happens whenever two exports get
    concatenated, or someone appends a sheet without dropping the top row.

    One row barely moves a count, which is why it survives review. But it's a
    fake respondent, it puts a junk level in every categorical, and it drags
    every column back to text.
    """
    if df.empty:
        return None

    names = [_norm(c) for c in df.columns]
    matches = df.apply(
        lambda row: sum(_norm(v) == n for v, n in zip(row, names, strict=False)),
        axis=1,
    )
    hits = matches[matches >= max(2, len(df.columns) * 0.5)]
    if hits.empty:
        return None

    return Finding(
        kind="repeated_header_row",
        column="<table>",
        confidence=0.95,
        detail=(
            f"{len(hits)} row(s) repeat the column headers as data — probably "
            f"two files concatenated without dropping the second header. They're "
            f"fake respondents: they add a junk level to every categorical and "
            f"keep numeric columns stored as text."
        ),
        suggested_op="drop_repeated_header_rows",
        suggested_params={},
    )


def detect_case_variants(series: pd.Series) -> Finding | None:
    """Catch 'Lagos' / 'lagos' / 'LAGOS' / ' Lagos ' living as four categories.

    Free-typed and multi-device survey collection produces this constantly, and
    it inflates category counts silently — your frequency table shows four
    Lagoses and nobody notices until review.
    """
    if not is_texty(series):
        return None

    values = series.dropna().unique()
    if not 2 <= len(values) <= 60:
        return None

    buckets: dict[str, list] = {}
    for value in values:
        buckets.setdefault(_norm(value), []).append(value)

    collapsible = {k: v for k, v in buckets.items() if len(v) > 1}
    if not collapsible:
        return None

    mapping: dict[object, object] = {}
    for variants in collapsible.values():
        # Keep the most common spelling as canonical.
        counts = series.value_counts()
        canonical = max(variants, key=lambda v: counts.get(v, 0))
        for variant in variants:
            if variant != canonical:
                mapping[variant] = canonical

    affected = int(series.isin(list(mapping)).sum())
    return Finding(
        kind="case_variants",
        column=str(series.name),
        confidence=0.92,
        detail=(
            f"{len(values)} categories collapse to {len(buckets)} once case and "
            f"whitespace are normalised ({affected} rows affected). "
            f"e.g. {sorted(map(str, next(iter(collapsible.values()))))[:3]}"
        ),
        suggested_op="map_categories",
        suggested_params={"column": str(series.name), "mapping": mapping},
    )


def detect_duplicate_rows(df: pd.DataFrame) -> Finding | None:
    """Flag exact duplicate submissions.

    Not survey-specific, but it belongs in the scan: the quality score docks
    points for duplicates, so if nothing here proposes removing them, "apply
    everything" leaves the score stuck below where it should be. A detector
    that scores a problem without offering the fix is just nagging.
    """
    try:
        n = int(df.duplicated().sum())
    except TypeError:
        return None  # unhashable cells (lists/dicts from JSON exports)

    if n == 0:
        return None

    return Finding(
        kind="duplicate_rows",
        column="<table>",
        confidence=0.95,
        detail=(
            f"{n} exactly duplicated rows ({n / len(df):.1%}). Usually a double "
            f"submission or a merge run twice. Check it isn't legitimate repeat "
            f"measurement before removing."
        ),
        suggested_op="drop_duplicates",
        suggested_params={},
    )


# Plausible ranges for column names that mean the same thing everywhere.
# Deliberately generous: the job is catching 999 and -1, not second-guessing
# a real respondent. Anything flagged is shown, never auto-corrected.
PLAUSIBLE_RANGES: dict[str, tuple[str, float, float]] = {
    r"\b(age|age in years|respondent age)\b": ("age in years", 0, 120),
    r"\b(percent|percentage|pct|rate)\b": ("a percentage", 0, 100),
    r"\b(year|birth year|yob)\b": ("a calendar year", 1900, 2100),
    r"\b(weight|weight kg)\b": ("weight in kg", 1, 400),
    r"\b(height|height cm)\b": ("height in cm", 30, 260),
    r"\b(bmi)\b": ("a BMI", 8, 100),
    r"\b(systolic|sbp)\b": ("systolic BP", 50, 300),
    r"\b(diastolic|dbp)\b": ("diastolic BP", 20, 200),
}


def detect_constant_columns(df: pd.DataFrame) -> Finding | None:
    """Flag columns holding one value in every row.

    The quality score docks points for these, so something has to offer the
    fix — a score that penalises a problem it won't help you solve is just
    nagging. They carry zero information for analysis, though `consent = Yes`
    across the board is worth a glance before you drop it: it may mean the
    non-consenting rows were already filtered out upstream.

    Note `dropna=False`, and note it hard. With `dropna=True`, a multi-select
    member like Q7_Part_1 — holding "Radio" or blank — reports ONE unique
    value and gets called constant, because the blanks are hidden from the
    count. It isn't constant: ticked and not-ticked are two real states, and
    dropping the column deletes half the answer. `dropna=False` counts the
    blank as the state it actually is.
    """
    constant = [
        str(c) for c in df.columns
        if df[c].nunique(dropna=False) <= 1 and len(df) > 1
    ]
    if not constant:
        return None

    return Finding(
        kind="constant_columns",
        column=", ".join(constant[:4]) + ("..." if len(constant) > 4 else ""),
        confidence=0.90,
        detail=(
            f"{len(constant)} column(s) hold the same value in every row: "
            f"{constant[:6]}. They carry no information. Check they aren't the "
            f"residue of an upstream filter before dropping."
        ),
        suggested_op="drop_constant_columns",
        suggested_params={"columns": constant},
    )


def detect_implausible_range(series: pd.Series) -> Finding | None:
    """Flag values a column's own name says are impossible.

    Only fires on names that mean the same thing in every dataset — `age`,
    `bmi`, `systolic`. Guessing bounds from the data itself would flag real
    outliers as errors, which is a different and much worse mistake.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None

    name = _name_tokens(series.name)
    for pattern, (label, lower, upper) in PLAUSIBLE_RANGES.items():
        if not re.search(pattern, name):
            continue

        values = series.dropna()
        if values.empty:
            return None

        bad = ((values < lower) | (values > upper)).sum()
        if not bad:
            return None

        return Finding(
            kind="implausible_range",
            column=str(series.name),
            confidence=0.75,
            detail=(
                f"{bad} value(s) fall outside {lower:g}–{upper:g}, the plausible "
                f"range for {label}. Often a leftover missing-data code. Flagged, "
                f"not changed — check them before deciding."
            ),
            suggested_op="flag_out_of_range",
            suggested_params={"column": str(series.name), "lower": lower, "upper": upper},
        )
    return None


# A date needs a date's shape: two separators between digits, or a month in
# words. One separator is not enough — the float 75.1 has one, and treating
# that as date-shaped turns a weight column into 1970 timestamps.
DATE_SEPARATORS = re.compile(r"[/\-.]")
MONTH_NAME = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE
)


def _looks_like_a_date(value: object) -> bool:
    """Structural test, run before any parse attempt.

    `pd.to_datetime` is far too permissive to be a detector. Handed the string
    "49" it returns 1970-01-01 00:00:00.000000049 — it reads a bare integer as
    nanoseconds since the epoch and reports success. So "does it parse?" is not
    a question that distinguishes a date from a number, and asking it is how an
    `Age` column of 49s becomes a column of 1970 timestamps.

    Requiring two separators costs us bare YYYYMMDD and 2026-03, both rare and
    both coercible by hand. It buys us never destroying a numeric column.
    """
    text = str(value).strip()
    if not text or not any(ch.isdigit() for ch in text):
        return False
    if MONTH_NAME.search(text):
        return True
    return len(DATE_SEPARATORS.findall(text)) >= 2


def detect_date_like(series: pd.Series) -> Finding | None:
    """Spot a date column that arrived as text.

    Survey exports collect dates from several devices and locales, so one
    column routinely holds `2026-03-01`, `01/03/2026` and `15-03-2026` at
    once. Left as text they sort alphabetically, which puts December before
    February and quietly ruins every trend line drawn from them.
    """
    if not is_texty(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return None

    values = series.dropna()
    if len(values) < 5:
        return None

    sample = values.sample(min(len(values), 250), random_state=0)

    # Shape first. A column of bare numbers never reaches the parser.
    shaped = sample.map(_looks_like_a_date)
    if shaped.mean() < 0.85:
        return None

    with warnings.catch_warnings():
        # Mixed formats warn per-row; we're measuring the rate, not fixing it here.
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample[shaped], errors="coerce", format="mixed")

    rate = parsed.notna().mean() * shaped.mean()
    if rate < 0.85:
        return None

    formats = {_date_shape(v) for v in sample.head(80)}
    return Finding(
        kind="date_like",
        column=str(series.name),
        confidence=round(min(rate, 0.95), 2),
        detail=(
            f"{rate:.0%} of values parse as dates but the column is text"
            + (f", in {len(formats)} different formats" if len(formats) > 1 else "")
            + ". As text it sorts alphabetically, so any trend over time is wrong."
        ),
        suggested_op="coerce_datetime",
        suggested_params={"column": str(series.name)},
    )


def _date_shape(value: object) -> str:
    """Crude format signature: digits -> 'd', everything else kept."""
    return re.sub(r"\d", "d", str(value))


def detect_numeric_like(series: pd.Series) -> Finding | None:
    """Spot a number column that arrived as text.

    One stray "N/A" or "12 years" in a column of integers makes pandas read
    the whole thing as text, and every mean, median and chart silently
    refuses to work on it afterwards.
    """
    if not is_texty(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return None

    values = series.dropna()
    if len(values) < 5:
        return None

    parsed = pd.to_numeric(values, errors="coerce")
    rate = parsed.notna().mean()

    # Below 0.80 it's a text column with some numbers in it, not a number
    # column. At 1.00 with very few distinct values it's probably a code
    # (1 = male, 2 = female) and coercing invites someone to average it.
    if rate < 0.80 or values.nunique() <= 2:
        return None

    lost = int((~parsed.notna()).sum())
    return Finding(
        kind="numeric_like",
        column=str(series.name),
        confidence=round(min(rate, 0.95), 2),
        detail=(
            f"{rate:.0%} of values are numbers but the column is text. "
            + (f"{lost} value(s) won't convert and would become blank — "
               f"recode any missing-data codes first. "
               if lost else "")
            + "Until it's numeric, mean and median won't work on it."
        ),
        suggested_op="coerce_numeric",
        suggested_params={"column": str(series.name)},
    )


def detect_multiselect_groups(columns: list[str]) -> list[Finding]:
    """Group Q7_Part_1, Q7_Part_2 ... back into the one question they were."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        for pattern in MULTISELECT_PATTERNS:
            match = re.match(pattern, col)
            if match:
                groups.setdefault(match.group("stem"), []).append(col)
                break

    findings = []
    for stem, members in sorted(groups.items()):
        if len(members) < 3:
            continue
        findings.append(
            Finding(
                kind="multiselect_group",
                column=stem,
                confidence=0.90,
                detail=(
                    f"{len(members)} columns are one multi-select question. "
                    f"Blank here means 'not ticked', not 'missing' — "
                    f"filling them would invent data."
                ),
                suggested_op="collapse_multiselect",
                suggested_params={"stem": stem, "members": sorted(members)},
            )
        )
    return findings


def detect_pii(df: pd.DataFrame) -> list[Finding]:
    """Flag columns that probably identify a respondent."""
    findings = []
    for col in df.columns:
        name = _name_tokens(col)
        for kind, pattern in PII_PATTERNS.items():
            if re.search(pattern, name):
                # Free text is a different problem and needs a different answer.
                # Hashing "The clinic was far in Ilorin" gives you a3f9b2… —
                # it destroys the data and protects nobody, because the risk in
                # prose is someone recognisable in the wording, not linkage
                # across files. There's no op for "read these and decide", so
                # this suggests nothing and says so.
                is_prose = kind == "free_text"
                findings.append(
                    Finding(
                        kind="pii",
                        column=str(col),
                        confidence=0.50 if is_prose else 0.80,
                        detail=(
                            "Free-text column. Read it before sharing — people "
                            "name themselves, their village or their clinic in "
                            "comments. Hashing it would destroy the data without "
                            "protecting anyone; this one needs your eyes."
                            if is_prose else
                            f"Column name suggests {kind.replace('_', ' ')}. "
                            f"De-identify before sharing or publishing."
                        ),
                        suggested_op=None if is_prose else "deidentify",
                        suggested_params=(
                            {} if is_prose else {"column": str(col), "method": "hash"}
                        ),
                    )
                )
                break
    return findings


def detect_structural_missing(df: pd.DataFrame, threshold: float = 0.45) -> list[Finding]:
    """Distinguish skip-logic gaps from genuine non-response.

    If a column is largely blank, the respondents were probably never asked —
    the question was gated behind an earlier answer. Imputing it is the single
    most common way survey analyses go quietly wrong: you manufacture opinions
    for people who were never in the room.

    A hard threshold would be dishonest here, because 60% blank is genuinely
    ambiguous between skip logic and bad response rates. So confidence scales
    with emptiness and the message says which one we think it is. Either way
    the recommendation is the same — don't impute, subset instead.
    """
    findings = []
    for col in df.columns:
        pct = float(df[col].isna().mean())
        if pct < threshold:
            continue

        if pct >= 0.95:
            confidence, verdict = 0.90, "almost certainly skip logic"
        elif pct >= 0.75:
            confidence, verdict = 0.75, "very likely skip logic"
        else:
            confidence, verdict = 0.55, "possibly skip logic, possibly poor response"

        findings.append(
            Finding(
                kind="structural_missing",
                column=str(col),
                confidence=confidence,
                detail=(
                    f"{pct:.1%} blank — {verdict}. Check the questionnaire for a "
                    f"gating question. Do NOT impute; analyse only among those "
                    f"who were actually asked."
                ),
                suggested_op=None,
            )
        )
    return findings


def scan(df: pd.DataFrame) -> list[Finding]:
    """Run every survey detector. Deterministic, zero tokens."""
    findings: list[Finding] = []

    header = detect_two_row_header(df)
    if header:
        # Everything else would read question text as data — stop here.
        return [header]

    header_rows = detect_repeated_header_rows(df)
    if header_rows:
        findings.append(header_rows)

    findings.extend(detect_multiselect_groups([str(c) for c in df.columns]))
    findings.extend(detect_pii(df))

    for whole_table in (detect_duplicate_rows(df), detect_constant_columns(df)):
        if whole_table:
            findings.append(whole_table)

    # Multi-select members must be identified BEFORE the missingness scan.
    # A blank in Q7_Part_3 means "didn't tick Television" — it is not a gap.
    # Flagging those as skip logic would be the exact confusion this tool
    # exists to prevent, so they're excluded from every missingness detector.
    multiselect_members = {
        m for f in findings if f.kind == "multiselect_group"
        for m in f.suggested_params.get("members", [])
    }
    remainder = df[[c for c in df.columns if str(c) not in multiselect_members]]
    findings.extend(detect_structural_missing(remainder))

    for col in df.columns:
        if str(col) in multiselect_members:
            continue
        for detector in (detect_sentinel_missing, detect_likert,
                         detect_value_synonyms, detect_case_variants,
                         detect_implausible_range, detect_date_like,
                         detect_numeric_like):
            found = detector(df[col])
            if found:
                findings.append(found)

    return sorted(findings, key=lambda f: -f.confidence)

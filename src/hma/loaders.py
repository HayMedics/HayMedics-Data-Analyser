"""File loading.

`pd.read_excel(file)` is a trap for survey data, and it fails silently three
different ways:

  1. A workbook with "Read Me", "Responses" and "Codebook" tabs returns the
     first sheet. You get a two-row instruction tab and no error at all.
  2. Exports routinely carry a title block — "Malaria KAP Survey 2026", a
     location, a blank line — above the real header. pandas takes the title as
     the header, the real headers become row 0, and every column arrives as
     text.
  3. `.xls` needs xlrd, which is a separate package from openpyxl. Offering
     .xls in the uploader without it means a crash at the worst moment.

So this module reads the workbook, decides which sheet holds the data, finds
the real header row, and — this is the part that matters — reports what it did
in `notes`. Guessing is fine. Guessing silently isn't.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CSV_EXT = {".csv", ".tsv", ".txt"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}
JSON_EXT = {".json"}
SUPPORTED = CSV_EXT | EXCEL_EXT | JSON_EXT

# Tabs that are documentation, not data. Scored down, never hard-excluded —
# somebody's real data will be on a sheet called "Notes" one day.
BORING_SHEETS = {
    "read me", "readme", "notes", "info", "about", "instructions",
    "codebook", "dictionary", "data dictionary", "legend", "key",
    "metadata", "cover", "index", "toc", "changelog",
}

MAX_HEADER_SCAN = 12


class UnsupportedFile(ValueError):
    """Not a format we read."""


@dataclass
class SheetInfo:
    name: str
    n_rows: int
    n_cols: int
    score: float

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.n_rows:,} rows × {self.n_cols} cols)"


@dataclass
class LoadResult:
    df: pd.DataFrame
    source_name: str
    sheet: str | None = None
    sheets: list[SheetInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_choices(self) -> bool:
        return len(self.sheets) > 1


def _suffix(name: str) -> str:
    return Path(str(name)).suffix.lower()


def _rewind(source) -> None:
    """Streamlit hands over a buffer that may already be at EOF."""
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass


def _find_header_row(raw: pd.DataFrame) -> int:
    """Locate the real header among any title rows above it.

    A title block is narrow — one or two filled cells on a row. The header is
    the first row that's nearly as wide as the table itself.
    """
    if raw.empty:
        return 0

    filled = raw.notna().sum(axis=1)
    widest = filled.max()
    if widest == 0:
        return 0

    for i in range(min(MAX_HEADER_SCAN, len(raw))):
        if filled.iloc[i] >= widest * 0.6:
            return int(i)
    return 0


def _score_sheet(name: str, df: pd.DataFrame) -> float:
    """How much this sheet looks like the dataset rather than the paperwork."""
    if df.empty:
        return 0.0
    cells = df.notna().sum().sum()
    score = float(cells) * (1 + 0.1 * min(len(df.columns), 40))
    if str(name).strip().lower() in BORING_SHEETS:
        score *= 0.01
    if len(df) < 3:
        score *= 0.1
    return score


def excel_sheets(source, name: str = "") -> list[SheetInfo]:
    """Every sheet in the workbook, best-looking first."""
    _rewind(source)
    engine = "xlrd" if _suffix(name) == ".xls" else "openpyxl"
    try:
        book = pd.read_excel(source, sheet_name=None, header=None, engine=engine)
    except ImportError as exc:
        raise UnsupportedFile(
            ".xls needs the xlrd package. Install it with `uv add xlrd`, or "
            "re-save the file as .xlsx."
        ) from exc

    infos = [
        SheetInfo(
            name=str(sheet),
            n_rows=max(len(frame) - _find_header_row(frame) - 1, 0),
            n_cols=int(frame.notna().any().sum()),
            score=_score_sheet(sheet, frame),
        )
        for sheet, frame in book.items()
    ]
    infos.sort(key=lambda s: -s.score)
    return infos


def _read_excel(source, name: str, sheet: str | None) -> LoadResult:
    sheets = excel_sheets(source, name)
    if not sheets:
        raise UnsupportedFile("That workbook has no sheets.")

    chosen = sheet or sheets[0].name
    known = {s.name for s in sheets}
    if chosen not in known:
        raise UnsupportedFile(f"No sheet named {chosen!r}. Found: {sorted(known)}")

    _rewind(source)
    engine = "xlrd" if _suffix(name) == ".xls" else "openpyxl"
    raw = pd.read_excel(source, sheet_name=chosen, header=None, engine=engine)

    header_row = _find_header_row(raw)
    _rewind(source)
    df = pd.read_excel(source, sheet_name=chosen, header=header_row, engine=engine)

    # Excel gives every empty column a name; they aren't data.
    blank = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if blank:
        df = df.drop(columns=blank)
    df = df.dropna(how="all").reset_index(drop=True)

    notes = []
    if len(sheets) > 1:
        others = ", ".join(s.name for s in sheets if s.name != chosen)
        notes.append(f"Read sheet **{chosen}** of {len(sheets)}. Also in this file: {others}.")
    if header_row > 0:
        notes.append(
            f"Skipped {header_row} row(s) above the header — they looked like a "
            f"title block, not data."
        )
    if blank:
        notes.append(f"Dropped {len(blank)} empty column(s) Excel had padded on.")

    return LoadResult(df=df, source_name=str(name), sheet=chosen, sheets=sheets, notes=notes)


def _read_csv(source, name: str) -> LoadResult:
    _rewind(source)
    sep = "\t" if _suffix(name) == ".tsv" else None  # None = sniff it

    try:
        df = pd.read_csv(source, sep=sep, engine="python")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error) as exc:
        # pandas surfaces "Could not determine delimiter" and friends straight
        # from the stdlib sniffer. That's a true statement about the parser and
        # a useless one to the person holding the file.
        raise UnsupportedFile(
            f"Couldn't read {name}. The file may be empty, or not really a "
            f"delimited table. Open it and check there's a header row and at "
            f"least one row of data. (Parser said: {exc})"
        ) from exc

    if df.empty and not len(df.columns):
        raise UnsupportedFile(f"{name} has no columns and no rows.")

    notes = []
    if len(df.columns) == 1 and _suffix(name) != ".txt":
        notes.append(
            "Only one column was found — the delimiter may not have been "
            "detected. Check the file separator."
        )
    return LoadResult(df=df, source_name=str(name), notes=notes)


def _read_json(source, name: str) -> LoadResult:
    _rewind(source)
    raw = source.read() if hasattr(source, "read") else Path(source).read_text()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    data = json.loads(raw)
    notes = []

    if isinstance(data, dict):
        # Survey APIs love {"results": [...]} or {"data": [...]}.
        for key in ("data", "results", "records", "rows", "responses"):
            if isinstance(data.get(key), list):
                notes.append(f"Read the list under **{key}**.")
                data = data[key]
                break

    df = pd.json_normalize(data)
    if df.empty:
        raise UnsupportedFile("That JSON has no records this can turn into a table.")
    if any("." in str(c) for c in df.columns):
        notes.append("Flattened nested fields into dotted column names.")

    return LoadResult(df=df, source_name=str(name), notes=notes)


def load(source, name: str | None = None, sheet: str | None = None) -> LoadResult:
    """Read a data file. Raises UnsupportedFile with something actionable.

    `source` is a path or an uploaded buffer; `name` supplies the extension
    when the buffer doesn't have one.
    """
    name = str(name or getattr(source, "name", "") or source)
    ext = _suffix(name)

    if ext in EXCEL_EXT:
        result = _read_excel(source, name, sheet)
    elif ext in CSV_EXT:
        result = _read_csv(source, name)
    elif ext in JSON_EXT:
        result = _read_json(source, name)
    else:
        raise UnsupportedFile(
            f"{ext or 'That file'} isn't a format this reads. "
            f"Use one of: {', '.join(sorted(SUPPORTED))}"
        )

    if result.df.empty:
        raise UnsupportedFile(f"{name} opened fine but has no rows in it.")

    result.df.columns = [str(c).strip() for c in result.df.columns]
    return result

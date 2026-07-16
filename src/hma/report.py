"""Branded PDF reports.

A research report that shows results without showing what was done to the data
is worth very little. So the ledger goes *in* the PDF, as a numbered audit
trail. A reviewer reading this document can see every transformation that
stands between the raw file and the numbers on page one.

The executive summary is the only LLM-written part, and it's clearly labelled
as such. Everything else is computed.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import brand, llm
from .ledger import Ledger
from .profiling import Profile

# Straight from brand.py, which took them straight from the logo files.
INK = colors.HexColor(brand.NAVY)
ACCENT = colors.HexColor(brand.BLUE)
ROOF = colors.HexColor(brand.ORANGE)
MUTED = colors.HexColor(brand.MUTED)
LINE = colors.HexColor(brand.LINE)
TINT = colors.HexColor(brand.TINT)

_SUMMARY_SYSTEM = """You are writing the executive summary of a survey data quality report.

Write 3-4 short paragraphs in plain English for a researcher or programme
manager, not a developer. Cover: what shape the data is in, what was fixed,
and what they must be careful about when analysing it.

Be specific and use the numbers given. Do not invent findings. Do not
recommend imputing skip-logic columns. If the data has serious problems, say
so plainly — this is a quality report, not marketing."""


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=21, textColor=INK,
                                spaceBefore=10, spaceAfter=3, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=16),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=ACCENT, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, leading=14, alignment=TA_LEFT, spaceAfter=6),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=8, textColor=MUTED, leading=11),
        "righttag": ParagraphStyle("rt", parent=base["Normal"], fontSize=8.5,
                                   alignment=TA_RIGHT, textColor=MUTED),
    }


def _letterhead(s: dict) -> Table:
    """Logo left, org tagline right, orange rule beneath.

    The rule echoes the roof in the mark. It's the cheapest way to make a
    reportlab PDF read as *yours* rather than as a default template.
    """
    logo_path = brand.asset(brand.LOGO_HORIZONTAL)
    if logo_path:
        logo = RLImage(logo_path)
        # Lock the aspect ratio off the real file — never hard-code it.
        with PILImage.open(logo_path) as im:
            width, height = im.size
        logo.drawWidth = 46 * mm
        logo.drawHeight = 46 * mm * height / width
        left = logo
    else:
        left = Paragraph(f"<b>{brand.ORG}</b>", s["h2"])

    right = Paragraph(
        f'<font color="{brand.SLATE}">{brand.ORG_TAGLINE}</font>', s["righttag"]
    )

    head = Table([[left, right]], colWidths=[90 * mm, 84 * mm])
    head.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 1.6, ROOF),
        ])
    )
    return head


def _table(data: list[list], widths: list[float], align_right: list[int] | None = None) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for col in align_right or []:
        style.append(("ALIGN", (col, 1), (col, -1), "RIGHT"))
    table.setStyle(TableStyle(style))
    return table


def executive_summary(profile_before: Profile, profile_after: Profile, ledger: Ledger) -> tuple[str, bool]:
    """Returns (text, was_llm_written). Degrades to a computed summary offline."""
    facts = (
        f"Before cleaning: {profile_before.n_rows:,} rows x {profile_before.n_cols} columns, "
        f"quality score {profile_before.quality_score}/100.\n"
        f"After cleaning: {profile_after.n_rows:,} rows x {profile_after.n_cols} columns, "
        f"quality score {profile_after.quality_score}/100.\n"
        f"Operations applied: {len(ledger)}.\n\n"
        f"Findings:\n" + "\n".join(f"- {f}" for f in profile_before.findings[:20])
    )
    try:
        return llm.complete(_SUMMARY_SYSTEM, facts, temperature=0.3), True
    except llm.LLMUnavailable:
        delta = profile_after.quality_score - profile_before.quality_score
        skip = [f.column for f in profile_before.findings if f.kind == "structural_missing"]
        pii = [f.column for f in profile_before.findings if f.kind == "pii"]
        parts = [
            f"The dataset arrived with {profile_before.n_rows:,} rows across "
            f"{profile_before.n_cols} columns and scored {profile_before.quality_score}/100 "
            f"on the quality index. After {len(ledger)} recorded operations it scores "
            f"{profile_after.quality_score}/100, a change of {delta:+d} points.",
        ]
        if skip:
            parts.append(
                f"Treat these columns as skip-logic gated and do not impute them: "
                f"{', '.join(skip[:6])}. Analyse them only among respondents who were asked."
            )
        if pii:
            parts.append(
                f"Potentially identifying columns were detected: {', '.join(pii[:6])}. "
                f"De-identify before sharing this dataset outside the study team."
            )
        parts.append(
            "Every change is listed in the audit trail below and is reproducible "
            "via the exported cleaning script."
        )
        return "\n\n".join(parts), False


def insights_markdown(
    profile_before: Profile,
    profile_after: Profile,
    ledger: Ledger,
    specs: list | None = None,
    never_impute: set[str] | None = None,
    dataset_name: str = "dataset.csv",
) -> str:
    """Findings, actions, and cautions as plain markdown.

    The PDF is for handing to someone. This is for working with: it diffs in
    git, greps in a terminal, and pastes into a methods section. A deliverable
    you can only look at is a deliverable you can't build on.
    """
    stamp = datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC")
    delta = profile_after.quality_score - profile_before.quality_score
    never_impute = never_impute or set()

    lines = [
        f"# Data quality insights — {dataset_name}",
        "",
        f"*{brand.PRODUCT} · {brand.ORG} · {brand.ORG_TAGLINE}*",
        f"*Generated {stamp}*",
        "",
        "## Scorecard",
        "",
        "| | Before | After |",
        "|---|---|---|",
        f"| Quality score | {profile_before.quality_score}/100 | "
        f"{profile_after.quality_score}/100 ({delta:+d}) |",
        f"| Rows | {profile_before.n_rows:,} | {profile_after.n_rows:,} |",
        f"| Columns | {profile_before.n_cols} | {profile_after.n_cols} |",
        f"| Duplicate rows | {profile_before.n_duplicate_rows:,} | "
        f"{profile_after.n_duplicate_rows:,} |",
        "",
        "## What was found",
        "",
    ]

    if profile_before.findings:
        lines += ["| Column | Issue | Confidence | Detail |", "|---|---|---|---|"]
        for f in profile_before.findings:
            detail = f.detail.replace("|", "\\|")
            lines.append(
                f"| `{f.column}` | {f.kind.replace('_', ' ')} | "
                f"{f.confidence:.0%} | {detail} |"
            )
    else:
        lines.append("Nothing flagged.")

    lines += ["", "## What was done", ""]
    if len(ledger):
        lines += ["| # | Operation | Rows affected | Why |", "|---|---|---|---|"]
        for e in ledger.entries:
            why = (e.rationale or "—").replace("|", "\\|")
            lines.append(
                f"| {e.seq} | `{e.op}` | {e.stats.get('rows_affected', 0):,} | {why} |"
            )
        lines += [
            "",
            "Every step above is reproducible: `clean.py` regenerates the cleaned "
            "file from the raw one using pandas alone.",
        ]
    else:
        lines.append("No operations were applied.")

    if specs:
        lines += ["", "## What to look at", ""]
        for spec in specs:
            lines += [
                f"### {spec.title}",
                "",
                f"*{spec.kind}, on {', '.join(f'`{c}`' for c in spec.columns[:4])}*",
                "",
                spec.rationale,
                "",
            ]

    lines += ["", "## Cautions", ""]
    cautions = []
    if never_impute:
        cautions.append(
            f"**Do not impute** these columns: "
            f"{', '.join(f'`{c}`' for c in sorted(never_impute))}. They're skip-logic "
            f"gated or multi-select members — a blank is an answer, not a gap. "
            f"Analyse them only among respondents who were actually asked."
        )
    pii = [f.column for f in profile_before.findings if f.kind == "pii"]
    if pii:
        cautions.append(
            f"**Potentially identifying columns**: {', '.join(f'`{c}`' for c in pii)}. "
            f"De-identify before sharing this dataset outside the study team."
        )
    if any(e.op == "fill_missing" for e in ledger.entries):
        cautions.append(
            "**Imputation was applied.** The fill values were computed from this "
            "dataset. If you later split into train and test sets, refit the "
            "imputer on the training set alone — fitting on everything leaks test "
            "information into training and inflates your scores."
        )
    cautions.append(
        "The quality score is a rough guide, not a certificate. It counts blanks, "
        "duplicates, constant columns and sentinel codes. It cannot tell you "
        "whether the questions were any good."
    )
    lines += [f"- {c}" for c in cautions]
    lines.append("")
    return "\n".join(lines)


_FINDINGS_SYSTEM = """You are writing the executive summary of a findings report for a
hospital director or programme manager. They will not read past this page.

Write 3-4 short paragraphs in plain English. Lead with the single most decision-
relevant number. Use the figures given and do not invent any.

Rules you must hold to:
- Never claim causation. Differences between groups are differences, not effects.
- Never use the word "significant" — no statistical testing was done.
- Always keep the denominator next to a percentage.
- If a finding rests on a small group or an incomplete field, say so in the
  same sentence, not in a footnote.

Write for someone deciding what to do, not for someone auditing the data."""


def findings_summary(
    dataset_name: str,
    n_rows: int,
    items: list,
) -> tuple[str, bool]:
    """Returns (text, was_llm_written). Degrades to a computed summary offline."""
    facts = (
        f"Dataset: {dataset_name}, {n_rows:,} records.\n\n"
        + "\n".join(f"- {f.headline}. {f.detail}" for f in items[:8])
    )
    try:
        return llm.complete(_FINDINGS_SYSTEM, facts, temperature=0.3), True
    except llm.LLMUnavailable:
        paragraphs = [
            f"This report covers {n_rows:,} records from {dataset_name}. "
            f"The figures below are computed directly from a cleaned copy of "
            f"that file; the cleaning script that produced it is reproducible "
            f"and available alongside this document."
        ]
        for item in items[:3]:
            paragraphs.append(f"{item.headline}. {item.detail}")
        paragraphs.append(
            "Differences reported here are differences, not causes. No "
            "statistical testing has been performed, and groups may differ in "
            "ways this dataset never recorded."
        )
        return "\n\n".join(paragraphs), False


def build_findings_report(
    df: pd.DataFrame,
    items: list,
    recommendations: list[str],
    dataset_name: str = "dataset.csv",
    charts: list | None = None,
    ledger: Ledger | None = None,
) -> bytes:
    """The report you submit. Not the one you keep.

    `build_report` documents the data's condition — what was wrong, what was
    changed. That's for you and your reviewer. This one answers the question a
    director actually asked: what does the data say, and what should we do?

    The cleaning is reduced to a single provenance line pointing at the audit
    report. Someone deciding whether to change a protocol does not need to read
    about sentinel recoding, and burying the finding under it is how good
    analysis goes unread.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Findings — {dataset_name}",
        author=brand.ORG,
    )
    s = _styles()
    story = [_letterhead(s), Paragraph("Findings Report", s["title"])]
    story.append(
        Paragraph(
            f"{dataset_name} &nbsp;·&nbsp; {len(df):,} records &nbsp;·&nbsp; "
            f"{datetime.now(UTC).strftime('%d %B %Y')}",
            s["subtitle"],
        )
    )

    summary, by_llm = findings_summary(dataset_name, len(df), items)
    story.append(Paragraph("Executive summary", s["h2"]))
    for para in summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("\n", " "), s["body"]))
    story.append(
        Paragraph(
            "Summary written by a language model from the computed figures in "
            "this report; every number below is computed from the data."
            if by_llm else
            "Summary generated from the computed figures below.",
            s["small"],
        )
    )

    story.append(Paragraph("What the data shows", s["h2"]))
    for item in items:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>{item.headline}</b>", s["body"]))
        story.append(Paragraph(item.detail, s["body"]))
        if item.caveat:
            story.append(
                Paragraph(f'<i>{item.caveat}</i>', s["small"])
            )
        if item.table is not None and not item.table.empty:
            story.append(Spacer(1, 4))
            story.append(_df_table(item.table.head(10), s))
        story.append(Spacer(1, 6))

    if recommendations:
        story.append(PageBreak())
        story.append(Paragraph("Recommended next steps", s["h2"]))
        for line in recommendations:
            story.append(
                Paragraph(
                    f"• {line.replace('**', '')}",
                    s["body"],
                )
            )
            story.append(Spacer(1, 4))

    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", s["h2"]))
        for spec, png in charts:
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{spec.title}</b>", s["body"]))
            image = RLImage(io.BytesIO(png))
            with PILImage.open(io.BytesIO(png)) as opened:
                width, height = opened.size
            image.drawWidth = 150 * mm
            image.drawHeight = 150 * mm * height / width
            story.append(image)
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 12))
    provenance = (
        f"These figures come from a cleaned copy of {dataset_name}"
        + (f", produced by {len(ledger)} recorded operations" if ledger else "")
        + ". The full audit trail and a script that reproduces the cleaned file "
        "from the raw export are available in the accompanying data quality "
        "report. No statistical testing has been performed; differences shown "
        "are descriptive."
    )
    story.append(Paragraph("Provenance", s["h2"]))
    story.append(Paragraph(provenance, s["small"]))
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            f"{brand.PRODUCT} · {brand.ORG} · {brand.ORG_TAGLINE}",
            s["small"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def build_report(
    profile_before: Profile,
    profile_after: Profile,
    ledger: Ledger,
    dataset_name: str = "dataset.csv",
    answers: list | None = None,
    charts: list | None = None,
) -> bytes:
    """Render the PDF and return the bytes.

    `charts` takes [(ChartSpec, png_bytes)] and is optional — the report is
    complete without it. Charts are placed after the audit trail on purpose:
    a reader should meet the evidence for how the data was cleaned before
    they meet the pictures drawn from it.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"{brand.PRODUCT} — {dataset_name}",
        author=brand.ORG,
    )
    s = _styles()
    story = []

    story.append(_letterhead(s))
    story.append(Paragraph("Data Quality Report", s["title"]))
    story.append(
        Paragraph(
            f"{dataset_name} &nbsp;·&nbsp; generated "
            f"{datetime.now(UTC).strftime('%d %B %Y, %H:%M UTC')}",
            s["subtitle"],
        )
    )

    # Scorecard
    story.append(
        _table(
            [
                ["", "Before", "After"],
                ["Quality score", f"{profile_before.quality_score}/100", f"{profile_after.quality_score}/100"],
                ["Rows", f"{profile_before.n_rows:,}", f"{profile_after.n_rows:,}"],
                ["Columns", str(profile_before.n_cols), str(profile_after.n_cols)],
                ["Duplicate rows", f"{profile_before.n_duplicate_rows:,}", f"{profile_after.n_duplicate_rows:,}"],
            ],
            [70 * mm, 50 * mm, 50 * mm],
            align_right=[1, 2],
        )
    )

    summary, by_llm = executive_summary(profile_before, profile_after, ledger)
    story.append(Paragraph("Executive summary", s["h2"]))
    for para in summary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("\n", " "), s["body"]))
    story.append(
        Paragraph(
            "Summary written by a language model from the computed figures above."
            if by_llm else
            "Summary generated from the computed figures above (no model was available).",
            s["small"],
        )
    )

    # Findings
    findings = profile_before.findings
    if findings:
        story.append(Paragraph("What we found", s["h2"]))
        rows = [["Column", "Issue", "Conf.", "Detail"]]
        for f in findings[:22]:
            rows.append([
                Paragraph(f"<b>{f.column}</b>", s["small"]),
                Paragraph(f.kind.replace("_", " "), s["small"]),
                f"{f.confidence:.0%}",
                Paragraph(f.detail, s["small"]),
            ])
        story.append(_table(rows, [30 * mm, 26 * mm, 13 * mm, 101 * mm], align_right=[2]))

    story.append(PageBreak())

    # The audit trail — the reason this report is worth anything
    story.append(Paragraph("Audit trail", s["h2"]))
    story.append(
        Paragraph(
            "Every change made to the data, in order. This report is reproducible: "
            "the exported cleaning script regenerates the cleaned dataset from the "
            "raw file using only pandas.",
            s["body"],
        )
    )
    if len(ledger):
        rows = [["#", "Operation", "Rows", "Shape", "Why"]]
        for e in ledger.entries:
            rows.append([
                str(e.seq),
                Paragraph(f"<b>{e.op}</b><br/><font size=6>{_params(e.params)}</font>", s["small"]),
                f"{e.stats.get('rows_affected', 0):,}",
                f"{e.rows_before}x{e.cols_before}\n-> {e.rows_after}x{e.cols_after}",
                Paragraph(e.rationale or "—", s["small"]),
            ])
        story.append(_table(rows, [8 * mm, 45 * mm, 15 * mm, 28 * mm, 78 * mm], align_right=[2]))
    else:
        story.append(Paragraph("No operations were applied.", s["body"]))

    # Charts — after the audit trail, never before it. A picture drawn from
    # data whose cleaning you haven't seen is a picture you can't evaluate.
    if charts:
        story.append(PageBreak())
        story.append(Paragraph("Charts", s["h2"]))
        story.append(
            Paragraph(
                "Drawn from the cleaned data above, not the raw file. Each one "
                "notes what to look for — and what would make it misleading.",
                s["body"],
            )
        )
        for spec, png in charts:
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{spec.title}</b>", s["body"]))
            story.append(Paragraph(spec.rationale, s["small"]))
            story.append(Spacer(1, 4))

            image = RLImage(io.BytesIO(png))
            with PILImage.open(io.BytesIO(png)) as opened:
                width, height = opened.size
            image.drawWidth = 150 * mm
            image.drawHeight = 150 * mm * height / width
            story.append(image)
            story.append(Spacer(1, 10))

    # Q&A
    if answers:
        story.append(Paragraph("Questions asked", s["h2"]))
        for answer in answers:
            story.append(Paragraph(f"<b>{answer.question}</b>", s["body"]))
            if answer.explanation:
                story.append(Paragraph(answer.explanation, s["small"]))
            if not answer.result.empty:
                story.append(Spacer(1, 3))
                story.append(_df_table(answer.result.head(12), s))
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Generated by {brand.PRODUCT} · {brand.ORG} · {brand.ORG_TAGLINE}<br/>"
            f"Figures are computed directly from the dataset; no values are "
            f"estimated by a language model.",
            s["small"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def _df_table(df: pd.DataFrame, s: dict) -> Table:
    header = [Paragraph(f"<b>{c}</b>", s["small"]) for c in df.columns]
    rows = [header] + [
        [Paragraph(_fmt(v), s["small"]) for v in row] for row in df.itertuples(index=False)
    ]
    width = 174 * mm / max(len(df.columns), 1)
    return _table(rows, [width] * len(df.columns))


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)[:60]


def _params(params: dict) -> str:
    parts = []
    for key, value in params.items():
        text = str(value)
        parts.append(f"{key}={text[:40] + '...' if len(text) > 40 else text}")
    return ", ".join(parts)

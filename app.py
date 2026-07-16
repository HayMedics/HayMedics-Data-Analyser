"""HayMedics Data Analyser — clean data with a receipt.

    uv run streamlit run app.py
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hma import brand, charts, findings, llm, loaders, questions  # noqa: E402
from hma.agents import answer_question, front_desk, plan_from_findings, propose_plan  # noqa: E402
from hma.config import settings  # noqa: E402
from hma.ledger import Ledger  # noqa: E402
from hma.pipeline import check_order  # noqa: E402
from hma.profiling import profile  # noqa: E402
from hma.report import (  # noqa: E402
    build_findings_report,
    build_report,
    insights_markdown,
)

st.set_page_config(
    page_title=brand.PRODUCT,
    page_icon=brand.asset(brand.ICON) or "📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# Brand styling
#
# The stylesheet lives in brand.py alongside the palette it uses, so the
# colours and the rules that consume them can't drift apart. It must be
# injected before anything else renders, or the first paint is unstyled.
# --------------------------------------------------------------------------

st.markdown(brand.css(), unsafe_allow_html=True)

KIND_ICONS = {
    "two_row_header": "🧱", "sentinel_missing": "🔢", "likert_scale": "📊",
    "multiselect_group": "☑️", "pii": "🔒", "structural_missing": "🚧",
    "case_variants": "🔤", "duplicate_rows": "👯", "constant_columns": "➖",
    "implausible_range": "📏", "date_like": "📅", "numeric_like": "🔣",
}


def _safe_name(text: str) -> str:
    """Chart titles become filenames in the zip. Not every title is a filename."""
    cleaned = re.sub(r"[^\w\s-]", "", str(text)).strip()
    return re.sub(r"[\s_-]+", "_", cleaned)[:60] or "chart"


def roofline() -> None:
    st.markdown('<div class="hma-roofline"></div>', unsafe_allow_html=True)


def reset(df: pd.DataFrame, name: str, notes: list[str] | None = None) -> None:
    st.session_state.df = df.copy()
    st.session_state.ledger = Ledger(source_name=name)
    st.session_state.profile_before = profile(df)
    st.session_state.answers = []
    st.session_state.name = name
    st.session_state.load_notes = notes or []
    for key in ("plan", "pdf", "findings_pdf", "rendered", "chart_specs",
                "picked_question"):
        st.session_state.pop(key, None)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    logo = brand.asset(brand.LOGO_STACKED_TAGLINE)
    if logo:
        st.image(logo, width="stretch")
    else:
        st.title(brand.ORG)

    st.markdown(
        f'<p class="hma-eyebrow">Data Analyser</p>'
        f'<p class="hma-promise">{brand.PROMISE}</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    upload = st.file_uploader(
        "Upload data",
        type=["csv", "tsv", "xlsx", "xlsm", "xls", "json"],
        help="A workbook with several tabs will ask which one holds the data.",
    )

    sheet_choice = None
    if upload is not None:
        # A workbook is often instructions + data + codebook. Reading whichever
        # tab happens to be first is how you silently analyse a Read Me.
        if Path(upload.name).suffix.lower() in loaders.EXCEL_EXT:
            try:
                sheets = loaders.excel_sheets(upload, upload.name)
            except loaders.UnsupportedFile as exc:
                sheets = []
                st.error(str(exc))

            if len(sheets) > 1:
                labels = [s.label for s in sheets]
                picked = st.selectbox(
                    "Which sheet?", labels, index=0,
                    help="Ordered by how much they look like data rather than notes.",
                )
                sheet_choice = sheets[labels.index(picked)].name
            elif sheets:
                sheet_choice = sheets[0].name

        signature = (upload.name, sheet_choice)
        if st.session_state.get("signature") != signature:
            try:
                result = loaders.load(upload, upload.name, sheet_choice)
                reset(result.df, result.source_name, result.notes)
                st.session_state.signature = signature
            except loaders.UnsupportedFile as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Couldn't read that file: {exc}")

    if st.button("Use the demo survey", width="stretch"):
        demo = Path(__file__).parent / "data" / "messy_survey.csv"
        if demo.exists():
            result = loaders.load(demo)
            reset(result.df, "messy_survey.csv", result.notes)
            st.session_state.signature = ("demo", None)
        else:
            st.warning("Generate it first: `uv run python scripts/make_messy_survey.py`")

    st.divider()
    if llm.is_available():
        st.success(f"Model connected · `{settings.model}`")
    else:
        st.info(
            "**No API key set.** Profiling, cleaning, charts, the ledger and the "
            "script export all still work. Only plain-English questions need a key."
        )
    st.caption(f"{brand.ORG} · {brand.ORG_TAGLINE}")

# --------------------------------------------------------------------------
# Landing
# --------------------------------------------------------------------------

if "df" not in st.session_state:
    header = brand.asset(brand.LOGO_HORIZONTAL)
    if header:
        st.image(header, width=380)
    roofline()

    st.markdown(f"### {brand.PROMISE}")
    st.markdown(
        """
        Upload a survey export. This profiles it, proposes fixes **in the order
        the fixes have to happen**, records every change, suggests the charts
        worth looking at, and hands you a runnable script that reproduces the
        whole session from the raw file.

        No spreadsheet formulas. No SQL. No dedicated data analyst.

        **The cleaning sequence it enforces**

        | # | Step | Why it can't move |
        |---|---|---|
        | 1 | Fix structure | Everything below reads the wrong cells until the shape is right |
        | 2 | Recode disguised missing | Before types, or `to_numeric` silently eats every refusal |
        | 3 | Fix types | Safe only once `999` is already `NaN` |
        | 4 | Standardise values | Before dedupe, or ` Lagos` and `lagos` survive as two rows |
        | 5 | Remove duplicates | Rows aren't comparable until their values are normalised |
        | 6 | Handle missing | Impute only what's genuinely missing, never skip logic |
        | 7 | Handle outliers | A range check needs a numeric column to check against |
        | 8 | Validate | Cross-field logic, once each field is individually sound |
        | 9 | De-identify | Last: dedupe and validation need the identifiers |

        Every ordering mistake above is silent. Nothing raises when you average
        a column that still contains `999` — you just get a wrong answer that
        looks fine.
        """
    )
    st.info("Upload a file in the sidebar, or click **Use the demo survey** to see it work.")
    st.markdown(
        f'<p class="hma-foot">{brand.ORG} · {brand.ORG_TAGLINE}</p>',
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------

df: pd.DataFrame = st.session_state.df
ledger: Ledger = st.session_state.ledger
before = st.session_state.profile_before
current = profile(df)

head_left, head_right = st.columns([4, 1])
with head_left:
    st.markdown(
        f'<p class="hma-eyebrow">{brand.ORG}</p>'
        f'<h2 style="margin-top:-0.5rem;margin-bottom:0;">Data Analyser</h2>',
        unsafe_allow_html=True,
    )
with head_right:
    icon = brand.asset(brand.ICON)
    if icon:
        st.image(icon, width=62)
roofline()

st.caption(f"Working on **{st.session_state.name}**")
for note in st.session_state.get("load_notes", []):
    st.info(note, icon="📄")

left, mid, right, far = st.columns(4)
delta = current.quality_score - before.quality_score
left.metric("Quality score", f"{current.quality_score}/100", f"{delta:+d}" if delta else None)
mid.metric("Rows", f"{current.n_rows:,}",
           f"{current.n_rows - before.n_rows:+,}" if current.n_rows != before.n_rows else None)
right.metric("Columns", current.n_cols,
             f"{current.n_cols - before.n_cols:+d}" if current.n_cols != before.n_cols else None)
far.metric("Ledger steps", len(ledger))

st.write("")

# Bound here, before the tabs, and not inside any of them.
#
# Streamlit executes every tab body on every rerun, but a name assigned inside
# one tab's `if` branch still doesn't exist when another tab reads it. This was
# only bound under `elif specs:` in the Charts tab, so Export crashed with a
# NameError on the most ordinary path there is: load, clean, build the PDF,
# never having opened Charts at all.
rendered: list = st.session_state.get("rendered", [])

tabs = st.tabs(["Findings", "Clean", "Charts", "Ask", "Ledger", "Export"])

# --- Findings -------------------------------------------------------------
with tabs[0]:
    st.caption("Found with plain Python. No model, no tokens, no cost.")
    if not current.findings:
        st.success("Nothing flagged. This data is unusually tidy.")
    for f in current.findings:
        icon = KIND_ICONS.get(f.kind, "•")
        with st.expander(f"{icon}  **{f.column}** — {f.kind.replace('_', ' ')} ({f.confidence:.0%})"):
            st.write(f.detail)
            if f.suggested_op:
                st.code(f"{f.suggested_op}({f.suggested_params})", language="python")
            else:
                st.info("No automatic fix. This one needs your judgement.")

    with st.expander("How the quality score is calculated"):
        st.write(current.score_breakdown)

# --- Clean ----------------------------------------------------------------
with tabs[1]:
    col_a, col_b = st.columns([3, 1])
    instruction = col_a.text_input(
        "Tell the Cleaning Agent what you want (optional)",
        placeholder="e.g. de-identify everything and fix the age column",
    )
    use_llm = col_b.toggle(
        "Use the model", value=False,
        help="Off = a deterministic plan built from the findings. Free, and identical every run.",
    )

    if st.button("Propose a plan", type="primary"):
        with st.spinner("Planning..."):
            st.session_state.plan = (
                propose_plan(current, instruction) if use_llm
                else plan_from_findings(current.findings)
            )

    plan = st.session_state.get("plan")
    if plan is not None and len(plan):
        st.caption(
            f"{len(plan.runnable)} operations, grouped into the stages they belong to. "
            f"Stages run top to bottom — that order is the method, not a preference."
        )

        counter = 0
        for stage, proposals in plan.by_stage().items():
            st.markdown(f"**{stage.value}. {stage.label}**")
            st.caption(stage.why)
            for step in proposals:
                counter += 1
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    if step.blocked:
                        c1.markdown(f"~~{step.op}~~ → `{step.target}`")
                        c1.warning(f"Held back: {step.blocked}", icon="🚧")
                    else:
                        c1.markdown(
                            f"**{step.op}** → `{step.target}`  ·  "
                            f"{step.confidence:.0%} confident"
                        )
                        c1.caption(step.rationale)
                        if c2.button("Apply", key=f"apply{counter}", width="stretch"):
                            try:
                                st.session_state.df = ledger.apply(
                                    df, step.op, step.params, step.rationale
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"{type(exc).__name__}: {exc}")

        if plan.blocked:
            st.caption(
                f"{len(plan.blocked)} operation(s) held back by a guard. Those rules "
                f"apply to the model's plans and to ours equally."
            )
    elif plan is not None:
        st.success("Nothing to fix. This data is already clean.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button(f"Apply all ≥{settings.auto_apply_threshold:.0%} confidence",
                 width="stretch", disabled=plan is None):
        applied, failed = 0, []
        for step in (plan.runnable if plan else []):
            if step.is_safe_to_auto_apply:
                try:
                    st.session_state.df = ledger.apply(
                        st.session_state.df, step.op, step.params, step.rationale
                    )
                    applied += 1
                except Exception as exc:
                    failed.append(f"{step.op}: {exc}")
        st.toast(f"Applied {applied} operations")
        for message in failed:
            st.warning(message)
        st.rerun()

    if c2.button("Undo last step", disabled=not len(ledger), width="stretch"):
        restored = ledger.undo()
        if restored is not None:
            st.session_state.df = restored
            st.rerun()

    st.dataframe(df.head(50), width="stretch")

# --- Charts ---------------------------------------------------------------
with tabs[2]:
    st.caption(
        "Suggested from the **cleaned** schema. Suggesting from raw data would "
        "recommend a histogram of an age column that still contains 999."
    )

    if not len(ledger):
        st.warning(
            "Nothing has been cleaned yet. These suggestions read the current "
            "schema, so clean first or they'll describe the mess.",
            icon="🧹",
        )

    if st.button("Suggest charts", type="primary"):
        with st.spinner("Reading the schema..."):
            st.session_state.chart_specs = charts.suggest(df)
            st.session_state.pop("rendered", None)

    specs = st.session_state.get("chart_specs")
    if specs is not None and not specs:
        st.info("No chart fits this data yet. Fix the column types and try again.")
    elif specs:
        st.caption(f"{len(specs)} charts worth your time, best first.")

        if st.button("Render all"):
            rendered, errors = [], []
            bar = st.progress(0.0)
            for i, spec in enumerate(specs):
                try:
                    rendered.append((spec, charts.to_png(charts.render(spec, df))))
                except Exception as exc:
                    errors.append(f"{spec.title}: {exc}")
                bar.progress((i + 1) / len(specs))
            bar.empty()
            st.session_state.rendered = rendered
            for message in errors:
                st.warning(message)

        rendered = st.session_state.get("rendered", [])   # refresh after "Render all"
        if rendered:
            for spec, png in rendered:
                with st.container(border=True):
                    a, b = st.columns([3, 2])
                    a.image(png, width="stretch")
                    b.markdown(f"**{spec.title}**")
                    b.caption(spec.rationale)
                    b.download_button(
                        "PNG", png, f"{spec.title.replace(' ', '_')}.png",
                        "image/png", key=f"dl_{spec.title}",
                    )
                    with b.expander("Code"):
                        st.code(spec.code, language="python")
        else:
            for spec in specs:
                with st.container(border=True):
                    st.markdown(f"**{spec.title}**  ·  `{spec.kind}`")
                    st.caption(spec.rationale)

# --- Ask ------------------------------------------------------------------
with tabs[3]:
    if not llm.is_available():
        st.info(
            "Turning a question into SQL needs an API key. The suggestions below "
            "are computed from your schema and work without one — they'll tell you "
            "what this dataset can answer even if you run the query elsewhere."
        )

    st.caption("Questions this data can actually answer, best first.")
    suggestions = questions.suggest_questions(df)

    if not suggestions:
        st.info("Clean the data first — suggestions are read off the cleaned columns.")
    else:
        # A blank "ask me anything" box asks you to already know what's in the
        # data. If you knew that, you wouldn't need the tool.
        picked = st.session_state.get("picked_question", "")
        for i, q in enumerate(suggestions):
            with st.container(border=True):
                a, b = st.columns([5, 1])
                a.markdown(f"**{q.text}**")
                a.caption(q.why)
                if b.button("Ask", key=f"q{i}", width="stretch"):
                    st.session_state.picked_question = q.text
                    st.rerun()

    st.divider()
    question = st.text_input(
        "Or ask your own",
        value=st.session_state.pop("picked_question", ""),
        placeholder="What share of each treatment arm was readmitted?",
    )
    if st.button("Ask", type="primary", key="ask_free") and question:
        with st.spinner("Routing..."):
            route = front_desk(question)
        st.caption(f"Front Desk → **{route.route}** · {route.reason}")
        with st.spinner("Computing..."):
            answer = answer_question(df, question)

        if answer.error:
            st.error(answer.error)
        else:
            st.session_state.answers.append(answer)
            st.write(answer.explanation)
            st.dataframe(answer.result, width="stretch")
            if answer.chart and len(answer.result.columns) >= 2:
                x, y = answer.result.columns[0], answer.result.columns[1]
                plot = answer.result.set_index(x)[y]
                chart = {"bar": st.bar_chart, "line": st.line_chart}.get(answer.chart, st.bar_chart)
                chart(plot, color=brand.BLUE)
            with st.expander("SQL that produced this"):
                st.code(answer.sql, language="sql")

# --- Ledger ---------------------------------------------------------------
with tabs[4]:
    if not len(ledger):
        st.info("Nothing applied yet. Propose a plan on the Clean tab to start the record.")
    else:
        st.caption("Every change, in order. This is what makes the numbers checkable.")
        violations = check_order([e.op for e in ledger.entries])
        if violations:
            st.warning("This session ran out of conventional order:", icon="⚠️")
            for v in violations:
                st.caption(f"• {v}")
        else:
            st.success("Every step ran in the conventional order.", icon="✅")
        st.dataframe(ledger.to_dataframe(), width="stretch", hide_index=True)

# --- Export ---------------------------------------------------------------
with tabs[5]:
    st.subheader("Cleaning script")
    st.caption(
        "Runs on pandas alone — no API key, and none of this app. Commit it beside "
        "your analysis and your cleaning stays reproducible for as long as pandas exists."
    )
    script = ledger.to_script()
    st.code(
        script[:1800] + ("\n\n# ... truncated for display" if len(script) > 1800 else ""),
        language="python",
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button("clean.py", script, "clean.py", "text/x-python", width="stretch")
    c2.download_button("cleaned.csv", df.to_csv(index=False), "cleaned.csv",
                       "text/csv", width="stretch")
    c3.download_button("ledger.json", ledger.to_json(), "ledger.json",
                       "application/json", width="stretch")

    st.divider()
    st.subheader("Insights, dashboard and report")

    rendered = st.session_state.get("rendered", [])
    stem = Path(st.session_state.name).stem
    plan_obj = st.session_state.get("plan")
    never_impute = getattr(plan_obj, "never_impute", set())

    insights = insights_markdown(
        before, current, ledger,
        specs=[spec for spec, _ in rendered] or st.session_state.get("chart_specs"),
        never_impute=never_impute,
        dataset_name=st.session_state.name,
    )
    html = charts.dashboard_html(rendered, st.session_state.name) if rendered else None

    if not rendered:
        st.caption(
            "Render the charts on the **Charts** tab and they'll be included here "
            "too — the downloads below work either way."
        )

    d1, d2 = st.columns(2)
    d1.download_button(
        "insights.md", insights, f"{stem}_insights.md", "text/markdown",
        width="stretch",
        help="Findings, what was done, what to look at, and what to be careful of.",
    )
    if html:
        d2.download_button(
            "dashboard.html", html, f"{stem}_dashboard.html", "text/html",
            width="stretch",
            help="Self-contained: images are embedded, so it opens anywhere with no internet.",
        )

    st.write("")
    st.markdown("**Two different reports. They answer different questions.**")

    r1, r2 = st.columns(2)
    with r1:
        st.caption(
            "**Findings report** — what the data says, and what to do about it. "
            "This is the one you submit."
        )
        if st.button("Build findings report", type="primary", width="stretch"):
            with st.spinner("Computing findings..."):
                items = findings.analyse(df)
                st.session_state.findings_pdf = build_findings_report(
                    df, items, findings.recommendations(df, items),
                    st.session_state.name, rendered, ledger,
                )
        if st.session_state.get("findings_pdf"):
            st.download_button(
                "Download findings.pdf", st.session_state.findings_pdf,
                f"hma_findings_{stem}.pdf", "application/pdf", width="stretch",
            )

    with r2:
        st.caption(
            "**Data quality report** — what was wrong and what was changed. "
            "This is the one your reviewer asks for."
        )
        if st.button("Build quality report", width="stretch"):
            with st.spinner("Building report..."):
                st.session_state.pdf = build_report(
                    before, current, ledger, st.session_state.name,
                    st.session_state.answers, rendered,
                )
        if st.session_state.get("pdf"):
            st.download_button(
                "Download quality.pdf", st.session_state.pdf,
                f"hma_quality_{stem}.pdf", "application/pdf", width="stretch",
            )

    # The bundle is built unconditionally. It used to appear only once charts
    # were rendered, which meant the one download that carries everything was
    # hidden from anyone who just wanted their cleaned data and the receipt.
    st.write("")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cleaned.csv", df.to_csv(index=False))
        zf.writestr("clean.py", script)
        zf.writestr("ledger.json", ledger.to_json())
        zf.writestr("insights.md", insights)
        if html:
            zf.writestr("dashboard.html", html)
        for spec, png in rendered:
            zf.writestr(f"charts/{_safe_name(spec.title)}.png", png)
        if st.session_state.get("pdf"):
            zf.writestr("quality_report.pdf", st.session_state.pdf)
        if st.session_state.get("findings_pdf"):
            zf.writestr("findings_report.pdf", st.session_state.findings_pdf)

    contents = ["cleaned.csv", "clean.py", "ledger.json", "insights.md"]
    if html:
        contents.append("dashboard.html")
    if rendered:
        contents.append(f"{len(rendered)} charts")
    if st.session_state.get("pdf"):
        contents.append("quality_report.pdf")
    if st.session_state.get("findings_pdf"):
        contents.append("findings_report.pdf")

    st.download_button(
        "Everything (.zip)", bundle.getvalue(), f"{stem}_bundle.zip",
        "application/zip", width="stretch",
        help="Contains: " + ", ".join(contents),
    )
    st.caption("Bundle contains: " + ", ".join(contents) + ".")

st.markdown(
    f'<p class="hma-foot">{brand.PRODUCT} · {brand.ORG} · {brand.ORG_TAGLINE}</p>',
    unsafe_allow_html=True,
)

"""Chart suggestions and rendering.

Two jobs, kept apart on purpose:

    suggest(df)   reads the CLEANED schema and says what's worth plotting
    render(spec)  draws it

Suggestions run on the *cleaned* frame, and that ordering is the whole point.
Suggest charts from raw data and you recommend a histogram of an age column
that still contains 999, or a bar chart of `gender` with Male, male, and
" Female " as three separate bars. The cleaning is what makes the suggestions
trustworthy — which is why this module refuses to guess when a column still
looks dirty.

Every spec carries the matplotlib code that drew it, for the same reason the
ledger carries the pandas code that cleaned it: a chart you can't reproduce is
a chart nobody should cite.
"""

from __future__ import annotations

import base64
import io
import re
import textwrap
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # headless: no display, no Tk, no surprises on a server

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import brand, survey  # noqa: E402

# Cardinality above this stops being a bar chart and starts being a wall.
MAX_CATEGORIES = 14
MIN_ROWS = 5

ORDER = [brand.BLUE, brand.ORANGE, brand.INDIGO, "#5E8AC7", "#F7C46C", "#7A6FBE"]


@dataclass
class ChartSpec:
    """One recommended chart, with the reason it's worth your time."""

    kind: str
    title: str
    columns: list[str]
    rationale: str
    score: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def code(self) -> str:
        """The matplotlib that draws this, for the export."""
        cols = ", ".join(repr(c) for c in self.columns)
        return CODE_TEMPLATES.get(self.kind, "# no template").format(
            cols=cols,
            col=repr(self.columns[0]) if self.columns else "''",
            col2=repr(self.columns[1]) if len(self.columns) > 1 else "''",
            title=repr(self.title),
        )


CODE_TEMPLATES = {
    "histogram": (
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.hist(df[{col}].dropna(), bins=20)\n"
        "ax.set_title({title})"
    ),
    "likert_bar": (
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "# .value_counts() on an ordered category keeps the scale order,\n"
        "# which is exactly what order_likert bought us.\n"
        "counts = df[{col}].value_counts().sort_index()\n"
        "ax.barh([str(i) for i in counts.index], counts.values)\n"
        "ax.set_title({title})"
    ),
    "category_bar": (
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "counts = df[{col}].value_counts().head(14)\n"
        "ax.barh([str(i) for i in counts.index][::-1], counts.values[::-1])\n"
        "ax.set_title({title})"
    ),
    "timeseries": (
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "series = df.set_index({col}).resample('D').size()\n"
        "ax.plot(series.index, series.values)\n"
        "ax.set_title({title})"
    ),
    "grouped_bar": (
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "table = pd.crosstab(df[{col}], df[{col2}], normalize='index') * 100\n"
        "table.plot(kind='barh', stacked=True, ax=ax)\n"
        "ax.set_title({title})"
    ),
    "box_by_group": (
        "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
        "groups = [g[{col2}].dropna().values for _, g in df.groupby({col}, observed=True)]\n"
        "ax.boxplot(groups, tick_labels=[str(k) for k, _ in df.groupby({col}, observed=True)])\n"
        "ax.set_title({title})"
    ),
    "scatter": (
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.scatter(df[{col}], df[{col2}], alpha=0.5)\n"
        "ax.set_title({title})"
    ),
    "missingness": (
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "pct = (df.isna().mean() * 100).sort_values(ascending=True)\n"
        "ax.barh(pct.index, pct.values)\n"
        "ax.set_title({title})"
    ),
}


# --------------------------------------------------------------------------
# Column classification
# --------------------------------------------------------------------------

def _is_ordered_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) and s.dtype.ordered


def _is_categorical(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    return bool(s.nunique(dropna=True) <= MAX_CATEGORIES)


def _is_plottable_numeric(s: pd.Series) -> bool:
    # A bool column is a flag, not a measurement — a histogram of it says nothing.
    if pd.api.types.is_bool_dtype(s):
        return False
    return pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 5


def _is_flag(name: str, s: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(s) or str(name).endswith(
        ("_out_of_range", "_invalid", "_flag")
    )


def _multiselect_members(columns: list[str]) -> set[str]:
    """Member columns of a collapsed multi-select.

    Once collapse_multiselect has run, `Q7_selected` and `Q7_count` hold the
    answer. Charting `Q7_Part_3` on its own asks "how many people ticked
    Television", stripped of the fact that they could tick five things — the
    bar is true and the reading of it is wrong.
    """
    stems = {c[: -len("_count")] for c in columns if c.endswith("_count")}
    members = set()
    for col in columns:
        for pattern in survey.MULTISELECT_PATTERNS:
            match = re.match(pattern, col)
            if match and match.group("stem") in stems:
                members.add(col)
                break
    return members


def classify(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sort the cleaned columns into the roles that decide what can be drawn.

    Two kinds of column are held back from plotting, because both produce
    charts that render perfectly and mean nothing:

      - identifiers. `respondent_name` with four distinct values looks exactly
        like a demographic to a cardinality check, so "Q2 by respondent_name"
        gets suggested. It's a chart of people's names, and it's a privacy
        breach dressed as an analysis.
      - collapsed multi-select members, which are now double-counted by the
        `_count` and `_selected` columns that replaced them.

    They're held back for different reasons, so they get different labels.
    Calling `Q7_Part_3` an identifier would be telling you it holds personal
    data, which is both untrue and the wrong lesson.
    """
    identifiers = {f.column for f in survey.detect_pii(df) if f.confidence >= 0.7}
    members = _multiselect_members([str(c) for c in df.columns])

    roles: dict[str, list[str]] = {
        "ordered": [], "categorical": [], "numeric": [],
        "datetime": [], "flag": [], "identifier": [], "free_text": [],
        "multiselect_member": [],
    }
    for name in df.columns:
        s = df[name]
        label = str(name)

        # Members first: a column can match both, and "superseded" is the
        # more specific and more useful thing to say about it.
        if label in members:
            roles["multiselect_member"].append(label)
            continue
        if label in identifiers:
            roles["identifier"].append(label)
            continue

        if _is_flag(label, s):
            roles["flag"].append(label)
        elif pd.api.types.is_datetime64_any_dtype(s):
            roles["datetime"].append(label)
        elif _is_ordered_categorical(s):
            roles["ordered"].append(label)
        elif _is_plottable_numeric(s):
            roles["numeric"].append(label)
        elif _is_categorical(s):
            roles["categorical"].append(label)
        elif s.nunique(dropna=True) > MAX_CATEGORIES:
            # High-cardinality text: either an ID or free prose. Neither plots.
            roles["identifier" if s.nunique() > len(s) * 0.9 else "free_text"].append(label)
    return roles


# --------------------------------------------------------------------------
# Suggestion engine
# --------------------------------------------------------------------------

def suggest(df: pd.DataFrame, limit: int = 12) -> list[ChartSpec]:
    """Recommend charts for this specific dataset, best first.

    Scores encode what a researcher actually needs to see, in order:
    a Likert distribution beats a histogram of a row index, and a scale
    broken down by demographic beats either — that's usually the paper.
    """
    roles = classify(df)
    specs: list[ChartSpec] = []

    if len(df) < MIN_ROWS:
        return specs

    # 1. Likert distributions. The reason order_likert exists.
    for col in roles["ordered"][:6]:
        specs.append(ChartSpec(
            kind="likert_bar",
            title=f"Distribution of {col}",
            columns=[col],
            score=0.95,
            rationale=(
                f"{col} is an ordered scale, so the bars come out in scale order "
                f"rather than alphabetically. Look for the shape: a flat spread "
                f"means genuine disagreement, a pile-up at one end often means "
                f"the question was leading."
            ),
        ))

    # 2. Scale by demographic — usually the actual research question.
    grouping = [c for c in roles["categorical"] if df[c].nunique(dropna=True) <= 8]
    for scale in roles["ordered"][:2]:
        for group in grouping[:2]:
            specs.append(ChartSpec(
                kind="grouped_bar",
                title=f"{scale} by {group}",
                columns=[group, scale],
                score=0.92,
                rationale=(
                    f"Splits {scale} by {group} as row percentages, so groups of "
                    f"different sizes stay comparable. This is usually the "
                    f"finding — a difference here is what the survey was for."
                ),
            ))

    # 3. Category frequencies.
    for col in roles["categorical"][:6]:
        n = df[col].nunique(dropna=True)
        specs.append(ChartSpec(
            kind="category_bar",
            title=f"Responses by {col}",
            columns=[col],
            score=0.80,
            rationale=(
                f"{n} categories. Check the balance — a category holding almost "
                f"everyone gives you no power to compare, and a tiny one won't "
                f"survive subgroup analysis."
            ),
        ))

    # 4. Numeric distributions.
    for col in roles["numeric"][:6]:
        specs.append(ChartSpec(
            kind="histogram",
            title=f"Distribution of {col}",
            columns=[col],
            score=0.78,
            rationale=(
                f"Shape of {col}. Look for a second hump (two populations mixed), "
                f"a hard wall at one end (a cap or a floor in the instrument), or "
                f"a lone spike (a missing-data code that survived cleaning)."
            ),
        ))

    # 5. Numeric by group.
    for num in roles["numeric"][:2]:
        for group in grouping[:2]:
            specs.append(ChartSpec(
                kind="box_by_group",
                title=f"{num} by {group}",
                columns=[group, num],
                score=0.74,
                rationale=(
                    f"Compares the spread of {num} across {group}, not just the "
                    f"averages. Two groups can share a mean and have completely "
                    f"different distributions."
                ),
            ))

    # 6. Fieldwork over time.
    for col in roles["datetime"][:2]:
        specs.append(ChartSpec(
            kind="timeseries",
            title=f"Responses over time ({col})",
            columns=[col],
            score=0.70,
            rationale=(
                "Response volume by day. Gaps mean fieldwork stopped; a single "
                "enormous spike often means one enumerator bulk-entering, which "
                "is worth checking before you trust that day's data."
            ),
        ))

    # 7. Relationships between measures.
    if len(roles["numeric"]) >= 2:
        a, b = roles["numeric"][0], roles["numeric"][1]
        specs.append(ChartSpec(
            kind="scatter",
            title=f"{a} vs {b}",
            columns=[a, b],
            score=0.66,
            rationale=(
                f"Whether {a} and {b} move together. Look at the shape before "
                f"reaching for a correlation — a curve or a few far-out points "
                f"will produce a number that means nothing."
            ),
        ))

    # 8. Missingness. Last by score, first thing a reviewer asks about.
    if df.isna().any().any():
        specs.append(ChartSpec(
            kind="missingness",
            title="Missing data by column",
            # Multi-select members are excluded deliberately. A blank in
            # Q7_Part_3 means "didn't tick Television" — counting it as
            # missing would draw a 60%-missing bar for a column that is not
            # missing anything, which is the single confusion this whole tool
            # exists to prevent. Identifiers stay: this plots column names and
            # counts, never values, so there's nothing to leak and knowing
            # your ID column has gaps is worth seeing.
            columns=[c for c in df.columns if str(c) not in _multiselect_members(
                [str(x) for x in df.columns]
            )],
            score=0.62,
            rationale=(
                "Percentage blank per column. Near-100% bars are skip logic and "
                "belong there. It's the middling ones — 20% to 60% — that decide "
                "whether a variable is usable at all. Collapsed multi-select "
                "columns are left out: a blank there is an answer, not a gap."
            ),
        ))

    specs.sort(key=lambda s: -s.score)
    return specs[:limit]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=brand.NAVY, fontsize=11, fontweight="600", pad=10, loc="left")
    ax.set_xlabel(xlabel, color=brand.MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=brand.MUTED, fontsize=9)
    ax.tick_params(colors=brand.MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(brand.LINE)
    ax.grid(axis="x" if ax.get_yticklabels() else "y", color=brand.LINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _wrap(labels, width: int = 22) -> list[str]:
    return ["\n".join(textwrap.wrap(str(v), width)) or str(v) for v in labels]


def render(spec: ChartSpec, df: pd.DataFrame):
    """Draw a spec. Returns a matplotlib Figure, branded.

    Raises ValueError if the data can't support the chart — better a clear
    error than an empty pair of axes claiming there's nothing there.
    """
    kind = spec.kind
    cols = [c for c in spec.columns if c in df.columns or kind == "missingness"]
    if kind != "missingness" and not cols:
        raise ValueError(f"{spec.title}: columns are gone from the frame")

    if kind == "histogram":
        fig, ax = plt.subplots(figsize=(7, 4))
        values = df[cols[0]].dropna()
        if values.empty:
            raise ValueError(f"{cols[0]} has no values left to plot")
        ax.hist(values, bins=min(20, max(5, values.nunique())),
                color=brand.BLUE, edgecolor="white", linewidth=0.6)
        _style(ax, spec.title, cols[0], "Respondents")

    elif kind in ("likert_bar", "category_bar"):
        fig, ax = plt.subplots(figsize=(7, 4))
        s = df[cols[0]]
        counts = s.value_counts().sort_index() if kind == "likert_bar" \
            else s.value_counts().head(MAX_CATEGORIES)
        if counts.empty:
            raise ValueError(f"{cols[0]} has no values left to plot")
        labels = _wrap(counts.index)[::-1]
        ax.barh(labels, counts.values[::-1], color=brand.BLUE, height=0.68)
        for y, v in enumerate(counts.values[::-1]):
            ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=brand.MUTED)
        _style(ax, spec.title, "Respondents")

    elif kind == "timeseries":
        fig, ax = plt.subplots(figsize=(7, 4))
        series = df.dropna(subset=[cols[0]]).set_index(cols[0]).resample("D").size()
        if series.empty:
            raise ValueError(f"{cols[0]} has no parseable dates")
        ax.plot(series.index, series.values, color=brand.BLUE, linewidth=1.8)
        ax.fill_between(series.index, series.values, color=brand.BLUE, alpha=0.12)
        _style(ax, spec.title, "", "Responses per day")
        fig.autofmt_xdate(rotation=30, ha="right")

    elif kind == "grouped_bar":
        fig, ax = plt.subplots(figsize=(8, 4.5))
        table = pd.crosstab(df[cols[0]], df[cols[1]], normalize="index") * 100
        if table.empty:
            raise ValueError(f"no overlap between {cols[0]} and {cols[1]}")
        table.index = _wrap(table.index, 18)
        table.plot(kind="barh", stacked=True, ax=ax,
                   color=ORDER[: len(table.columns)], width=0.7)
        _style(ax, spec.title, "% of group")
        ax.legend(title=cols[1], fontsize=7, title_fontsize=7,
                  bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    elif kind == "box_by_group":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        grouped = [(k, g[cols[1]].dropna().values)
                   for k, g in df.groupby(cols[0], observed=True)]
        grouped = [(k, v) for k, v in grouped if len(v)]
        if not grouped:
            raise ValueError(f"no values of {cols[1]} within {cols[0]}")
        box = ax.boxplot([v for _, v in grouped], patch_artist=True,
                         tick_labels=_wrap([k for k, _ in grouped], 12),
                         medianprops={"color": brand.ORANGE, "linewidth": 1.6})
        for patch in box["boxes"]:
            patch.set_facecolor(brand.BLUE)
            patch.set_alpha(0.55)
            patch.set_edgecolor(brand.BLUE)
        _style(ax, spec.title, cols[0], cols[1])

    elif kind == "scatter":
        fig, ax = plt.subplots(figsize=(6, 5))
        pair = df[[cols[0], cols[1]]].dropna()
        if pair.empty:
            raise ValueError(f"{cols[0]} and {cols[1]} never both have values")
        ax.scatter(pair[cols[0]], pair[cols[1]], color=brand.BLUE, alpha=0.45, s=18,
                   edgecolors="none")
        _style(ax, spec.title, cols[0], cols[1])

    elif kind == "missingness":
        pct = (df.isna().mean() * 100).sort_values()
        fig, ax = plt.subplots(figsize=(7, max(3.0, 0.26 * len(pct))))
        # Colour carries meaning: orange is the band where you have a decision.
        colours = [brand.ORANGE if 20 <= v <= 60 else brand.BLUE for v in pct.values]
        ax.barh(list(pct.index), pct.values, color=colours, height=0.7)
        _style(ax, spec.title, "% missing")
        ax.set_xlim(0, 100)

    else:
        raise ValueError(f"unknown chart kind: {kind}")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def to_png(fig, dpi: int = 150) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def dashboard_html(rendered: list[tuple[ChartSpec, bytes]], dataset: str) -> str:
    """A self-contained HTML dashboard. No server, no CDN, no internet.

    Images are inlined as base64 so the file survives being emailed, put on a
    USB stick, or opened in three years. A dashboard with external
    dependencies is a dashboard with an expiry date.
    """
    cards = []
    for spec, png in rendered:
        b64 = base64.b64encode(png).decode()
        cards.append(f"""
      <figure class="card">
        <img src="data:image/png;base64,{b64}" alt="{spec.title}">
        <figcaption>
          <h3>{spec.title}</h3>
          <p>{spec.rationale}</p>
        </figcaption>
      </figure>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{brand.ORG} — Dashboard — {dataset}</title>
<style>
  :root {{
    --blue: {brand.BLUE}; --orange: {brand.ORANGE};
    --navy: {brand.NAVY}; --line: {brand.LINE}; --muted: {brand.MUTED};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2rem 1.5rem 4rem;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--navy); background: {brand.TINT};
  }}
  header {{ max-width: 1180px; margin: 0 auto 1.6rem; }}
  .eyebrow {{
    font-size: .7rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: var(--blue); margin: 0;
  }}
  h1 {{ margin: .1rem 0 .2rem; font-size: 1.65rem; }}
  .sub {{ color: var(--muted); font-size: .85rem; margin: 0; }}
  .roofline {{
    height: 3px; border-radius: 2px; margin: .9rem 0 0;
    background: linear-gradient(90deg, var(--orange) 0 62%, var(--blue) 62% 100%);
  }}
  .grid {{
    max-width: 1180px; margin: 0 auto; display: grid; gap: 1.1rem;
    grid-template-columns: repeat(auto-fit, minmax(430px, 1fr));
  }}
  .card {{
    margin: 0; background: #fff; border: 1px solid var(--line);
    border-radius: 10px; padding: 1rem; display: flex; flex-direction: column;
  }}
  .card img {{ width: 100%; height: auto; }}
  figcaption h3 {{ margin: .7rem 0 .3rem; font-size: .95rem; }}
  figcaption p {{ margin: 0; font-size: .82rem; color: var(--muted); line-height: 1.5; }}
  footer {{
    max-width: 1180px; margin: 2.4rem auto 0; padding-top: .8rem;
    border-top: 1px solid var(--line); color: var(--muted); font-size: .76rem;
  }}
  @media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
  <header>
    <p class="eyebrow">{brand.ORG}</p>
    <h1>Dashboard — {dataset}</h1>
    <p class="sub">{len(rendered)} charts · every figure computed from the cleaned dataset</p>
    <div class="roofline"></div>
  </header>
  <main class="grid">{"".join(cards)}
  </main>
  <footer>{brand.PRODUCT} · {brand.ORG} · {brand.ORG_TAGLINE}</footer>
</body>
</html>"""

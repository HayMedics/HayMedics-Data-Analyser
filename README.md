<p align="center">
  <img src="assets/hma_stacked_tagline.png" width="300" alt="HayMedics Academy">
</p>

<p align="center">
  <a href="https://github.com/HayMedics/HayMedics-Data-Analyser/actions/workflows/ci.yml"><img src="https://github.com/HayMedics/HayMedics-Data-Analyser/actions/workflows/ci.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-189-brightgreen" alt="tests">
</p>

<h1 align="center">HayMedics Data Analyser</h1>

<p align="center"><strong>Clean data with a receipt.</strong></p>

<p align="center">
  <em>HayMedics Academy · Data | Research | Innovation</em>
</p>

---

Upload a survey export. This profiles it, proposes fixes, records every change,
and hands you a **runnable Python script** that reproduces the entire session
from the raw file — using pandas alone, with no API key and none of this app.

No spreadsheet formulas. No SQL. No dedicated data analyst.

That last part is the point. A cleaning tool you can't audit is a tool that
makes your results unverifiable.

---

## Why this exists

Generic AI data cleaners treat every blank cell the same way: fill it. On
survey data that's not a shortcut, it's a fabrication.

| Looks like | Actually is | Generic cleaner does | This does |
|---|---|---|---|
| `999` in an age column | "declined to answer" | averages it in → mean age 340 | recodes to NaN, tells you why |
| Blank in `Q7_Part_3` | "didn't tick that option" | imputes the mode | leaves it — a blank *is* the answer |
| `Q9` 95% empty | skip logic — never asked | imputes 95% of the column | flags it, refuses to impute |
| `Agree` / `Strongly agree` | an ordered scale | unordered strings → wrong median | restores ordinality |
| `Lagos` / `lagos` / `LAGOS` | one place | three categories | collapses them |
| `respondent_name` | a person | ships it to your repo | flags for de-identification |

---

## Quick start

```bash
git clone https://github.com/HayMedics/HayMedics-Data-Analyser.git
cd haymedics-data-analyser

uv sync                                        # install
uv run python scripts/make_messy_survey.py     # generate synthetic test data
uv run streamlit run app.py                    # launch
```

Click **Use the demo survey** in the sidebar. No API key needed for that.

For plain-English questions, add a key:

```bash
cp .env.example .env
# paste your key from https://openrouter.ai/keys
```

Run the tests:

```bash
uv run pytest
```

## Testing on real Kaggle data

The synthetic generator covers every pathology, but real data is better proof.
The Kaggle ML & DS Survey is ideal — it has the two-row header and multi-select
columns this is built for:

```bash
uv pip install kaggle
# put your kaggle.json in ~/.kaggle/ first — https://www.kaggle.com/settings

kaggle datasets download -d kaggle/kaggle-survey-2018 -p data --unzip
```

Then upload `data/multipleChoiceResponses.csv` in the app. Watch it catch the
question-text header row before anything else.

---

## The cleaning pipeline

The conventional order, encoded as `Stage` in `ops.py`. Every op declares its
stage, and the planner sorts by it — so the sequence is a property of the code,
not a convention someone has to remember.

| # | Stage | Ops | Why it sits here |
|---|---|---|---|
| 1 | Fix structure | `promote_header_row`, `drop_repeated_header_rows`, `collapse_multiselect`, `drop_constant_columns`, `drop_empty_rows`, `drop_column`, `rename_column` | A question-text row above the data means every step below reads the wrong cells |
| 2 | Recode disguised missing | `recode_sentinel_missing` | Before types. Coerce first and `to_numeric` silently eats "Prefer not to say" — you lose the fact someone refused, which is itself a finding |
| 3 | Fix types | `coerce_numeric`, `coerce_datetime`, `order_likert` | Only safe once the 999s are already NaN |
| 4 | Standardise values | `strip_whitespace`, `map_categories` | Must precede dedupe. Also where `Male`/`m`/`Man`/`1` become one value |
| 5 | Remove duplicates | `drop_duplicates` | `"  Lagos"` and `"lagos"` aren't equal until step 4 has run |
| 6 | Handle missing | `fill_missing` | Impute only once you know what's genuinely missing rather than disguised |
| 7 | Handle outliers | `clip_outliers` | A range check is meaningless until the column is numeric |
| 8 | Validate | `flag_out_of_range`, `flag_date_order` | Cross-field logic, once every field is individually sound |
| 9 | De-identify | `deidentify` | Last. Hash an identifier earlier and you can't dedupe on it |

`pipeline.check_order()` runs over any hand-applied session and reports where
it left the conventional order.

### Steps 13 and 14 are deliberately absent

Encoding and scaling aren't here, and that's not an oversight. Both learn
parameters from data, so they belong inside a train/test split, fitted on the
training set alone. Offering them in a cleaning tool would quietly encourage
**data leakage** — the model scores brilliantly and fails in production.

Imputation (step 6) has the same property, so where it's used the exported
script bakes in the literal fill value and `insights.md` says so.

---

## Values written more than one way

`detect_case_variants` merges `Lagos` / `lagos` / `" Lagos"` — values already
identical once lowered. That's the easy half. The hard half is that `Male`,
`m`, `Man` and `1` are four different strings meaning one thing, and no amount
of normalising whitespace merges them. A real healthcare export arrived with
`Gender` in 10 spellings and `Smoker?` in 10, and every detector returned
nothing.

`detect_value_synonyms` matches a column's values against vocabularies
(`yes_no`, `sex`) and proposes the collapse. Each vocabulary carries a regex
hint on the column NAME, because the values alone cannot resolve the real
ambiguity: **`1` means Yes in `Smoker?` and Male in `Gender`.** You have to
read the header.

Values the vocabulary doesn't recognise — `U`, `?` — are left alone. Guessing
at them would be the tool inventing answers.

| | Before | After |
|---|---|---|
| `Gender` | 10 spellings | Male, Female, U |
| `Smoker?` | 10 spellings | Yes, No (`?` → missing) |
| `Readmitted 30d` | 10 spellings | Yes, No |

---

## Charts

`charts.suggest()` reads the **cleaned** schema and ranks what's worth plotting.
Ordering matters: suggest from raw data and you recommend a histogram of an age
column that still contains 999, or a bar chart of `gender` with `Male`, `male`
and `" Female "` as three separate bars.

Every column gets a role — `ordered`, `categorical`, `numeric`, `datetime`,
`flag`, `identifier`, `free_text`, `multiselect_member` — and the roles decide
what can be drawn. Eight chart kinds: `likert_bar`, `grouped_bar`,
`category_bar`, `histogram`, `box_by_group`, `scatter`, `timeseries`,
`missingness`.

Two things are never plotted:

- **identifiers**, because "Q2 by respondent_name" is a privacy breach dressed
  as an analysis. (The missingness chart is the one exception: it plots column
  names and blank counts, never values.)
- **collapsed multi-select members**, because they're already counted by the
  `_count` column that replaced them — and a missingness bar for `Q7_Part_3`
  would report 60% missing for a column missing nothing.

Each suggestion carries the reason it's worth your time and what would make it
misleading, and every spec renders its own matplotlib source — a chart you
can't reproduce is a chart nobody should cite.

---

## Architecture

```
Upload → loaders     (sheet choice, header detection)
            ↓
         Profile     (deterministic, zero tokens)
            ↓
         Findings    (survey.py — suggests, never applies)
            ↓
         Pipeline    (stages 1-9, guard rails)
            ↓
    ┌───────┴────────┐
 Cleaning Agent   Analyst Agent
 emits JSON ops   emits SQL
    ↓                ↓
 Op Registry      DuckDB
    ↓                ↓
         Ledger  →  clean.py + cleaned.csv + insights.md
                    + charts + dashboard.html + report.pdf
```

### Four rules the code enforces

**1. The LLM never computes.** It routes, picks ops by name, writes SQL. Every
number comes from pandas or DuckDB over your actual data.

**2. The registry is the boundary.** The Cleaning Agent returns
`{"op": ..., "params": ...}`. Not in `ops.py`, doesn't run.
`analyst.guard()` rejects anything but a single read-only SELECT.

**3. The guard binds us too.** `never_impute_columns()` lives in exactly one
place and both planners call it. A rule that only constrains the model is a
rule you don't actually believe.

**4. Every LLM call is optional.** No key? Profiling, findings, cleaning,
charts, the ledger and the script export all still work.

---

## Branding

Everything that identifies this as a HayMedics product lives in
`src/hma/brand.py` — name, palette, logos, tagline. Changing the product name
is one edit in one file, never a find-and-replace.

The palette was sampled from the logo files, not eyeballed:

| Token | Hex | Where it comes from |
|---|---|---|
| `BLUE` | `#2E57A6` | the pillars, and "Medics" |
| `ORANGE` | `#FEA621` | the roof |
| `INDIGO` | `#14077B` | "Hay" |
| `NAVY` | `#0D033F` | "Academy" — the darkest ink |

Streamlit's own widgets are themed in `.streamlit/config.toml` so tab
underlines, toggles and focus rings pick up the brand too — CSS alone can't
reach most of those.

### Two traps worth knowing

`brand.css()` opens with `<style>` and loads fonts via `@import`, and that
ordering is load-bearing. Streamlit renders `st.markdown` through a CommonMark
parser. A `<link>` tag opens a *type 6* HTML block, which terminates at the
first **blank line** — so the blank lines between rule groups end the raw-HTML
passthrough, and every rule after that point gets parsed as a markdown
paragraph and printed on the page as visible text. `<style>` opens a *type 1*
block that only closes at `</style>`, so blank lines are harmless.

It fails silently and it looks catastrophic. `test_brand_css_does_not_leak_into_the_page`
guards it.

Second: an op runs one way in the app and renders as pandas source for the
export. If those two ever disagree, the app shows you clean data while the
script your reviewer runs produces something else — and nothing raises. It
happened twice (`order_likert` skipped a normalisation; `coerce_datetime`
dropped `format='mixed'` and NaN'd every date pandas didn't guess first).
`tests/test_drift.py` now executes **every** op both ways and compares the
frames, and a new op can't be added without a case there.

---

## Layout

```
assets/          — logos, cropped with transparent backgrounds
src/hma/
  brand.py       — name, palette, logos, stylesheet: single source of truth
  loaders.py     — CSV/Excel/JSON; sheet choice and header detection
  survey.py      — survey detectors (sentinel codes, Likert, skip logic, PII)
  ops.py         — the operation registry: stage + run + render + schema
  pipeline.py    — stage ordering, guards, out-of-order detection
  ledger.py      — append-only record; exports the runnable script
  profiling.py   — deterministic profile + explainable quality score
  charts.py      — chart suggestions and rendering
  questions.py   — what's worth asking, read off the cleaned schema
  findings.py    — what the data says (descriptive, never inferential)
  analyst.py     — DuckDB compute layer + SQL guard
  agents.py      — Front Desk, Cleaning Agent, Analyst Agent
  llm.py         — OpenRouter client
  report.py      — branded PDF + insights.md
app.py           — Streamlit UI
tests/
  test_core.py     — detection, ledger, reproducibility
  test_drift.py    — every op's renderer vs its runner
  test_pipeline.py — stage order and the refusals
  test_loading.py  — Excel sheets, headers, declared dependencies
  test_charts.py   — every chart kind renders; nothing private is plotted
  test_values.py   — synonyms, binaries, questions, findings
  test_app.py      — the Streamlit app end to end
```

189 tests. `uv run pytest`, or `uv run pytest -m "not slow"` to skip the app
click-throughs.

## Two reports

They answer different questions, so they're different documents.

- **Findings report** — what the data says and what to do about it. Executive
  summary, the numbers, recommended next steps. This is the one you submit.
  Descriptive only: no p-values, no significance claims, every percentage
  carries its denominator, and every group difference carries the reminder
  that a difference is not a cause.
- **Data quality report** — what was wrong and what was changed, with the full
  audit trail. This is the one your reviewer asks for.

Burying a finding under sentinel-recoding detail is how good analysis goes
unread.

---

## What you can download

| File | What it's for |
|---|---|
| `cleaned.csv` | the data |
| `clean.py` | reproduces it from the raw file, pandas only |
| `insights.md` | findings, actions, what to look at, what to be careful of |
| `ledger.json` | the record, machine-readable |
| `dashboard.html` | self-contained, images embedded, opens offline |
| `charts/*.png` | every rendered chart |
| `findings_report.pdf` | what the data says — the one you submit |
| `quality_report.pdf` | what was cleaned — the one your reviewer asks for |
| `bundle.zip` | all of the above |

## Deploying to Streamlit Community Cloud

Entrypoint: `app.py`. Nothing else to configure — `uv.lock` is at the repo
root, and Community Cloud reads it with `uv` ahead of every other dependency
file. There is deliberately **no `requirements.txt`**: Cloud uses only the
first dependency file it finds, in the order `uv.lock` → `Pipfile` →
`environment.yml` → `requirements.txt` → `pyproject.toml`, so a requirements
file here would be silently ignored and would drift out of date unnoticed.

If you add a key for the Ask tab, put it in the app's **Secrets** (Settings →
Secrets), not in a committed `.env`:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```

### Push with git, not the web uploader

Drag-and-drop upload flattens directories and silently skips dotfolders. That
turns `src/hma/charts.py` into `charts.py`, drops `.streamlit/config.toml` and
`.github/`, and the app dies with `attempted relative import with no known
parent package` — which looks like a code bug and isn't one. Use `git push`;
it preserves the tree exactly.

---

## Licence

MIT.

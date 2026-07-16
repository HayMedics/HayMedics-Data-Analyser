"""Brand constants.

Everything that identifies this as a HayMedics product lives here and nowhere
else. The name, the palette, the logos, the tagline. If the name ever has to
change again — and it did once already — it's one edit in this file, not a
find-and-replace across a codebase.

The hex values were sampled directly out of the logo files, not eyeballed.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

PRODUCT = "HayMedics Data Analyser"
PRODUCT_SHORT = "HMA Analyser"
ORG = "HayMedics Academy"
ORG_TAGLINE = "Data | Research | Innovation"
PROMISE = "Clean data with a receipt."
REPO = "https://github.com/HayMedics/haymedics-data-analyser"

# --------------------------------------------------------------------------
# Palette — sampled from the logo, not guessed
# --------------------------------------------------------------------------

BLUE = "#2E57A6"     # the pillars, and "Medics"
ORANGE = "#FEA621"   # the roof
INDIGO = "#14077B"   # "Hay"
NAVY = "#0D033F"     # "Academy" — the darkest ink
SLATE = "#23224A"    # the tagline

# Derived, for UI states. Kept here so nothing invents its own greys.
INK = NAVY
MUTED = "#6B7280"
LINE = "#DDE3EE"
CANVAS = "#FFFFFF"
TINT = "#F4F7FC"     # very pale blue for panel fills
SUCCESS = "#1B7F5A"
WARNING = ORANGE
DANGER = "#B3261E"

# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

ASSETS = Path(__file__).resolve().parents[2] / "assets"

ICON = ASSETS / "hma_icon.png"                      # page icon / favicon
LOGO_STACKED_TAGLINE = ASSETS / "hma_stacked_tagline.png"   # sidebar
LOGO_HORIZONTAL = ASSETS / "hma_horizontal.png"     # PDF header
LOGO_STACKED = ASSETS / "hma_stacked.png"


def asset(path: Path) -> str | None:
    """Return a usable path, or None if the asset is missing.

    Assets should never be load-bearing. A missing logo degrades the look;
    it must never take the app down.
    """
    return str(path) if path.exists() else None


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------

def css() -> str:
    """The brand stylesheet, as a raw HTML block for st.markdown.

    Two rules here are not stylistic, they're structural, and breaking either
    one dumps raw CSS into the page as visible text:

    1. `<style>` MUST be the first tag. Streamlit renders this through a
       CommonMark parser. A `<link>` tag opens a "type 6" HTML block, which
       terminates at the first BLANK LINE — so the blank lines between rule
       groups would end the raw-HTML passthrough and every rule after it gets
       parsed as a markdown paragraph. `<style>` opens a "type 1" block, which
       only ends at `</style>`. Blank lines are then harmless.

    2. Fonts load via `@import`, not `<link>`, for exactly that reason. The
       `@import` must stay the first rule in the sheet or the browser drops it.

    tests/test_core.py renders this through a CommonMark parser and asserts
    nothing leaks. Don't reorder without running it.
    """
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {{
  --hma-blue: {BLUE};
  --hma-orange: {ORANGE};
  --hma-indigo: {INDIGO};
  --hma-navy: {NAVY};
  --hma-line: {LINE};
  --hma-tint: {TINT};
  --hma-muted: {MUTED};
}}

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}

/* Poppins is the closest match to the geometric sans in the wordmark. */
h1, h2, h3, h4 {{
  font-family: 'Poppins', system-ui, sans-serif !important;
  color: var(--hma-navy);
  letter-spacing: -0.01em;
}}

/* The roofline — the one bold move, taken straight from the mark. */
.hma-roofline {{
  height: 3px;
  background: linear-gradient(90deg,
              var(--hma-orange) 0%, var(--hma-orange) 62%,
              var(--hma-blue) 62%, var(--hma-blue) 100%);
  border-radius: 2px;
  margin: 0.15rem 0 1.4rem 0;
}}

.hma-eyebrow {{
  font-family: 'Poppins', sans-serif;
  font-size: 0.70rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--hma-blue);
  margin-bottom: 0.1rem;
}}

.hma-promise {{
  font-family: 'Poppins', sans-serif;
  font-size: 1.0rem;
  font-weight: 500;
  color: var(--hma-indigo);
  margin: 0;
}}

/* Metric cards stay quiet so the roofline remains the loud thing. */
div[data-testid="stMetric"] {{
  background: var(--hma-tint);
  border: 1px solid var(--hma-line);
  border-left: 3px solid var(--hma-blue);
  border-radius: 8px;
  padding: 0.85rem 1rem;
}}

div[data-testid="stMetricLabel"] p {{
  font-size: 0.72rem !important;
  font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--hma-muted) !important;
}}

div[data-testid="stMetricValue"] {{
  font-family: 'Poppins', sans-serif;
  color: var(--hma-navy);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 1.6rem;
  border-bottom: 1px solid var(--hma-line);
}}

.stTabs [data-baseweb="tab"] {{
  font-family: 'Poppins', sans-serif;
  font-weight: 500;
  padding: 0.4rem 0;
}}

section[data-testid="stSidebar"] {{ border-right: 1px solid var(--hma-line); }}

.stButton button[kind="primary"] {{
  background: var(--hma-blue);
  border: 1px solid var(--hma-blue);
  font-family: 'Poppins', sans-serif;
  font-weight: 500;
}}

.stButton button[kind="primary"]:hover {{
  background: var(--hma-indigo);
  border-color: var(--hma-indigo);
}}

.stDownloadButton button {{
  border: 1px solid var(--hma-blue);
  color: var(--hma-blue);
  font-weight: 500;
}}

.stDownloadButton button:hover {{
  background: var(--hma-blue);
  color: #fff;
}}

.hma-foot {{
  color: var(--hma-muted);
  font-size: 0.76rem;
  border-top: 1px solid var(--hma-line);
  padding-top: 0.7rem;
  margin-top: 2.2rem;
}}
</style>
"""

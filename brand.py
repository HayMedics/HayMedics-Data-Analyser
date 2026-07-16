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

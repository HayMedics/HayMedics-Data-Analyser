"""Settings. Everything tunable lives here, nothing is hard-coded elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import brand


def _load_dotenv() -> None:
    """Read .env without a dependency. Real env vars always win."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


_load_dotenv()


@dataclass
class Settings:
    # `openrouter/free` is the auto-router: it picks whichever free model is
    # actually up right now. Pinning a specific free slug means your demo dies
    # the day that model gets rate-limited or rotated out.
    model: str = field(default_factory=lambda: os.getenv("HMA_MODEL", "openrouter/free"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    app_url: str = field(default_factory=lambda: os.getenv("HMA_APP_URL", brand.REPO))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("HMA_MAX_TOKENS", "1400")))

    # A finding below this confidence is shown but never auto-applied.
    auto_apply_threshold: float = field(
        default_factory=lambda: float(os.getenv("HMA_AUTO_APPLY", "0.90"))
    )
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("HMA_MAX_UPLOAD_MB", "200")))


settings = Settings()

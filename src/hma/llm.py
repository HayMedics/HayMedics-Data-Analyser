"""OpenRouter client.

Same setup as your outreach agent: OpenAI SDK pointed at OpenRouter, using the
`openrouter/free` auto-router so free-model rotation and rate limits don't take
the app down mid-demo.

One design rule worth keeping: every LLM call here is optional. If the key is
missing or the free tier is throttled, profiling, cleaning, the ledger and the
script export all still work — you just lose plain-English input. The
deterministic core is the product; the LLM is a convenience layer on top.
"""

from __future__ import annotations

import json
import os
import re

from .config import settings


class LLMUnavailable(RuntimeError):
    """No key, no network, or the free tier is throttled."""


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise LLMUnavailable("openai package not installed") from exc

    key = settings.openrouter_api_key
    if not key:
        raise LLMUnavailable(
            "No OPENROUTER_API_KEY found. Copy .env.example to .env and add your key."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        default_headers={
            "HTTP-Referer": settings.app_url,
            "X-Title": "HayMedics Data Analyser",
        },
    )


def complete(system: str, user: str, temperature: float = 0.0) -> str:
    """One completion. Temperature 0 by default — this is analysis, not prose."""
    try:
        response = _client().chat.completions.create(
            model=settings.model,
            temperature=temperature,
            max_tokens=settings.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(f"OpenRouter call failed: {exc}") from exc

    if not response.choices:
        raise LLMUnavailable("OpenRouter returned no choices (free tier throttled?)")
    return (response.choices[0].message.content or "").strip()


def complete_json(system: str, user: str) -> dict | list:
    """Completion that must return JSON.

    Free models ignore response_format and wrap JSON in prose or fences about
    a third of the time, so we extract rather than trust.
    """
    raw = complete(
        system + "\n\nRespond with valid JSON only. No prose, no markdown fences.",
        user,
    )
    return extract_json(raw)


def extract_json(raw: str) -> dict | list:
    """Pull JSON out of whatever the model actually sent back."""
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMUnavailable(f"Could not parse JSON from model output: {raw[:200]}")


def is_available() -> bool:
    return bool(settings.openrouter_api_key) or bool(os.getenv("OPENROUTER_API_KEY"))

"""Configuration from .env. No dependency; the format is trivial enough to parse.

The API key is read from the environment first so an exported key or an `ant auth login`
profile still wins — .env is a convenience for local development, not a lockout.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Reads KEY=VALUE lines. Never overrides a variable already in the environment."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


load_env()

MODEL_PROVIDER = os.environ.get("MCGOD_MODEL_PROVIDER", "anthropic").strip().lower()
if MODEL_PROVIDER not in {"anthropic", "openrouter"}:
    raise ValueError("MCGOD_MODEL_PROVIDER must be anthropic or openrouter")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_REASONING_EFFORT = os.environ.get(
    "MCGOD_OPENROUTER_REASONING_EFFORT", "low").strip().lower()

if MODEL_PROVIDER == "openrouter":
    CLASSIFY_MODEL = os.environ.get(
        "MCGOD_OPENROUTER_CLASSIFY_MODEL", "openai/gpt-5.6-luna")
    DIALOGUE_MODEL = os.environ.get(
        "MCGOD_OPENROUTER_DIALOGUE_MODEL", "openai/gpt-5.6-luna")
    VISION_MODEL = os.environ.get(
        "MCGOD_OPENROUTER_VISION_MODEL", "google/gemini-3.7-flash")
    #: Building is its own skill. Reading a picture, holding a shape in mind and writing
    #: the blocks that make it is not what a cheap dialogue model is good at, and the
    #: statues showed it. Separate so it can be swapped and measured on its own.
    BUILD_MODEL = os.environ.get("MCGOD_OPENROUTER_BUILD_MODEL", "openai/gpt-6-astra")
else:
    CLASSIFY_MODEL = os.environ.get("MCGOD_CLASSIFY_MODEL", "claude-haiku-4-5")
    DIALOGUE_MODEL = os.environ.get("MCGOD_DIALOGUE_MODEL", "claude-opus-5")
    # Preserve the known-good Claude path unless explicitly separated.
    VISION_MODEL = os.environ.get("MCGOD_VISION_MODEL", DIALOGUE_MODEL)
    BUILD_MODEL = os.environ.get("MCGOD_BUILD_MODEL", DIALOGUE_MODEL)
INFERRED_THRESHOLD = float(os.environ.get("MCGOD_INFERRED_THRESHOLD", "0.75"))


def have_key() -> bool:
    if MODEL_PROVIDER == "openrouter":
        return bool(OPENROUTER_API_KEY)
    return bool(os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def thinking_for(model: str) -> dict:
    """The thinking parameter this model will actually accept.

    Adaptive thinking is 4.6-and-later only; asking an older model for it is a 400, not a
    graceful downgrade. That failure looked exactly like a model returning nothing useful,
    so a cost comparison silently scored Haiku on two calls it never made.
    """
    if any(tag in model for tag in ("haiku-4-5", "-4-5", "3-5", "3-7")):
        return {"type": "enabled", "budget_tokens": 4000}
    return {"type": "adaptive"}

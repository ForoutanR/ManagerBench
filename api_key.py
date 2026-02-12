from __future__ import annotations

import os


def _get_env(name: str) -> str:
    v = os.getenv(name, "")
    return v.strip()


# OpenRouter API configuration
OPENROUTER_API_KEY: str = _get_env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = _get_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

# Optional OpenRouter metadata
OPENROUTER_APP_NAME: str = _get_env("OPENROUTER_APP_NAME") or "ManagerBench"
OPENROUTER_HTTP_REFERER: str = _get_env("OPENROUTER_HTTP_REFERER")

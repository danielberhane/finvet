"""Restore the real environment for opt-in integration runs.

The root conftest injects dummy credentials so the unit suite runs with no
.env and no services. Those dummies are exported into os.environ, and
pydantic-settings ranks environment variables above .env -- so an integration
test inheriting them points at a database that does not exist and skips itself
with "RAG service unavailable", which looks like a missing service rather than
a misconfigured test.

Loading .env with override=True here, before any finvet module is imported,
puts the real values back for this directory only.
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

if _ENV_FILE.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE, override=True)
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

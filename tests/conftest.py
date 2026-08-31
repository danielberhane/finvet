"""Provide dummy credentials so the unit suite runs without a .env file.

Settings has two fields with no default (tavily_api_key, postgres_password)
and is instantiated at import time. DEEPSEEK_API_KEY is not one of them -- it
is no longer a Settings field -- but create_llm reads whichever variable a role
names, and the default role config names that one. Unit tests are fully mocked
and never reach a real service, but the values must exist first.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")

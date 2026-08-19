"""Provide dummy credentials so the unit suite runs without a .env file.

Settings has three fields with no default (deepseek_api_key, tavily_api_key,
postgres_password) and is instantiated at import time. Unit tests are fully
mocked and never reach a real service, but the values must exist first.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")

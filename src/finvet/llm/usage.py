"""Token usage as the provider reports it, read off the message it came on.

LangChain puts usage on `AIMessage.usage_metadata`; the OpenAI-compatible
adapters also mirror it under `response_metadata["token_usage"]`. The parser
used to read `response_metadata["usage"]`, a key no adapter sets, so the
counter the API published was 0 on every response while traces showed ~21k.
"""
from typing import Any


def tokens_of(message: Any) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total is None:
        meta = getattr(message, "response_metadata", None) or {}
        total = (meta.get("token_usage") or {}).get("total_tokens")
    return int(total or 0)


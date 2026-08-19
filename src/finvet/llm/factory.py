"""LLM factory — creates model instances from config.

Usage:
    from finvet.llm import create_llm

    llm = create_llm("parser")                           # bare ChatModel
    llm = create_llm("agent").bind_tools(tools)          # with tools
    llm = create_llm("verdict").with_structured_output(V, method="json_mode")
"""

import os
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config.settings import settings


def create_llm(purpose: Literal["parser", "agent", "verdict"]) -> BaseChatModel:
    """Create an LLM instance for the given purpose.

    Reads the matching LLMConfig from settings (llm_parser, llm_agent, llm_verdict).
    Falls back to settings.deepseek_api_key when the purpose-specific env var is empty,
    so the current .env works without changes.
    """
    config = getattr(settings, f"llm_{purpose}")
    api_key = os.environ.get(config.api_key_env, "") or settings.deepseek_api_key
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=config.temperature,
    )

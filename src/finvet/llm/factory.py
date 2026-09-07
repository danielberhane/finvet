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


LLM_ROLES = ("parser", "agent", "verdict")


def active_llm_config() -> dict:
    """Which model stands behind each role, for the audit record.

    A stored result that does not say what produced it cannot be compared
    against a result from another model. `base_url` is included because a model
    name alone is ambiguous across providers -- the same name can be served by
    a hosted API and by a local Ollama endpoint, and they are not the same run.
    No API key is included.
    """
    return {
        role: {
            "model": getattr(settings, f"llm_{role}").model,
            "base_url": getattr(settings, f"llm_{role}").base_url,
            "temperature": getattr(settings, f"llm_{role}").temperature,
            "structured_output_method":
                getattr(settings, f"llm_{role}").structured_output_method,
        }
        for role in LLM_ROLES
    }


def create_llm(purpose: Literal["parser", "agent", "verdict"]) -> BaseChatModel:
    """Create an LLM instance for the given purpose.

    Reads the matching LLMConfig from settings (llm_parser, llm_agent,
    llm_verdict), each independently pointable at a different provider via
    `LLM_<ROLE>__MODEL`, `__BASE_URL` and `__API_KEY_ENV`.

    **No fallback key.** This used to read

        os.environ.get(config.api_key_env, "") or settings.deepseek_api_key

    so that a .env carrying only DEEPSEEK_API_KEY worked everywhere. Harmless
    while all three roles point at DeepSeek, and a credential leak the moment
    one does not: pointing a role at another endpoint without setting that
    provider's key sent the DeepSeek key to it. The symptom is an upstream 401,
    which reads as a network or allowlist fault, so the real event -- one
    provider's credential posted to another provider -- leaves no trace.

    Raising here costs a clear error on a misconfigured .env and removes that
    failure mode entirely. The message names the variable, the role and the
    model, because a reader has three roles to check.
    """
    config = getattr(settings, f"llm_{purpose}")
    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is unset or empty, required by llm_{purpose} "
            f"({config.model} at {config.base_url}). Set it in .env, or point "
            f"this role at another provider with LLM_{purpose.upper()}__API_KEY_ENV."
        )
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=config.temperature,
        timeout=config.timeout_s,
    )

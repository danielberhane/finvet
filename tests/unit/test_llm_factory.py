"""Tests for the LLM factory module."""

import pytest
from langchain_openai import ChatOpenAI

from finvet.llm import create_llm
from finvet.config.settings import settings


class TestLLMFactory:

    @pytest.mark.parametrize("purpose", ["parser", "agent", "verdict"])
    def test_creates_chat_openai(self, purpose):
        llm = create_llm(purpose)
        assert isinstance(llm, ChatOpenAI)

    def test_parser_uses_config_model(self):
        llm = create_llm("parser")
        assert llm.model_name == settings.llm_parser.model

    def test_agent_uses_config_model(self):
        llm = create_llm("agent")
        assert llm.model_name == settings.llm_agent.model

    def test_verdict_uses_config_model(self):
        llm = create_llm("verdict")
        assert llm.model_name == settings.llm_verdict.model

    def test_default_temperature_is_zero(self):
        llm = create_llm("parser")
        assert llm.temperature == 0.0

    def test_default_model_is_deepseek(self):
        """Default config should point to deepseek-chat."""
        assert settings.llm_parser.model == "deepseek-chat"
        assert settings.llm_agent.model == "deepseek-chat"
        assert settings.llm_verdict.model == "deepseek-chat"

    def test_default_base_url(self):
        assert settings.llm_parser.base_url == "https://api.deepseek.com"

    def test_structured_output_method_default(self):
        assert settings.llm_verdict.structured_output_method == "json_mode"

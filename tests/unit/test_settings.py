"""Tests for settings configuration."""

from finvet.config.settings import LLMConfig, settings


class TestLLMConfig:

    def test_defaults(self):
        config = LLMConfig()
        assert config.model == "deepseek-chat"
        assert config.base_url == "https://api.deepseek.com"
        assert config.api_key_env == "DEEPSEEK_API_KEY"
        assert config.temperature == 0.0
        assert config.structured_output_method == "json_mode"

    def test_custom_values(self):
        config = LLMConfig(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            temperature=0.7,
            structured_output_method="json_schema",
        )
        assert config.model == "gpt-4o"
        assert config.base_url == "https://api.openai.com/v1"


class TestSettingsLLMFields:

    def test_has_llm_parser(self):
        assert hasattr(settings, "llm_parser")
        assert isinstance(settings.llm_parser, LLMConfig)

    def test_has_llm_agent(self):
        assert hasattr(settings, "llm_agent")
        assert isinstance(settings.llm_agent, LLMConfig)

    def test_has_llm_verdict(self):
        assert hasattr(settings, "llm_verdict")
        assert isinstance(settings.llm_verdict, LLMConfig)

    def test_has_llama_guard_settings(self):
        assert hasattr(settings, "llama_guard_url")
        assert hasattr(settings, "llama_guard_model")
        assert hasattr(settings, "enable_llama_guard")
        assert hasattr(settings, "llama_guard_timeout_s")

    def test_llama_guard_defaults(self):
        assert settings.llama_guard_url == "http://localhost:11434"
        assert settings.llama_guard_model == "llama-guard3:8b"
        assert settings.llama_guard_timeout_s == 15.0

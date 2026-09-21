"""Configuration settings for FinVet using Pydantic Settings."""

from typing import Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Configuration for an LLM endpoint. Override via env vars like LLM_PARSER__MODEL."""

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    structured_output_method: str = "json_mode"
    # Request bound. Without one, ChatOpenAI forwards None to the OpenAI SDK,
    # which disables the SDK's own 600 s default, and a stalled provider hangs
    # the request indefinitely.
    timeout_s: float = 120.0


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # LangSmith Configuration (Optional - for debugging)
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "finvet"

    # MCP Server URLs (HTTP transport)
    sec_edgar_mcp_url: str = "http://localhost:9870"

    # The SEC EDGAR MCP server walks a filing's XBRL once per requested concept
    # with no caching, so a call costs roughly 1.25s per concept. The widest
    # request (income, 12 concepts) measures 15-17s, so a 15s budget failed it
    # systematically. 60s clears the widest request with room for a slow filing.
    sec_mcp_timeout_s: float = 60.0

    # Finnhub Configuration (Market Data)
    finnhub_api_key: Optional[str] = None
    finnhub_mock_mode: bool = False  # Use mock data instead of real API

    # SEC EDGAR user agent. SEC Fair Access requires a real name + contact address
    # on every automated request: https://www.sec.gov/os/webmaster-faq#developers
    sec_edgar_user_agent: str = "FinVet (your@email.com)"

    @property
    def sec_user_agent_is_placeholder(self) -> bool:
        """True while SEC_EDGAR_USER_AGENT still carries the shipped example contact."""
        return "your@email.com" in self.sec_edgar_user_agent


    # Embeddings (RAG + claim memory) — served locally by Ollama, no API key
    ollama_url: str = "http://localhost:11434"
    embedding_timeout_s: float = 60.0

    # Tavily Configuration
    tavily_api_key: str

    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "finvet"
    postgres_password: str
    postgres_db: str = "finvet"

    # LLM Configs (override via LLM_PARSER__MODEL=gpt-4o etc.)
    llm_parser: LLMConfig = LLMConfig()
    llm_agent: LLMConfig = LLMConfig()
    llm_verdict: LLMConfig = LLMConfig()

    # Llama Guard (semantic safety layer via Ollama)
    llama_guard_url: str = "http://localhost:11434"
    llama_guard_model: str = "llama-guard3:8b"
    enable_llama_guard: bool = False
    llama_guard_timeout_s: float = 15.0

    # Application Settings
    log_level: str = "INFO"
    confidence_threshold_hitl: float = 0.70

    # Feature Flags
    #
    # Off by default, and deliberately so. Claim memory is the one subsystem
    # whose output is prior *model* output rather than a source: reusing a
    # cached verdict, or feeding a past summary back to an agent, moves
    # something the system said into the position of something it found. That
    # is worth experimenting with and not worth shipping on. Set
    # ENABLE_CLAIM_MEMORY=true to study it; the implementation stays in the
    # repository either way.
    enable_claim_memory: bool = False

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Global settings instance
settings = Settings()

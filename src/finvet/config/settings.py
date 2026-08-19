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


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # DeepSeek Configuration
    deepseek_api_key: str

    # LangSmith Configuration (Optional - for debugging)
    langchain_api_key: Optional[str] = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "finvet"

    # MCP Server URLs (HTTP transport)
    sec_edgar_mcp_url: str = "http://localhost:9870"

    # Finnhub Configuration (Market Data)
    finnhub_api_key: Optional[str] = None
    finnhub_mock_mode: bool = False  # Use mock data instead of real API

    # SEC EDGAR user agent (required by SEC EDGAR API fair access policy)
    sec_edgar_user_agent: str = "FinVet (your@email.com)"

    # OpenAI Configuration (for embeddings)
    openai_api_key: Optional[str] = None

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
    max_tool_calls_per_agent: int = 10
    confidence_threshold_hitl: float = 0.70

    # Feature Flags
    enable_claim_memory: bool = True

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Global settings instance
settings = Settings()

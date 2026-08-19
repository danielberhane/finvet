"""Custom exceptions for FinVet."""


class FinVetError(Exception):
    """Base exception for all FinVet errors."""
    pass


class ConfigurationError(FinVetError):
    """Exception raised for configuration errors."""
    pass


class InputValidationError(FinVetError):
    """Exception raised when input validation fails."""
    pass


class GuardrailViolation(InputValidationError):
    """Exception raised when an input guardrail is violated."""

    def __init__(self, message: str, violation_type: str, details: dict = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class ParsingError(FinVetError):
    """Exception raised when claim parsing fails."""
    pass


class PeriodResolutionError(FinVetError):
    """Exception raised when period resolution fails."""
    pass


class AgentExecutionError(FinVetError):
    """Exception raised when an agent fails to execute."""

    def __init__(self, message: str, agent_name: str, details: dict = None):
        super().__init__(message)
        self.agent_name = agent_name
        self.details = details or {}


class DataRetrievalError(FinVetError):
    """Exception raised when data retrieval fails."""
    pass


class ConsensusError(FinVetError):
    """Exception raised when consensus building fails."""
    pass


class StorageError(FinVetError):
    """Exception raised for storage-related errors."""
    pass


class RateLimitExceeded(InputValidationError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after_seconds: int = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

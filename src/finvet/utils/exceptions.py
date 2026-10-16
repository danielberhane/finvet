"""Custom exceptions for FinVet."""


class FinVetError(Exception):
    """Base exception for all FinVet errors."""
    pass


class InputValidationError(FinVetError):
    """Exception raised when input validation fails."""
    pass


class AuditPersistenceError(FinVetError):
    """The execution could not be recorded, so the verdict must not be served.

    A verdict returned with no durable audit row is exactly the state the
    system claims cannot happen. Callers map this to 503 rather than degrading
    silently to an unaudited success.
    """

    def __init__(self, request_id: str):
        super().__init__(f"audit persistence failed for {request_id}")
        self.request_id = request_id


class GuardrailViolation(InputValidationError):
    """Exception raised when an input guardrail is violated."""

    def __init__(self, message: str, violation_type: str, details: dict = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class ParsingError(FinVetError):
    """Exception raised when claim parsing fails."""
    pass

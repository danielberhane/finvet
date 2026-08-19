"""FastAPI application for FinVet financial claim verification service."""

import os
from dotenv import load_dotenv

# Load .env and set LangSmith tracing BEFORE any LangChain imports
load_dotenv()

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.postgres import PostgresStore

from . import __version__
from .api import deps
from .api.routes import health, verify, review, memory, audit
from .graph.workflow import create_verification_graph
from .utils.logging import setup_logging, get_logger
from .utils.exceptions import GuardrailViolation
from .config.settings import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FinVet API",
    description="Financial claim verification system with explainable verdicts",
    version=__version__,
)

# Create verification graph with MemorySaver for HITL interrupt/resume.
checkpointer = MemorySaver()
graph = create_verification_graph(checkpointer=checkpointer)
deps.verification_graph = graph
logger.info("FinVet application started with HITL checkpointer enabled")


# Create LangGraph Store for episodic memory
def _embed_texts(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [e.embedding for e in response.data]


claim_store = None
_store_cm = None  # Keep context manager alive for app lifetime
if settings.enable_claim_memory and settings.openai_api_key:
    try:
        _store_cm = PostgresStore.from_conn_string(
            settings.postgres_url,
            index={"dims": 1536, "embed": _embed_texts, "fields": ["claim_text"]},
        )
        claim_store = _store_cm.__enter__()
        claim_store.setup()
        logger.info("LangGraph Store initialized for episodic memory")
    except Exception as e:
        logger.warning(f"Failed to initialize LangGraph Store: {e}")
        claim_store = None

if claim_store:
    from .memory.store_service import ClaimMemoryService
    from .tools import memory_tools
    deps.claim_memory = ClaimMemoryService(claim_store)
    memory_tools._set_store(claim_store)
    logger.info("Episodic memory service and agent tool configured")

# Register routers (no prefix — preserve existing endpoint paths)
app.include_router(health.router)
app.include_router(verify.router)
app.include_router(review.router)
app.include_router(memory.router)
app.include_router(audit.router)


# Exception handlers
@app.exception_handler(GuardrailViolation)
async def guardrail_violation_handler(request: Request, exc: GuardrailViolation):
    """Handle guardrail violations."""
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error_code": exc.violation_type,
            "error_message": str(exc),
            "details": exc.details,
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info(f"FinVet v{__version__} Starting Up")
    logger.info("=" * 60)
    logger.info(f"Log Level: {settings.log_level}")
    logger.info(f"HITL Checkpointer: MemorySaver (in-memory)")
    logger.info(f"HITL Confidence Threshold: {settings.confidence_threshold_hitl}")
    logger.info("Graph compiled with interrupt support")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("FinVet shutting down gracefully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

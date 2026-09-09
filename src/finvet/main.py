"""FastAPI application for FinVet financial claim verification service."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env and set LangSmith tracing BEFORE any LangChain imports
load_dotenv()

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from . import __version__
from .api import deps
from .api.routes import health, verify, review, memory, audit
from .graph.workflow import create_verification_graph
from .utils.logging import setup_logging, get_logger
from .utils.exceptions import GuardrailViolation
from .config.settings import settings
from .config.database import init_db

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown, as one hook.

    These were two `@app.on_event` handlers, deprecated since FastAPI 0.93 and
    slated for removal. A lifespan context manager is the supported form and
    keeps the pair readable: everything before `yield` runs at boot, everything
    after at shutdown.
    """
    logger.info("=" * 60)
    logger.info(f"FinVet v{__version__} Starting Up")
    logger.info("=" * 60)
    logger.info(f"Log Level: {settings.log_level}")
    logger.info("HITL Checkpointer: MemorySaver (in-memory)")
    logger.info(f"HITL Confidence Threshold: {settings.confidence_threshold_hitl}")
    logger.info("Graph compiled with interrupt support")
    logger.info("=" * 60)

    # Create the audit schema if missing (idempotent). Wrapped so a Postgres
    # that is unreachable at boot logs and continues instead of crash-looping
    # the container — audit writes degrade on their own, and the schema is
    # created on the next boot once the database is up.
    try:
        init_db()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.warning(f"Could not initialize database schema at startup: {e}")

    yield

    logger.info("FinVet shutting down gracefully")


# Create FastAPI app
app = FastAPI(
    title="FinVet API",
    description="Financial claim verification system with explainable verdicts",
    version=__version__,
    lifespan=lifespan,
)

# Create verification graph with MemorySaver for HITL interrupt/resume.
#
# Release A decision: checkpoints live in process memory, so a pending review
# does not survive an API restart. Resuming a lost checkpoint returns 409
# checkpoint_unavailable rather than a verdict computed from the reviewer's
# request -- an unresumable claim has nothing to review. Release B swaps in
# PostgresSaver via the langgraph-checkpoint-postgres dependency already
# declared, with setup during lifespan and a restart-resume test.
checkpointer = MemorySaver()
graph = create_verification_graph(checkpointer=checkpointer)
deps.verification_graph = graph
logger.info("FinVet application started with HITL checkpointer enabled")


# Create LangGraph Store for episodic memory. Shares the RAG embedder so both
# subsystems stay on one locally-served model and one vector dimension.
from .config.constants import EMBEDDING_DIMS
from .rag.service import _embed_texts

claim_store = None
_store_cm = None  # Keep context manager alive for app lifetime
if settings.enable_claim_memory:
    try:
        # Imported here, not at module scope. langgraph-checkpoint-postgres is
        # an optional Release-A dependency (extra: memory), and claim memory
        # ships disabled -- a top-level import would stop the API booting on a
        # default install for a feature that install does not use.
        #
        # Inside the `try` so that enabling memory *without* the extra degrades
        # to disabled, which is the documented behaviour, rather than crashing
        # at startup.
        from langgraph.store.postgres import PostgresStore

        _store_cm = PostgresStore.from_conn_string(
            settings.postgres_url,
            index={
                "dims": EMBEDDING_DIMS,
                "embed": _embed_texts,
                "fields": ["claim_text"],
            },
        )
        claim_store = _store_cm.__enter__()
        claim_store.setup()
        logger.info("LangGraph Store initialized for episodic memory")
    except ImportError as e:
        logger.warning(
            f"Claim memory is enabled but its optional dependency is missing "
            f"({e}); continuing with memory disabled. Install it with: "
            f"uv sync --extra memory")
        claim_store = None
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

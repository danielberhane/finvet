"""Health and stats endpoints."""

from datetime import datetime
from fastapi import APIRouter

from ... import __version__
from ...llm.factory import active_llm_config
from ..models import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info."""
    return {
        "status": "running",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Reports the models this process will actually use. A benchmark client
    cannot determine that for itself -- it runs elsewhere, and reading its own
    environment tells it only what it was launched with.
    """
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
        "llm": active_llm_config(),
    }


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "service": "FinVet",
        "version": __version__,
        "status": "operational",
        "features": {
            "agents": ["SEC", "News", "Market"],
            "truth_regimes": ["XBRL_GAAP", "MARKET_DATA", "NEWS_REPORTING"],
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

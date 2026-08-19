"""Health and stats endpoints."""

from datetime import datetime
from fastapi import APIRouter

from ... import __version__
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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
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

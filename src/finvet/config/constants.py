"""Named constants extracted from across the codebase.

Centralizes magic numbers for tolerances, thresholds, agent limits,
embedding config, and consensus adjustments.
"""

# ---------------------------------------------------------------------------
# Verdict tolerance thresholds (used in base.py Python verdict override)
# ---------------------------------------------------------------------------
# Measured across 255 real-sourced eq rows (claimed vs SEC-filed value):
# median 0.004%, p95 1.034%, max 3.067%. 1.0% sat below p95, so true claims
# were refuted on rounding alone; 1.5% covers p95 with margin while staying
# far under the 3.067% outlier.
TOLERANCE_SEC_LARGE = 1.5       # SEC/financial values > $1B
TOLERANCE_SEC_SMALL = 2.0       # SEC/financial values <= $1B
TOLERANCE_MARKET = 5.0          # Market data (stock prices, market cap)
TOLERANCE_NEWS = 5.0            # News-reported values
TOLERANCE_DEFAULT = 2.0         # Fallback
TOLERANCE_LARGE_VALUE_THRESHOLD = 1_000_000_000  # $1B boundary

# ---------------------------------------------------------------------------
# Consensus adjustments (used in workflow.py _simple_consensus)
# ---------------------------------------------------------------------------
CONSENSUS_LARGE_DIFF_THRESHOLD = 20     # % magnitude diff for penalty
CONSENSUS_CLOSE_MATCH_THRESHOLD = 2     # % magnitude diff for bonus
CONSENSUS_LARGE_DIFF_PENALTY = -0.1     # confidence adjustment
CONSENSUS_CLOSE_MATCH_BONUS = 0.05      # confidence adjustment
CONSENSUS_THOROUGH_BONUS = 0.05         # bonus for >= 3 tool calls
CONSENSUS_THOROUGH_TOOL_COUNT = 3       # min tools for thorough bonus
CONSENSUS_MAX_CONFIDENCE = 0.95         # confidence cap

# ---------------------------------------------------------------------------
# Agent limits (used in base.py)
# ---------------------------------------------------------------------------
AGENT_MAX_ITERATIONS = 5
AGENT_MAX_RESULT_CHARS = 50000  # ~12K tokens, safe for 131K context

# ---------------------------------------------------------------------------
# Confidence thresholds (used in helpers.py)
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MODERATE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# RAG constants (used in rag/service.py)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
EMBED_BATCH_SIZE = 50
RRF_K = 60  # Reciprocal Rank Fusion constant

# ---------------------------------------------------------------------------
# Memory thresholds (used in memory/service.py and main.py)
# ---------------------------------------------------------------------------
MEMORY_CACHE_THRESHOLD = 0.95   # near-exact match for cache hit
MEMORY_CONTEXT_THRESHOLD = 0.75 # related claims for agent augmentation
MEMORY_SIMILAR_THRESHOLD = 0.60 # broad similarity for "People Also Verified"

# ---------------------------------------------------------------------------
# Audit callback truncation (used in audit/callbacks.py)
# ---------------------------------------------------------------------------
MAX_CALLBACK_DATA_CHARS = 1000  # max chars for tool output/error in audit events

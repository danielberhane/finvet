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
# approx/range widen the base tolerance by this factor. Measured across the 36
# approx rows with a filed value: spread median 0.014%, p95 1.25%, max 2.14%;
# 2.0x the SEC-large tolerance captures 97% of them, and 3.0x would buy only a
# single 2.14% outlier at real cost to REFUTES discrimination.
TOLERANCE_APPROX_MULTIPLIER = 2.0

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

# News claims whose truth an issuer's own filing can settle, so the News agent
# delegates to SEC even when the model does not think to. Deliberately narrow:
# fines and settlements land in Legal Proceedings and contingency notes, which
# the RAG corpus indexes. Acquisitions live in 8-Ks and exhibits that are not
# ingested, and layoffs are often absent from periodic filings — both would
# manufacture NOT_ENOUGH_INFO results that say nothing. Widen only after
# measuring these two.
CORROBORATION_METRICS = frozenset({"fine_amount", "settlement_amount"})

# Budget for a delegated (nested) agent. Lower than a top-level run: it answers
# one bounded question, and it is spending the caller's step budget.
A2A_MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Confidence thresholds (used in helpers.py)
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_MODERATE_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# RAG constants (used in rag/service.py)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMS = 768
EMBED_BATCH_SIZE = 50
RRF_K = 60  # Reciprocal Rank Fusion constant
RRF_ABSENT_RANK = 1000  # Rank assigned to a chunk missing from one arm

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
# Stored tool-result preview in tool_calls_detail. Must cover a full income
# statement: 14 line-item dicts run ~2,000 chars, and a 1,000-char window cut
# NetIncomeLoss (9th item) out of real output — the metric-guided fallback
# could see revenue but not net income in the same statement. 3,000 covers 14
# items with margin at modest audit-row cost.
TOOL_RESULT_PREVIEW_CHARS = 3000

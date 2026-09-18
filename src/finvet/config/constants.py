"""Named constants extracted from across the codebase.

Centralizes magic numbers for tolerances, thresholds, agent limits,
embedding config, and confidence adjustments.
"""

# ---------------------------------------------------------------------------
# Verdict tolerance thresholds (used in base.py Python verdict override)
# ---------------------------------------------------------------------------
# Measured across 255 real-sourced eq rows (claimed vs SEC-filed value):
# median 0.004%, p95 1.034%, max 3.067%. 1.0% sat below p95, so true claims
# were refuted on rounding alone; 1.5% covers p95 with margin while staying
# far under the 3.067% outlier.
TOLERANCE_SEC_LARGE_VALUES = 1.5   # tolerance (%) for SEC values above the threshold
TOLERANCE_SEC_SMALL_VALUES = 2.0   # tolerance (%) for SEC values at or below it
TOLERANCE_MARKET = 5.0          # Market data (stock prices, market cap)
TOLERANCE_NEWS = 5.0            # News-reported values
TOLERANCE_DEFAULT = 2.0         # Fallback
SEC_LARGE_VALUE_THRESHOLD = 1_000_000_000  # the value, not a tolerance: $1B
# approx/range widen the base tolerance by this factor. Measured across the 36
# approx rows with a filed value: spread median 0.014%, p95 1.25%, max 2.14%;
# 2.0x the SEC-large tolerance captures 97% of them, and 3.0x would buy only a
# single 2.14% outlier at real cost to REFUTES discrimination.
TOLERANCE_APPROX_MULTIPLIER = 2.0

# ---------------------------------------------------------------------------
# Confidence adjustments (used in workflow.py _adjust_confidence)
# ---------------------------------------------------------------------------
CONFIDENCE_LARGE_DIFF_PCT = 20     # % magnitude diff for penalty
CONFIDENCE_CLOSE_MATCH_PCT = 2     # % magnitude diff for bonus
CONFIDENCE_LARGE_DIFF_PENALTY = -0.1     # confidence adjustment
CONFIDENCE_CLOSE_MATCH_BONUS = 0.05      # confidence adjustment
CONFIDENCE_THOROUGH_BONUS = 0.05         # bonus for >= 3 tool calls
CONFIDENCE_THOROUGH_TOOL_COUNT = 3       # min tools for thorough bonus
CONFIDENCE_AUTOMATED_CAP = 0.95         # cap for an automated verdict; 1.0 is a human's

# ---------------------------------------------------------------------------
# Agent limits (used in base.py)
# ---------------------------------------------------------------------------
# Raised from 5 on 2026-09-13. `deepseek-chat` is an alias, and the provider
# repointed it from deepseek-v4-flash to deepseek-flash between two runs on the
# same afternoon. The new model takes five tool calls where the old took four,
# and a budget of 5 gives a recursion limit of 11 -- exactly five calls with no
# headroom -- so every claim began exhausting its budget and escalating. Nothing
# in the code changed; the model behind the name did.
#
# FINNHUB_TIMEOUT_SECONDS moves with this: the two multiply, and a dead feed
# must still not cost more than a minute of pure waiting per claim. See
# tests/unit/test_finnhub_timeout_is_bounded.py, which enforces the product.
AGENT_MAX_ITERATIONS = 8
AGENT_MAX_RESULT_CHARS = 50000  # ~12K tokens, safe for 131K context

# How long to wait for a Finnhub quote before giving up. It was 30s, which is
# the wait multiplied by every retry: when Finnhub returned read timeouts during
# a benchmark, the tool reported an error, the agent tried again, and single
# claims took 600s. A quote endpoint that has not answered in 10s is not about
# to; the cost of being wrong is one NOT_ENOUGH_INFO on a live-price claim.
# 10.0 until the iteration budget rose to 8, which would have put the worst case
# at 80s. The bound is the product, so the timeout came down rather than the
# guarantee being relaxed. A quote endpoint silent for 7s is no likelier to
# answer than one silent for 10.
FINNHUB_TIMEOUT_SECONDS = 7.0

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
#
# Raised from 3 alongside AGENT_MAX_ITERATIONS, and for the same reason: the
# model behind the `deepseek-chat` alias changed and now needs more steps. At 3
# the nested run hit its recursion limit of 7 on every delegation, so the SEC
# side came back with zero sources and every fine or settlement claim declined.
# Kept well under the top-level budget, because the caller pays for it and a
# news run may delegate more than once.
A2A_MAX_ITERATIONS = 5

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

# Dense-arm relevance floor. pgvector always returns a nearest neighbour, so
# without a floor a query about a disclosure that does not exist still comes
# back with the closest passages and the tool reports success. Calibrated
# against a labelled set rather than chosen: see
# tests/accuracy/rag_relevance_cases.json and the header of test_rag.py.
# Re-measured 2026-08-26 against 1,398 chunks from 15 filings (5 tickers)
# after the Release-A re-ingestion, using the same 30 positive and 30 negative
# query/ticker pairs. The full evidence, per case, is in
# tests/accuracy/rag_release_a_manifest.json.
#
#   positives  n=30  min 0.5752
#   negatives  n=30  max 0.5307   (off-topic queries)
#
# The separating band is therefore (0.5307, 0.5752), and 0.55 sits inside it
# with margin on both sides: 0.0193 above the worst negative, 0.0252 below the
# weakest positive.
#
# It was 0.53, calibrated against the previous 988-chunk index whose band was
# (0.5168, 0.5469). Re-ingestion moved both edges up, and 0.53 ended up 0.0007
# *below* the worst negative -- close enough that an off-topic query ("scuba
# diving decompression tables", 0.5307 against TSLA filings) would clear the
# floor and the tool would return filing text for it. The threshold follows
# the corpus; it is not a constant anyone chose to like.
#
# Regenerate with: PYTHONPATH=src .venv/bin/python scripts/build_rag_manifest.py
RAG_MIN_VECTOR_SIMILARITY = 0.55

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

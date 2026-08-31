"""Layer 2 — trajectory: was the path sound, not just the answer right.

The guide calls this "the only way to distinguish good agents from lucky ones".
Outcome scoring said row 88 failed; it could not say the agent had spent 14 tool
calls hunting a company that was never named.

**Expectations come from the routing design, not from past runs.** Reading tool
calls out of previous artifacts and calling that the expectation would only
assert that the system does what it already does.
`verification_strategy_for` declares how a claim will be handled before any
agent starts, and each strategy implies the work that must happen:

    xbrl          a financial statement must actually be fetched
    filing_rag    filing text must actually be searched
    market        a quote must actually be pulled
    news_search   news must actually be searched
    unsupported   nothing should run at all

`get_company_info` and `get_recent_filings` are allowed but never required:
they are prerequisites the agent chooses on the way, not the work itself.

Two facts about this codebase that would otherwise produce wrong scores:

**A delegated SEC call is logged under the parent's request_id.** When the News
agent calls `corroborate_with_filing`, the nested SEC agent's tool calls appear
on the *news* claim's trace. SEC tools therefore stay permitted under
`news_search` -- forbidding them would flag every successful delegation as a
lane violation.

**`search_past_verifications` belongs to all three agents**, so it can never be
out of lane anywhere.

Scoring sits behind `_score_required`, the one function that touches DeepEval.
`ToolCorrectnessMetric` is a deterministic set comparison -- no judge, no API
key -- and if it is unavailable the same comparison runs inline. Only the name
is lost.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .artifacts import Run

# DeepEval ships analytics on by default and emits an event per `measure()`.
# This scores a privately held claim set, so the run stays local. Set before any
# deepeval import, since the client reads it at module load.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

# What each agent can reach. Built from the agents rather than the `*_TOOLS`
# constants, because every agent is constructed with `*_TOOLS + [extras]` and
# the extras -- search_filing_text, corroborate_with_filing -- appear in no
# constant. `tests/unit/test_measures_trajectory.py` pins these to the agent
# modules so the map cannot drift from the code.
SEC_AGENT_TOOLS = frozenset({
    "get_company_info", "get_recent_filings", "get_income_statement",
    "get_balance_sheet", "get_cash_flow", "search_filing_text",
})
MARKET_AGENT_TOOLS = frozenset({
    "get_stock_quote", "get_daily_prices", "get_company_overview", "get_earnings",
})
NEWS_AGENT_TOOLS = frozenset({
    "search_financial_news", "verify_news_source", "corroborate_with_filing",
})
# Shared by every agent, so never a lane violation.
SHARED_TOOLS = frozenset({"search_past_verifications"})

STATEMENT_TOOLS = frozenset({
    "get_income_statement", "get_balance_sheet", "get_cash_flow"})

# strategy -> (one of these must be called, none of these may be)
EXPECTED: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    "xbrl": (STATEMENT_TOOLS, MARKET_AGENT_TOOLS | NEWS_AGENT_TOOLS),
    "filing_rag": (frozenset({"search_filing_text"}),
                   MARKET_AGENT_TOOLS | NEWS_AGENT_TOOLS),
    "market": (frozenset({"get_stock_quote"}),
               SEC_AGENT_TOOLS | NEWS_AGENT_TOOLS),
    # SEC tools stay allowed: a delegated corroboration logs them here.
    "news_search": (frozenset({"search_financial_news"}), MARKET_AGENT_TOOLS),
    "unsupported": (frozenset(), SEC_AGENT_TOOLS | MARKET_AGENT_TOOLS
                    | NEWS_AGENT_TOOLS | SHARED_TOOLS),
}

# The dataset categories that must reach no agent at all.
NO_TOOL_CATEGORIES = frozenset({"reject", "guard", "declined"})


@dataclass(frozen=True)
class Trajectory:
    scored: int
    required_met: int
    lane_violations: List[Tuple[int, str]] = field(default_factory=list)
    missing_required: List[Tuple[int, str]] = field(default_factory=list)
    zero_tool_expected: int = 0
    zero_tool_actual: int = 0
    zero_tool_breaches: List[int] = field(default_factory=list)
    not_executed: int = 0
    scorer: str = "inline"

    @property
    def correctness(self) -> float:
        return self.required_met / self.scored if self.scored else 1.0


def _score_required(called: List[str], required: FrozenSet[str]) -> bool:
    """Was at least one required tool called. The only DeepEval touchpoint."""
    if not required:
        return True
    try:
        from deepeval.metrics import ToolCorrectnessMetric
        from deepeval.test_case import LLMTestCase, ToolCall
    except ImportError:
        return bool(set(called) & required)

    # DeepEval asks whether every expected tool was called, so one representative
    # is passed: the strategies here require *one of* a family, not all of it.
    hit = next((t for t in called if t in required), None)
    case = LLMTestCase(
        input="", actual_output="",
        tools_called=[ToolCall(name=t) for t in called] or [ToolCall(name="none")],
        expected_tools=[ToolCall(name=hit or sorted(required)[0])],
    )
    metric = ToolCorrectnessMetric()
    metric.measure(case)
    return bool(metric.score)


def _strategy_of(row: dict) -> Optional[str]:
    """The routing decision the pipeline actually made for this row.

    From the recorded parse, never from the dataset category. Category is
    authored intent and is not a reliable proxy: two `a2a` rows -- "Tesla
    disclosed a legal settlement in its most recent annual report" -- parse as
    sec claims about what a filing says and route to `filing_rag`, not
    `news_search`. Scoring them by category marked six correct runs as
    failures.

    A row with no recorded parse is skipped rather than guessed at.
    """
    from ...config.metrics import verification_strategy_for

    parsed = (row.get("actual") or {}).get("parsed_claim")
    if not isinstance(parsed, dict) or not parsed.get("claim_type"):
        return None
    claim = type("ParsedView", (), {
        "claim_type": parsed.get("claim_type"),
        "metric": parsed.get("metric"),
    })()
    return verification_strategy_for(claim)


def measure(runs: List[Run]) -> Trajectory:
    """Score every row that recorded its tool calls."""
    scored = met = zero_expected = zero_actual = not_executed = 0
    lane: List[Tuple[int, str]] = []
    missing: List[Tuple[int, str]] = []
    breaches: List[int] = []
    scorer = "inline"
    try:  # noqa: SIM105
        import deepeval  # noqa: F401
        scorer = "deepeval"
    except ImportError:
        pass

    for run in runs:
        label = run.label or run.started_utc
        for row_id, row in run.rows.items():
            actual = row.get("actual") or {}
            if "tools_called" not in actual:
                continue
            called = list(actual.get("tools_called") or [])

            if row.get("category") in NO_TOOL_CATEGORIES:
                zero_expected += 1
                # No request_id means an input guardrail refused before any
                # execution existed. Zero tools there is true by construction,
                # not by restraint, so it is counted apart.
                if not row.get("request_id"):
                    not_executed += 1
                elif called:
                    breaches.append(row_id)
                else:
                    zero_actual += 1

            # A claim declined before any agent ran has no path to score.
            # `_unsupported_claim` refuses Q4 derivations, unservable metrics
            # and unnamed companies up front and states why, so calling nothing
            # is the correct trajectory -- the strategy the parse implies was
            # never entered. A Q4 claim still parses as sec/revenue and so
            # implies `xbrl`, which is why scoring it by strategy alone marked
            # eight correct declines as failures.
            #
            # The zero-tool accounting above has already run, so nothing is
            # lost by skipping here.
            if actual.get("limitation"):
                continue

            strategy = _strategy_of(row)
            if strategy is None or strategy not in EXPECTED:
                continue
            required, forbidden = EXPECTED[strategy]

            out_of_lane = sorted(set(called) & forbidden - SHARED_TOOLS)
            if out_of_lane:
                lane.append((row_id, f"{label}: {', '.join(out_of_lane)}"))

            if not required:
                continue
            scored += 1
            if _score_required(called, required):
                met += 1
            else:
                missing.append((row_id, f"{label}: {strategy} needs one of "
                                        f"{sorted(required)}, called {called}"))

    return Trajectory(
        scored=scored, required_met=met, lane_violations=sorted(lane),
        missing_required=sorted(missing), zero_tool_expected=zero_expected,
        zero_tool_actual=zero_actual, zero_tool_breaches=sorted(set(breaches)),
        not_executed=not_executed, scorer=scorer,
    )

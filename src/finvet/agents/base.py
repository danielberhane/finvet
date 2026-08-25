"""Base ReAct agent implementation using LangGraph create_react_agent.

This module provides the foundation for domain-specific verification agents.
Each agent uses LangGraph's built-in ReAct pattern where the LLM reasons
about which tool to call, observes the result, and continues until it can
make a verification decision.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from ..config.constants import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_RESULT_CHARS,
    TOOL_RESULT_PREVIEW_CHARS,
    TOLERANCE_APPROX_MULTIPLIER,
    TOLERANCE_DEFAULT,
    TOLERANCE_LARGE_VALUE_THRESHOLD,
    TOLERANCE_MARKET,
    TOLERANCE_NEWS,
    TOLERANCE_SEC_LARGE,
    TOLERANCE_SEC_SMALL,
)
from ..config.settings import settings
from ..llm import create_llm
from ..models.evidence import (
    ToolExecutionRecord,
    TrustedObservation,
    resolve_trusted_observation,
)
from ..models.state import VerificationState
from ..utils.logging import get_logger

logger = get_logger(__name__)


def compose_failure_reasoning(error_msg: str) -> str:
    """Reviewer-facing summary of an agent crash. The raw error is preserved
    in the audit trail's node_error event; the HITL screen gets prose, not a
    framework exception with a docs link."""
    first_line = (error_msg or "").strip().splitlines()[0:1]
    first = first_line[0] if first_line else "no error detail was captured"
    if "Recursion limit" in first:
        return (
            "The agent used up its tool-call budget without retrieving the "
            "claimed figure from its data sources, and stopped rather than "
            "guess. Escalated for human review; the full error is preserved "
            "in the audit trail."
        )
    return (
        f"The agent stopped before reaching a verdict: {first[:160]}. "
        "Escalated for human review; the full error is preserved in the "
        "audit trail."
    )


class VerdictOutput(BaseModel):
    """Structured verdict output from the LLM."""
    verdict: Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    retrieved_value: Optional[float] = None
    source_description: str = ""


class BaseVerificationAgent(ABC):
    """
    Base class for ReAct verification agents.

    Each domain agent (SEC, Market, News) extends this class with:
    - Its specific system prompt
    - Its set of tools
    - Its agent type identifier
    """

    # Subclasses can override to capture full (non-truncated) results from
    # specific tools for provenance tracking. The SEC agent uses it for RAG
    # (search_filing_text); the News agent uses it for its delegation tool
    # (corroborate_with_filing). The Market agent leaves it empty.
    _provenance_tool_names: set = set()

    def __init__(
        self,
        agent_type: str,
        tools: List[BaseTool],
        system_prompt: str,
        max_iterations: int = AGENT_MAX_ITERATIONS,
    ):
        """
        Initialize the verification agent.

        Args:
            agent_type: Identifier for this agent ("sec", "market", "news")
            tools: List of LangChain tools available to this agent
            system_prompt: System prompt defining the agent's role
            max_iterations: Maximum ReAct loops before giving up
        """
        self.agent_type = agent_type
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # Create tool name -> tool mapping for provenance lookup
        self.tool_map = {tool.name: tool for tool in tools}

        # LLM (model configurable via LLM_AGENT__MODEL env var)
        self.llm = create_llm("agent")

        # LangGraph ReAct agent handles the reasoning + tool calling loop
        self.react_agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt,
        )

    def execute(self, state: VerificationState) -> Dict[str, Any]:
        """Execute verification using LangGraph's create_react_agent."""
        start_time = time.time()
        context = self._build_context(state)

        # Run LangGraph react agent (handles ReAct loop internally)
        try:
            agent_result = self.react_agent.invoke(
                {"messages": [HumanMessage(content=context)]},
                config={"recursion_limit": self.max_iterations * 2 + 1},
            )
            messages = agent_result["messages"]
        except Exception as e:
            logger.error(f"{self.agent_type} react agent failed: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._error_evidence(str(e), execution_time_ms)

        # Extract tools called and provenance from messages
        tools_called, tool_calls_detail, provenance, tool_records = \
            self._extract_tool_info(messages)

        # Extract verdict (separate LLM call — anti-anchoring)
        try:
            verdict_output = self._extract_verdict(messages, state)
        except Exception as e:
            logger.error(f"{self.agent_type} verdict extraction failed: {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._error_evidence(
                str(e), execution_time_ms, tools_called, tool_calls_detail
            )

        # The one number the deterministic layer is allowed to compare. Bound
        # to the resolved period so a correct figure from the wrong fiscal year
        # cannot satisfy the claim -- the comparator has no way to notice.
        canonical_period = state.get("canonical_period")
        observation = resolve_trusted_observation(
            state.get("parsed_claim"),
            tool_records,
            expected_period_end=getattr(canonical_period, "end_date", None),
        )

        # Python-based verdict override (deterministic number comparison)
        original_verdict = verdict_output.verdict
        try:
            verdict, confidence, magnitude_diff = self._apply_override(
                verdict_output, state, observation
            )
        except Exception as e:
            logger.error(
                f"{self.agent_type} _apply_override failed: {e}", exc_info=True
            )
            verdict = verdict_output.verdict
            confidence = verdict_output.confidence
            magnitude_diff = None

        override_applied = verdict != original_verdict
        logger.info(
            f"Verdict: LLM={original_verdict} → final={verdict} "
            f"(override={override_applied})"
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return {
            "agent": self.agent_type,
            "verdict": verdict,
            "confidence": confidence,
            "retrieved_value": verdict_output.retrieved_value,
            "source_description": verdict_output.source_description or self._get_source_description(),
            "source_url": None,
            "magnitude_difference_percent": magnitude_diff,
            "tools_called": tools_called,
            "tool_calls_detail": tool_calls_detail,
            "provenance": provenance,
            "reasoning": verdict_output.reasoning,
            "execution_time_ms": execution_time_ms,
            "override_applied": override_applied,
            "llm_original_verdict": original_verdict,
            "execution_status": "completed",
            "error": None,
        }

    def _extract_tool_info(self, messages) -> tuple:
        """Extract tool call details, provenance, and evidence records.

        Also truncates oversized ToolMessages in-place to control token spend
        on subsequent LLM calls (verdict extraction).

        Returns (tools_called, tool_calls_detail, provenance, tool_records).
        The records carry the parsed payload and both notions of success; the
        detail dicts keep their existing shape because the API and UI read
        them.
        """
        tools_called = []
        tool_calls_detail = []
        provenance = []
        tool_records = []

        # Map tool_call_id -> call info for pairing with ToolMessages
        pending_calls = {}

        for msg in messages:
            # AIMessage with tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_called.append(tc["name"])
                    pending_calls[tc["id"]] = {
                        "tool": tc["name"],
                        "args": tc["args"],
                    }

            # ToolMessage — pair with its AIMessage tool_call
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                call_info = pending_calls.pop(getattr(msg, "tool_call_id", ""), {})
                tool_name = call_info.get("tool", getattr(msg, "name", "unknown"))

                # Capture structured provenance for tracked tools BEFORE truncation
                if tool_name in self._provenance_tool_names:
                    provenance.append({
                        "tool": tool_name,
                        "args": call_info.get("args", {}),
                        "result": self._parse_provenance(content),
                    })

                # Two different questions, previously conflated. LangChain's
                # status says whether the call completed; the tool's own
                # success field says whether it retrieved anything. Every SEC
                # and Market tool catches its exceptions and returns an error
                # payload, so a failed retrieval is a transport success -- and
                # was recorded as a successful call.
                payload = self._parse_provenance(content)
                transport_success = getattr(msg, "status", "success") != "error"
                application_success = payload.get("success")
                if not isinstance(application_success, bool):
                    application_success = None

                record = ToolExecutionRecord(
                    tool=tool_name,
                    args=call_info.get("args", {}),
                    payload=payload,
                    transport_success=transport_success,
                    application_success=application_success,
                    result_preview=content[:TOOL_RESULT_PREVIEW_CHARS],
                )
                tool_records.append(record)

                # Built by the record so the audit view and the evidence gate
                # cannot drift apart.
                tool_calls_detail.append(record.to_detail())

                # Truncate oversized tool results to control token spend
                # on the verdict extraction LLM call
                if len(content) > AGENT_MAX_RESULT_CHARS:
                    logger.warning(
                        f"Truncating {tool_name} result from {len(content)} "
                        f"to {AGENT_MAX_RESULT_CHARS} chars"
                    )
                    msg.content = content[:AGENT_MAX_RESULT_CHARS] + "\n... [TRUNCATED]"

        return tools_called, tool_calls_detail, provenance, tool_records

    @staticmethod
    def _parse_provenance(content: str) -> Dict[str, Any]:
        """Parse tool result into structured provenance data.

        Attempts JSON/dict parsing for richer audit trails. Falls back
        to raw string (capped at 5000 chars) if parsing fails.
        """
        import ast
        import json

        # Try JSON first (tool results that return JSON strings)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Try Python dict literal (tool results using repr())
        try:
            parsed = ast.literal_eval(content)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # Fallback: raw string, capped
        return {"raw": content[:5000]}

    def _extract_verdict(self, messages, state) -> VerdictOutput:
        """Separate LLM call for structured verdict (anti-anchoring).

        Excludes the agent's final free-text reasoning from the message
        history so the verdict LLM isn't biased by the agent's conclusion.
        """
        verdict_llm = create_llm("verdict").with_structured_output(
            VerdictOutput, method=settings.llm_verdict.structured_output_method
        )

        # Skip the final AI message if it's free-text reasoning (no tool calls)
        # to prevent anchoring bias on a potentially wrong verdict
        verdict_messages = []
        for msg in messages:
            if (msg == messages[-1]
                    and hasattr(msg, "tool_calls") and not msg.tool_calls
                    and hasattr(msg, "content") and msg.content):
                continue
            verdict_messages.append(msg)

        verdict_messages.append(HumanMessage(
            content=(
                "Based on the tool results above, provide your FINAL verdict as JSON.\n\n"
                "CRITICAL RULE: The verdict is determined ONLY by the numeric comparison. "
                "If the percentage difference between claimed and retrieved values is within "
                "the tolerance threshold, the verdict MUST be SUPPORTS — regardless of any "
                "other considerations (period naming, rounding, fiscal vs calendar year, etc.). "
                "Do NOT second-guess the period interpretation — that was already resolved.\n\n"
                "If the value requires derivation (e.g., Q4 = Annual - 9-month cumulative), "
                "perform that calculation from the data.\n\n"
                "Do NOT apply a tolerance yourself. When the claim states a "
                "number, Python recomputes the comparison against the value "
                "resolved from tool output and that result is authoritative; a "
                "threshold quoted here would only compete with it.\n\n"
                "Verdict:\n"
                "- The retrieved value matches the claim → SUPPORTS\n"
                "- The retrieved value contradicts the claim → REFUTES\n"
                "- Could not retrieve a reliable value → NOT_ENOUGH_INFO\n\n"
                "JSON fields:\n"
                "- verdict: \"SUPPORTS\", \"REFUTES\", or \"NOT_ENOUGH_INFO\"\n"
                "- confidence: float 0.0-1.0 (0.9+ exact/near-exact, 0.7-0.9 within tolerance, "
                "0.5-0.7 ambiguous, <0.5 weak)\n"
                "- reasoning: concise analysis (source, retrieved value, % difference, verdict)\n"
                "- retrieved_value: the exact numeric value you found in the data as a raw number "
                "(e.g., 350018000000.0 for $350B, 59200000000.0 for $59.2B). "
                "ALWAYS provide this when you retrieved a number. null ONLY if no number was found.\n"
                "- source_description: string describing the data source"
            )
        ))

        verdict_output = verdict_llm.invoke(verdict_messages)
        logger.info(
            f"{self.agent_type} LLM verdict: {verdict_output.verdict} "
            f"(confidence: {verdict_output.confidence:.2f})"
        )
        return verdict_output

    def _apply_override(
        self,
        verdict_output: VerdictOutput,
        state: VerificationState,
        observation: Optional[TrustedObservation],
    ) -> tuple:
        """Deterministic comparison against a trusted observation.

        LLMs are unreliable at number comparison ("0.98% > 1%"), which is why
        Python recomputes it. That only helps if the number Python receives is
        itself trustworthy: the model used to supply it, and a value it read
        out of narrative prose -- or out of an error string -- was compared
        with the same confidence as a figure lifted from an XBRL fact.

        So the observation is the only numeric input. A claim that names a
        value but produced no trusted observation fails closed: the honest
        answer is that nothing was verified, not that the model's reading
        passed a tolerance check.
        """
        verdict = verdict_output.verdict
        confidence = verdict_output.confidence

        parsed_claim = state.get("parsed_claim")
        claimed_val = parsed_claim.value if parsed_claim else None

        if claimed_val is not None and observation is None:
            logger.info(
                f"{self.agent_type} no trusted observation for a numeric claim; "
                f"failing closed to NOT_ENOUGH_INFO"
            )
            return "NOT_ENOUGH_INFO", min(confidence, 0.5), None

        retrieved_value = observation.value if observation else None
        # What the response shows must be what Python compared, not what the
        # model reported finding.
        verdict_output.retrieved_value = retrieved_value

        # Calculate magnitude difference
        magnitude_diff = None
        if claimed_val is not None and retrieved_value is not None:
            divisor = max(abs(claimed_val), abs(retrieved_value))
            magnitude_diff = abs(claimed_val - retrieved_value) / divisor * 100 if divisor > 0 else 0.0

        # Override verdict based on deterministic comparison
        if magnitude_diff is not None:
            comparison = getattr(parsed_claim, "operator", None) or "eq"
            tolerance = self._get_tolerance(claimed_val)

            # A range is an interval, tested for membership. It used to be
            # collapsed to its midpoint and compared as equality, which
            # refutes any value far from the centre however clearly it sits
            # inside the stated band: "between $50B and $150B" became $100B,
            # and a filed $149B scored a 32.89% difference and was REFUTED.
            if comparison == "range":
                lower = getattr(parsed_claim, "range_min", None)
                upper = getattr(parsed_claim, "range_max", None)
                if lower is None or upper is None:
                    # ParsedClaim forbids this; reachable only through drift.
                    logger.error(
                        f"{self.agent_type} range claim without bounds; "
                        f"failing closed to NOT_ENOUGH_INFO")
                    return "NOT_ENOUGH_INFO", min(confidence, 0.5), magnitude_diff

                # Tolerance applies to the edges, not the midpoint: a stated
                # band already expresses the claimant's imprecision, so only
                # rounding at the boundary needs slack.
                margin = tolerance / 100.0
                lo = lower - abs(lower) * margin
                hi = upper + abs(upper) * margin
                verdict = "SUPPORTS" if lo <= retrieved_value <= hi else "REFUTES"
                confidence = max(confidence, 0.90)
                logger.info(
                    f"{self.agent_type} range check: {retrieved_value:,.0f} "
                    f"in [{lower:,.0f}, {upper:,.0f}] -> {verdict}")
                return verdict, confidence, magnitude_diff

            # approx is equality with stated imprecision: no band exists, so
            # widen the tolerance by the measured multiplier.
            if comparison == "approx":
                tolerance *= TOLERANCE_APPROX_MULTIPLIER
                comparison = "eq"

            if comparison == "eq":
                if magnitude_diff <= tolerance:
                    if verdict != "SUPPORTS":
                        logger.info(
                            f"{self.agent_type} override: {verdict} → SUPPORTS "
                            f"(diff {magnitude_diff:.2f}% <= tolerance {tolerance}%)"
                        )
                    verdict = "SUPPORTS"
                    confidence = max(confidence, 0.90 if magnitude_diff < tolerance / 2 else 0.85)
                else:
                    if verdict != "REFUTES":
                        logger.info(
                            f"{self.agent_type} override: {verdict} → REFUTES "
                            f"(diff {magnitude_diff:.2f}% > tolerance {tolerance}%)"
                        )
                    verdict = "REFUTES"
                    confidence = max(confidence, 0.90)
            elif comparison in ("gt", "gte"):
                passes = (
                    (comparison == "gt" and retrieved_value > claimed_val) or
                    (comparison == "gte" and retrieved_value >= claimed_val)
                )
                new_verdict = "SUPPORTS" if passes else "REFUTES"
                if verdict != new_verdict:
                    logger.info(
                        f"{self.agent_type} override: {verdict} → {new_verdict} "
                        f"(comparison={comparison}, claimed={claimed_val}, actual={retrieved_value})"
                    )
                verdict = new_verdict
                confidence = max(confidence, 0.90)
            elif comparison in ("lt", "lte"):
                passes = (
                    (comparison == "lt" and retrieved_value < claimed_val) or
                    (comparison == "lte" and retrieved_value <= claimed_val)
                )
                new_verdict = "SUPPORTS" if passes else "REFUTES"
                if verdict != new_verdict:
                    logger.info(
                        f"{self.agent_type} override: {verdict} → {new_verdict} "
                        f"(comparison={comparison}, claimed={claimed_val}, actual={retrieved_value})"
                    )
                verdict = new_verdict
                confidence = max(confidence, 0.90)
            else:
                # Fail closed. An operator we cannot interpret means we hold
                # both numbers but no way to compare them — the deterministic
                # layer declines to verify rather than letting the LLM verdict
                # pass unchecked, which is what silently falling through every
                # branch used to do. Reachable only through schema drift.
                logger.error(
                    f"{self.agent_type} unknown operator {comparison!r}; "
                    f"failing closed to NOT_ENOUGH_INFO"
                )
                verdict = "NOT_ENOUGH_INFO"
                confidence = min(confidence, 0.5)

        logger.info(f"{self.agent_type} final verdict: {verdict} (confidence: {confidence:.2f})")
        return verdict, confidence, magnitude_diff

    def _error_evidence(
        self,
        error_msg: str,
        execution_time_ms: int,
        tools_called: Optional[List[str]] = None,
        tool_calls_detail: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build fallback evidence dict on failure."""
        return {
            "agent": self.agent_type,
            "verdict": "NOT_ENOUGH_INFO",
            "confidence": 0.2,
            "retrieved_value": None,
            "source_description": self._get_source_description(),
            "source_url": None,
            "magnitude_difference_percent": None,
            "tools_called": tools_called or [],
            "tool_calls_detail": tool_calls_detail or [],
            "provenance": [],
            "reasoning": compose_failure_reasoning(error_msg),
            "execution_time_ms": execution_time_ms,
            # Same shape as the success path — consumers read these keys
            # unconditionally.
            "override_applied": False,
            "llm_original_verdict": None,
            "execution_status": "failed",
            "error": error_msg,
        }

    def _build_context(self, state: VerificationState) -> str:
        """Build the context message for the agent from state."""
        claim_raw = state.get("claim_raw", "")
        parsed_claim = state.get("parsed_claim")
        canonical_period = state.get("canonical_period")
        company_info = state.get("company_info")

        context_parts = [
            f"# Claim to Verify\n{claim_raw}\n",
        ]

        if parsed_claim:
            context_parts.append("# Parsed Information")
            if parsed_claim.ticker:
                context_parts.append(f"- Ticker: {parsed_claim.ticker}")
            # The canonical metric, when the parser resolved one. Null stays
            # silent — the fall-through policy for absent and derived metrics
            # is that the agent infers from claim text, exactly the
            # pre-migration behaviour.
            if getattr(parsed_claim, "metric", None):
                context_parts.append(f"- Metric: {parsed_claim.metric}")
            if parsed_claim.value is not None:
                context_parts.append(f"- Claimed Value: {parsed_claim.value:,.0f}")
            if parsed_claim.period:
                context_parts.append(f"- Period: {parsed_claim.period}")
            context_parts.append("")

        if canonical_period:
            context_parts.append("# Resolved Period")
            context_parts.append(f"- Type: {canonical_period.period_type}")
            context_parts.append(f"- Start: {canonical_period.start_date}")
            context_parts.append(f"- End: {canonical_period.end_date}")
            if canonical_period.fiscal_year:
                context_parts.append(f"- Fiscal Year: {canonical_period.fiscal_year}")
            if canonical_period.fiscal_quarter:
                context_parts.append(f"- Fiscal Quarter: {canonical_period.fiscal_quarter}")
            context_parts.append("")

        if company_info:
            context_parts.append("# Company Information")
            context_parts.append(f"- CIK: {company_info.cik}")
            context_parts.append(f"- Name: {company_info.name}")
            context_parts.append(f"- Fiscal Year End: {company_info.fiscal_year_end}")
            context_parts.append("")

        memory_context = state.get("memory_context")
        if memory_context:
            context_parts.append("# Prior Verification (from memory)")
            context_parts.append(f"- Similar claim: {memory_context.get('claim', '')}")
            context_parts.append(f"- Verdict: {memory_context.get('verdict', '')}")
            context_parts.append(f"- Confidence: {memory_context.get('confidence', 0):.0%}")
            context_parts.append(f"- Similarity: {memory_context.get('similarity', 0):.0%}")
            if memory_context.get("summary"):
                context_parts.append(f"- Summary: {memory_context['summary']}")
            context_parts.append("")
            context_parts.append(
                "NOTE: This prior verification is provided as context only. "
                "You MUST verify independently using current data. "
                "Do not blindly trust the prior result — it may be outdated."
            )
            context_parts.append("")

        context_parts.append(
            "# Your Task\n"
            "Use the available tools to retrieve evidence and determine if the claim is "
            "SUPPORTS (claim matches evidence within tolerance), "
            "REFUTES (claim contradicts evidence), or "
            "NOT_ENOUGH_INFO (insufficient evidence to verify).\n\n"
            "After gathering evidence, provide your final verdict with reasoning."
        )

        return "\n".join(context_parts)

    def _get_tolerance(self, claimed_value: Optional[float]) -> float:
        """Get the tolerance threshold based on agent type and value magnitude."""
        if self.agent_type == "market":
            return TOLERANCE_MARKET
        elif self.agent_type == "sec":
            if claimed_value and abs(claimed_value) >= TOLERANCE_LARGE_VALUE_THRESHOLD:
                return TOLERANCE_SEC_LARGE
            return TOLERANCE_SEC_SMALL
        elif self.agent_type == "news":
            return TOLERANCE_NEWS
        return TOLERANCE_DEFAULT

    @abstractmethod
    def _get_source_description(self) -> str:
        """Return description of the data source for this agent."""
        pass

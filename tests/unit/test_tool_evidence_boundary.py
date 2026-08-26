"""The boundary between tool execution and numeric verdicts.

The deterministic override is FinVet's headline guarantee, and it is only as
good as the number it is handed. Two defects sit under it:

1. Every SEC and Market tool catches its exceptions and *returns* a result with
   success=False rather than raising. LangChain therefore reports the call's
   transport status as success, and _extract_tool_info records the failed call
   as successful.
2. The retrieved-value fallback regex-scraped numbers out of the serialized
   result string, truncated to TOOL_RESULT_PREVIEW_CHARS -- so what the
   comparator received depended on where a string was cut. It now reads
   structured fields from ToolExecutionRecord instead.

These tests drive the real decorated tools. An earlier test in
test_base_agent.py hand-built ToolMessage(status="error"), a shape production
never emits, which is why it passed while the defect shipped.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from finvet.agents.base import BaseVerificationAgent
from finvet.mcp.sec_edgar import FinancialItem
from finvet.models.evidence import resolve_trusted_observation
from finvet.tools.sec_tools import get_income_statement


class _Claim:
    """Minimal ParsedClaim stand-in for metric resolution."""
    metric = "revenue"
    value = 150_000_000_000.0
    operator = "eq"


def _records_for(result):
    """Run a tool result through the real extraction, as the agent does."""
    call = {"name": "get_income_statement", "args": {}, "id": "call_1"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=str(result), tool_call_id="call_1",
                    name="get_income_statement"),
    ]
    _, detail, _, records = BaseVerificationAgent._extract_tool_info(
        BaseVerificationAgent, messages)
    return detail, records


def _invoke_failing(message):
    with patch("finvet.tools.sec_tools._sec_client", None), \
         patch("finvet.tools.sec_tools._get_client",
               side_effect=Exception(message)):
        return get_income_statement.invoke(
            {"cik": "1", "accession_number": "acc", "period": "annual"})


def _invoke_succeeding(items):
    with patch("finvet.tools.sec_tools._get_client") as client:
        client.return_value.get_financials.return_value = items
        return get_income_statement.invoke(
            {"cik": "1", "accession_number": "acc", "period": "annual"})


# ---------------------------------------------------------------------------
# Application-level failure must not be recorded as a successful call
# ---------------------------------------------------------------------------

class TestApplicationFailureIsNotSuccess:

    @pytest.mark.parametrize("message", [
        "Connection refused to MCP server at localhost:9870",
        "1 validation error for FinancialsResult items.0.value Input should be "
        "a valid number [input_value=[{'line_item': 'Revenues', "
        "'value': 150000000000.0}]]",
    ])
    def test_caught_failure_is_recorded_as_failed(self, message):
        """The tool reports success=False; the audit trail must agree.

        This holds for every caught failure, not only ones a number can be
        scraped from -- a failed call recorded as successful corrupts the
        record regardless of what happens downstream.
        """
        result = _invoke_failing(message)
        assert result["success"] is False, "precondition: the tool caught the error"

        detail, _ = _records_for(result)
        assert detail[0]["success"] is False

    def test_successful_call_is_still_recorded_as_successful(self):
        """The guard must not swing the other way."""
        result = _invoke_succeeding([
            FinancialItem(line_item="Revenues", concept="Revenues",
                          value=391_035_000_000.0, units="USD",
                          period="annual", period_end="2024-09-28"),
        ])
        assert _records_for(result)[0][0]["success"] is True


# ---------------------------------------------------------------------------
# No number may be taken from a failed call
# ---------------------------------------------------------------------------

class TestNoValueFromFailedCall:

    def test_error_text_carrying_a_line_item_yields_no_value(self):
        """A parse error that echoes the payload it choked on puts a
        financial-looking dict into the error string. Scraping it produced
        150,000,000,000 from a call that retrieved nothing, which the override
        then promoted to SUPPORTS at 0.90.
        """
        result = _invoke_failing(
            "1 validation error for FinancialsResult items.0.value Input "
            "should be a valid number [input_value=[{'line_item': 'Revenues', "
            "'value': 150000000000.0}]]")

        _, records = _records_for(result)
        observation = resolve_trusted_observation(_Claim(), records)

        assert observation is None


# ---------------------------------------------------------------------------
# Evidence must not depend on where the preview string was cut
# ---------------------------------------------------------------------------

class TestEvidenceSurvivesTruncation:

    def test_value_is_found_beyond_the_preview_window(self):
        """TOOL_RESULT_PREVIEW_CHARS truncates the serialized result at 3000
        characters, and the fallback scrapes that truncated copy. A filing with
        enough line items pushes the relevant one past the cut, and the value
        silently disappears.
        """
        items = [
            FinancialItem(line_item=f"Filler{i}", concept="F", value=1234.0,
                          units="USD", period="annual", period_end="2024-09-28")
            for i in range(30)
        ]
        items.append(FinancialItem(
            line_item="Revenues", concept="Revenues", value=391_035_000_000.0,
            units="USD", period="annual", period_end="2024-09-28"))

        result = _invoke_succeeding(items)
        assert len(str(result)) > 3000, "precondition: result exceeds the preview"

        _, records = _records_for(result)
        observation = resolve_trusted_observation(_Claim(), records)

        assert observation is not None, "value lost past the 3000-char preview"
        assert observation.value == 391_035_000_000.0
        assert observation.concept == "Revenues"


class TestFailClosedScopeIsRecorded:
    """What failing closed costs, pinned so the cost stays visible.

    Task 2 makes a trusted observation the only numeric input, and a trusted
    observation can only come from a structured field: an XBRL line item, a
    market quote field, or a FRED series. All 35 metrics in SERVABLE_METRICS
    have one.

    The parser's whitelist is wider than SERVABLE_METRICS, and 39 of the 74
    metrics it may emit have no structured source at all. Numeric claims on
    those now return NOT_ENOUGH_INFO instead of a verdict derived from a number
    the model read out of prose. That is the intended direction -- but it is a
    real reduction in what the system will answer, and two of the affected
    metrics are the A2A corroboration metrics, which has a consequence recorded
    below.

    OPEN DECISION for the release gate: a claim whose metric has no structured
    source is not verifiable, and the settled split says the parser owns
    verifiability rejections. Rejecting these at parse time would be more
    honest than answering NOT_ENOUGH_INFO from the agent. Not done here --
    it is Task 3/Task 10 scope.
    """

    def test_every_servable_metric_still_has_a_structured_source(self):
        from finvet.config.metrics import METRIC_TO_CONCEPTS, SERVABLE_METRICS
        from finvet.mcp.fred import FRED_SERIES
        from finvet.models.evidence import _MARKET_FIELD_FOR_METRIC

        resolvable = (set(METRIC_TO_CONCEPTS) | set(_MARKET_FIELD_FOR_METRIC)
                      | set(FRED_SERIES))
        for kind in ("sec", "market", "news"):
            missing = SERVABLE_METRICS[kind] - resolvable
            assert not missing, f"{kind} metrics with no structured source: {missing}"

    def test_corroboration_metrics_have_no_structured_source(self):
        """What the delegation can and cannot establish for these metrics.

        fine_amount and settlement_amount are the only metrics the News -> SEC
        delegation acts on, and a fine is a narrative fact -- there is no XBRL
        concept for a penalty. So neither side can produce a trusted
        observation, both verdicts are NOT_ENOUGH_INFO, and CONTRADICTS (which
        needs two decisive verdicts) is unreachable for exactly the claims the
        delegation exists to check.

        An earlier release answered that by escalating on absence:
        UNDISCLOSED_MATERIAL_CLAIM fired when a filing covering the period did
        not mention the amount. That required deciding the issuer should have
        disclosed it -- a materiality judgment with nothing calibrating it --
        so Release A removed the status rather than leave an uncalibrated
        escalation in the product.

        What the delegation still yields for these claims is an attributable
        record of what the filing did or did not say. That is worth keeping and
        is not an escalation. Asserted here so the trade-off cannot drift: if
        someone gives these metrics a structured source, the first assertion
        fails and disagreement becomes reachable again.
        """
        from finvet.config.constants import CORROBORATION_METRICS
        from finvet.config.metrics import METRIC_TO_CONCEPTS
        from finvet.mcp.fred import FRED_SERIES
        from finvet.models.a2a import (
            A2A_CONTRADICTS,
            A2A_NO_MATCHING_DISCLOSURE,
            A2A_PENDING_CLASSIFICATION,
            A2AResult,
            classify_status,
            reclassify_corroboration,
        )
        from finvet.models.evidence import _MARKET_FIELD_FOR_METRIC

        resolvable = (set(METRIC_TO_CONCEPTS) | set(_MARKET_FIELD_FOR_METRIC)
                      | set(FRED_SERIES))
        assert not (CORROBORATION_METRICS & resolvable)

        for target in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            assert classify_status("NOT_ENOUGH_INFO", target) != A2A_CONTRADICTS

        # A filing that was read and said nothing is recorded as silence, and
        # silence does not escalate.
        silent = A2AResult(
            success=True, source_agent="news", target_agent="sec",
            status=A2A_PENDING_CLASSIFICATION, verdict="NOT_ENOUGH_INFO",
            metric="fine_amount", claimed_value=5e8,
            temporal_scope="claim_period",
            provenance=[{"tool": "search_filing_text",
                         "args": {"query": "fine", "ticker": "AAPL"},
                         "result": {"success": True, "chunks": [],
                                    "total_found": 0,
                                    "reason": "no_relevant_evidence"}}],
        ).model_dump()
        assert reclassify_corroboration("NOT_ENOUGH_INFO", silent)["status"] \
            == A2A_NO_MATCHING_DISCLOSURE


class TestComparatorFailureFailsClosed:
    """Spec invariant 2: a comparator failure cannot release a decisive verdict.

    _apply_override is wrapped in a try/except that, on failure, restores the
    verdict and confidence the model produced. That is the deterministic layer
    failing *open*: the one component whose job is to overrule the model hands
    control back to it precisely when it breaks.

    Driven through the real execute() path -- only the LLM boundaries are
    stubbed -- so it fails if the production lifecycle changes.
    """

    def _run_with_broken_comparator(self, monkeypatch):
        from finvet.agents.base import BaseVerificationAgent, VerdictOutput
        from finvet.models.claim import ParsedClaim
        from tests.unit.test_base_agent import ConcreteAgent

        agent = ConcreteAgent(agent_type="sec")
        parsed = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                             operator="eq", value=391_035_000_000.0)

        react = MagicMock()
        react.invoke.return_value = {"messages": [AIMessage(content="done",
                                                            tool_calls=[])]}
        # ConcreteAgent skips __init__ to avoid building an LLM, so the react
        # agent is attached here rather than replaced.
        agent.react_agent = react
        monkeypatch.setattr(
            BaseVerificationAgent, "_extract_verdict",
            lambda self, m, s: VerdictOutput(
                verdict="SUPPORTS", confidence=0.95,
                reasoning="the model was confident",
                retrieved_value=391_035_000_000.0))
        monkeypatch.setattr(
            BaseVerificationAgent, "_apply_override",
            lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("comparator blew up")))

        return agent.execute({"parsed_claim": parsed, "claim_raw": "c",
                              "request_id": "t"})

    def test_verdict_is_not_enough_info(self, monkeypatch):
        assert self._run_with_broken_comparator(monkeypatch)["verdict"] == \
            "NOT_ENOUGH_INFO"

    def test_confidence_is_at_or_below_the_hitl_threshold(self, monkeypatch):
        from finvet.config.settings import settings

        evidence = self._run_with_broken_comparator(monkeypatch)
        assert evidence["confidence"] <= settings.confidence_threshold_hitl

    def test_failure_is_typed_in_the_evidence(self, monkeypatch):
        evidence = self._run_with_broken_comparator(monkeypatch)
        assert evidence["execution_status"] == "failed"
        assert "comparator blew up" in (evidence.get("error") or "")


class TestPeriodBoundClaimsRequireAPeriod:
    """Spec invariant 2: matched to the requested period where the source
    provides one.

    The guard read `expected and observed and expected != observed`, so a fact
    carrying no period at all sailed through -- the check only rejected
    evidence that was already labelled well enough to be checked.
    """

    def _resolve(self, item, expected="2024-09-28"):
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import (
            ToolExecutionRecord,
            resolve_trusted_observation,
        )

        record = ToolExecutionRecord(
            tool="get_income_statement", transport_success=True,
            application_success=True,
            payload={"success": True, "items": [item]})
        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                            operator="eq", value=391_035_000_000.0)
        return resolve_trusted_observation(claim, [record],
                                           expected_period_end=expected)

    BASE = {"line_item": "Revenues", "value": 391_035_000_000.0, "units": "USD"}

    def test_missing_period_is_rejected_when_the_claim_is_period_bound(self):
        assert self._resolve(dict(self.BASE)) is None

    def test_matching_period_is_accepted(self):
        obs = self._resolve({**self.BASE, "period_end": "2024-09-28"})
        assert obs is not None and obs.period_end == "2024-09-28"

    def test_wrong_period_is_rejected(self):
        assert self._resolve({**self.BASE, "period_end": "2023-09-30"}) is None

    def test_missing_period_is_fine_when_the_claim_is_not_period_bound(self):
        obs = self._resolve(dict(self.BASE), expected=None)
        assert obs is not None


class TestUnverifiedFactsAreNotEvidence:
    """A fact the issuer did not report entity-wide is not a verified fact.

    sec_edgar tracks `consolidated` per fact precisely because a segment or
    subsidiary figure is not the company-level number a claim asks about.
    _format_financial_items dropped it, so the distinction never reached the
    trust boundary.
    """

    def test_consolidated_survives_formatting(self):
        from finvet.mcp.sec_edgar import FinancialItem
        from finvet.tools.sec_tools import _format_financial_items

        item = FinancialItem(line_item="Revenues", concept="Revenues",
                             value=1.0, units="USD", period="annual",
                             period_end="2024-09-28", consolidated=False)
        assert "consolidated" in _format_financial_items([item])[0]

    def test_an_unconsolidated_fact_is_not_trusted(self):
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import (
            ToolExecutionRecord,
            resolve_trusted_observation,
        )

        record = ToolExecutionRecord(
            tool="get_income_statement", transport_success=True,
            application_success=True,
            payload={"success": True, "items": [{
                "line_item": "Revenues", "value": 391_035_000_000.0,
                "units": "USD", "period_end": "2024-09-28",
                "consolidated": False}]})
        claim = ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                            operator="eq", value=391_035_000_000.0)

        assert resolve_trusted_observation(
            claim, [record], expected_period_end="2024-09-28") is None


class TestObservationIdentitySurvives:
    """Spec invariant 6: the TrustedObservation's identity reaches the audit
    record.

    The agent reduced the observation to `verdict_output.retrieved_value` -- a
    bare float taken from the model's own summary of what it read. Downstream,
    nothing could distinguish a number Python verified against a source from a
    number the model asserted, which is the entire distinction the trust
    boundary exists to draw.
    """

    OBSERVATION = {
        "tool": "get_income_statement", "metric": "revenue",
        "value": 391_035_000_000.0, "units": "USD",
        "period_end": "2024-09-28", "concept": "RevenueFromContract",
    }

    def _evidence(self):
        """Drive the real execute() path with a real resolved observation."""
        from unittest.mock import MagicMock, patch

        from finvet.agents.base import VerdictOutput
        from finvet.agents.sec_agent.react_agent import SECAgent
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import TrustedObservation

        agent = SECAgent.__new__(SECAgent)
        agent.agent_type = "sec"
        agent.max_iterations = 3
        agent.react_agent = MagicMock()
        agent.react_agent.invoke.return_value = {"messages": []}

        state = {"parsed_claim": ParsedClaim(
            claim_type="sec", ticker="AAPL", metric="revenue",
            operator="eq", value=391_035_000_000.0)}

        verdict_output = VerdictOutput(
            verdict="SUPPORTS", confidence=0.9,
            # The model's own number, deliberately different from the
            # observation: the evidence must carry what was verified.
            retrieved_value=999.0,
            reasoning="filed revenue matches", source_description="10-K")

        with patch.object(SECAgent, "_extract_verdict",
                          return_value=verdict_output), \
             patch("finvet.agents.base.resolve_trusted_observation",
                   return_value=TrustedObservation(**self.OBSERVATION)):
            return agent.execute(state)

    def test_the_agent_carries_the_observation_not_just_a_number(self):
        evidence = self._evidence()
        assert evidence.get("trusted_observation") is not None, (
            "the verified observation never left the agent")
        assert evidence["trusted_observation"]["concept"] == "RevenueFromContract"
        assert evidence["trusted_observation"]["tool"] == "get_income_statement"
        assert evidence["trusted_observation"]["period_end"] == "2024-09-28"

    def test_the_compared_value_is_the_verified_one(self):
        """Not the model's retrieved_value."""
        assert self._evidence()["retrieved_value"] == 391_035_000_000.0

    def test_it_reaches_the_response_metadata_and_the_audit_envelope(self):
        from finvet.graph.nodes.response_generator import response_generator
        from finvet.models.claim import ParsedClaim

        evidence = self._evidence()
        result = response_generator({
            "request_id": "req_000000000001",
            "claim_raw": "Apple FY2024 revenue was $391 billion",
            "parsed_claim": ParsedClaim(
                claim_type="sec", ticker="AAPL", metric="revenue",
                operator="eq", value=391_035_000_000.0),
            "agent_evidence": evidence,
            "final_verdict": "SUPPORTS",
            "final_confidence": 0.9,
            "execution_start_time": "2026-08-25T00:00:00",
        })

        metadata = result["final_response"]["metadata"]
        assert metadata.get("trusted_observation") is not None, (
            "the audited response cannot show what was actually verified")
        assert metadata["trusted_observation"]["concept"] == "RevenueFromContract"

    def test_the_provenance_column_records_what_was_verified(self):
        """data_sources is its own audit column; naming the tools that ran
        does not show what the comparison rested on."""
        from finvet.graph.nodes.response_generator import response_generator
        from finvet.models.claim import ParsedClaim

        evidence = self._evidence()
        evidence["tools_called"] = ["get_income_statement"]
        result = response_generator({
            "request_id": "req_000000000002",
            "claim_raw": "Apple FY2024 revenue was $391 billion",
            "parsed_claim": ParsedClaim(
                claim_type="sec", ticker="AAPL", metric="revenue",
                operator="eq", value=391_035_000_000.0),
            "agent_evidence": evidence,
            "final_verdict": "SUPPORTS",
            "final_confidence": 0.9,
            "execution_start_time": "2026-08-25T00:00:00",
        })

        xbrl = result["final_response"]["metadata"]["data_sources"]["xbrl"]
        assert xbrl.get("observation", {}).get("concept") == "RevenueFromContract"

    def test_the_evidence_contract_declares_it(self):
        from finvet.models.state import AgentEvidence

        assert "trusted_observation" in AgentEvidence.__annotations__


class TestDatedClaimsWithNoResolvedPeriodFailClosed:
    """A period-bound claim on a route that resolves no period.

    Only the SEC route runs `period_resolver`, so `canonical_period` is None
    for market and news claims and `expected_period_end` was always None there.
    A claim naming a specific day -- "AAPL closed at $150 on 2024-03-15" --
    was therefore compared against whatever the quote tool returned for
    *today*, with nothing checking the two referred to the same date. The
    comparison looked deterministic and was answering a different question.

    Release A does not implement per-source temporal matching (trading-day
    alignment, quote freshness windows), so the honest outcome is to decline
    the numeric comparison rather than issue a decisive verdict on it.
    """

    QUOTE = {"tool": "get_stock_quote", "metric": "closing_price",
             "value": 150.0, "period_end": "2026-08-24", "source_id": "AAPL"}

    def _evidence(self, *, claim_period, canonical_period=None):
        from unittest.mock import MagicMock, patch

        from finvet.agents.base import VerdictOutput
        from finvet.agents.market_agent.react_agent import MarketAgent
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import TrustedObservation

        agent = MarketAgent.__new__(MarketAgent)
        agent.agent_type = "market"
        agent.max_iterations = 3
        agent.react_agent = MagicMock()
        agent.react_agent.invoke.return_value = {"messages": []}

        state = {"parsed_claim": ParsedClaim(
            claim_type="market", ticker="AAPL", metric="closing_price",
            operator="eq", value=150.0, period=claim_period)}
        if canonical_period is not None:
            state["canonical_period"] = canonical_period

        verdict_output = VerdictOutput(
            verdict="SUPPORTS", confidence=0.95, retrieved_value=150.0,
            reasoning="quote matches", source_description="quote")

        with patch.object(MarketAgent, "_extract_verdict",
                          return_value=verdict_output), \
             patch("finvet.agents.base.resolve_trusted_observation",
                   return_value=TrustedObservation(**self.QUOTE)):
            return agent.execute(state)

    def test_a_dated_claim_does_not_get_a_decisive_verdict(self):
        evidence = self._evidence(claim_period="2024-03-15")
        assert evidence["verdict"] == "NOT_ENOUGH_INFO", (
            "a dated market claim was decided against an undated quote")

    def test_the_reason_is_recorded(self):
        evidence = self._evidence(claim_period="2024-03-15")
        assert evidence.get("temporal_status") == "unresolved_period"

    def test_an_undated_claim_is_unaffected(self):
        """"What is AAPL trading at" has no period to align."""
        evidence = self._evidence(claim_period=None)
        assert evidence["verdict"] == "SUPPORTS"
        assert evidence.get("temporal_status") == "not_period_bound"

    def test_a_resolved_period_is_unaffected(self):
        """The SEC route resolves one; the existing period check applies."""
        from finvet.models.claim import CanonicalPeriod

        evidence = self._evidence(
            claim_period="FY2024",
            canonical_period=CanonicalPeriod(
                period_type="annual", start_date="2023-10-01",
                end_date="2024-09-28"))
        assert evidence["verdict"] == "SUPPORTS"
        assert evidence.get("temporal_status") == "resolved"


class TestFiscalCalendarsAreNotCalendarYears:
    """A fiscal year labelled 2024 does not have to end on 2024-12-31.

    `period_resolver` turns "fiscal year 2024" into 2024-01-01..2024-12-31,
    because it runs before any agent has asked SEC what the issuer's calendar
    is. Apple's fiscal 2024 ended 2024-09-28; Microsoft's fiscal 2025 ended
    2025-06-30; Nvidia's fiscal 2025 ended 2025-01-26.

    Comparing the filing's period_end to that calendar approximation by string
    equality rejected the correct fact, so a numeric claim resolved no trusted
    observation and the comparator failed closed. The claim in the README --
    Apple's FY2024 revenue -- came back NOT_ENOUGH_INFO with the right number
    on screen, and went to a human reviewer.

    The question the guard should ask is not "does this date equal the one we
    guessed" but "does this fact's period fall inside the period the claim is
    about". Membership answers it; equality cannot, because only the issuer
    knows where its year ends.
    """

    def _claim(self, period="fiscal year 2024"):
        from finvet.models.claim import ParsedClaim

        return ParsedClaim(claim_type="sec", ticker="AAPL", metric="revenue",
                           operator="eq", value=391_000_000_000.0, period=period)

    def _fact(self, period_end):
        from finvet.models.evidence import tool_record_from_result

        return tool_record_from_result("get_income_statement", {}, {
            "success": True,
            "items": [{"line_item": "Revenues", "concept": "Revenues",
                       "value": 391_035_000_000.0, "units": "USD",
                       "period_end": period_end}]})

    def _resolve(self, period_end, *, start, end):
        from finvet.models.evidence import resolve_trusted_observation

        return resolve_trusted_observation(
            self._claim(), [self._fact(period_end)],
            expected_period_end=end, expected_period_start=start)

    ANNUAL_2024 = {"start": "2024-01-01", "end": "2024-12-31"}

    def test_a_september_fiscal_year_end_is_accepted_for_fy2024(self):
        """Apple. The exact case that reached a reviewer with the right
        number already retrieved."""
        assert self._resolve("2024-09-28", **self.ANNUAL_2024) is not None

    @pytest.mark.parametrize("period_end,issuer", [
        ("2024-06-30", "Microsoft"),
        ("2024-01-28", "Nvidia"),
        ("2024-12-31", "a calendar-year issuer"),
        ("2024-02-29", "a February year-end"),
    ])
    def test_any_year_end_inside_the_window_is_accepted(self, period_end, issuer):
        assert self._resolve(period_end, **self.ANNUAL_2024) is not None, issuer

    @pytest.mark.parametrize("period_end", ["2023-09-30", "2025-09-27",
                                            "2023-12-31", "2025-01-01"])
    def test_a_year_outside_the_window_is_still_rejected(self, period_end):
        """The guard's purpose survives: a correct figure from the wrong
        fiscal year is the wrong evidence, not weak evidence."""
        assert self._resolve(period_end, **self.ANNUAL_2024) is None

    def test_a_quarter_from_the_wrong_quarter_is_still_rejected(self):
        """Membership must not be so loose that it admits any quarter of the
        year. A Q1 window ends in March; a Q2 filing is outside it."""
        assert self._resolve("2024-06-29", start="2024-01-01",
                             end="2024-03-31") is None

    def test_the_right_quarter_is_accepted(self):
        assert self._resolve("2024-03-30", start="2024-01-01",
                             end="2024-03-31") is not None

    def test_a_fact_with_no_period_is_still_rejected(self):
        assert self._resolve(None, **self.ANNUAL_2024) is None

    def test_equality_still_works_when_no_start_is_known(self):
        """Callers that know only the end date keep the old exact behaviour."""
        from finvet.models.evidence import resolve_trusted_observation

        assert resolve_trusted_observation(
            self._claim(), [self._fact("2024-09-28")],
            expected_period_end="2024-09-28") is not None
        assert resolve_trusted_observation(
            self._claim(), [self._fact("2024-09-28")],
            expected_period_end="2024-12-31") is None


class TestThePeriodFilterSelectsAmongCandidates:
    """A filing returns several concepts for one metric. Pick the right one.

    `get_xbrl_concepts` returns one context per concept, and they are not all
    from the filing's own year: Nvidia's FY2025 10-K comes back with
    `RevenueFromContractWithCustomerExcludingAssessedTax` at **2017-01-29**
    alongside `Revenues` at 2025-01-26.

    `_observation_from_items` returned the first concept in
    `METRIC_TO_CONCEPTS` order, and the period check then ran *after* it --
    rejecting the stale fact and moving to the next tool *record*, never to the
    next item in the same payload. The correct figure was in the response and
    was never considered, so a numeric claim resolved nothing and failed
    closed.

    The filter has to choose among candidates, not audit a choice already made.
    """

    def _record(self):
        from finvet.models.evidence import tool_record_from_result

        return tool_record_from_result("get_income_statement", {}, {
            "success": True,
            "items": [
                # Stale comparative context, listed first by concept order.
                {"line_item": "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "value": 6_910_000_000.0, "units": "USD",
                 "period_end": "2017-01-29"},
                # The filing's own year.
                {"line_item": "Revenues", "value": 130_497_000_000.0,
                 "units": "USD", "period_end": "2025-01-26"},
            ]})

    def _claim(self):
        from finvet.models.claim import ParsedClaim

        return ParsedClaim(claim_type="sec", ticker="NVDA", metric="revenue",
                           operator="eq", value=130_500_000_000.0,
                           period="fiscal year 2025")

    def _resolve(self, **kwargs):
        from finvet.models.evidence import resolve_trusted_observation

        return resolve_trusted_observation(self._claim(), [self._record()], **kwargs)

    def test_the_in_period_concept_is_chosen_over_the_stale_one(self):
        observation = self._resolve(expected_period_start="2025-01-01",
                                    expected_period_end="2025-12-31")

        assert observation is not None, (
            "the correct fact was in the payload and was not considered")
        assert observation.value == 130_497_000_000.0
        assert observation.period_end == "2025-01-26"

    def test_it_works_with_an_exact_expected_end_too(self):
        observation = self._resolve(expected_period_end="2025-01-26")

        assert observation is not None
        assert observation.value == 130_497_000_000.0

    def test_concept_order_still_decides_when_both_are_in_period(self):
        """The tie-break is unchanged: with nothing to separate them on
        period, the first concept in METRIC_TO_CONCEPTS order still wins."""
        from finvet.models.claim import ParsedClaim
        from finvet.models.evidence import (
            resolve_trusted_observation, tool_record_from_result)

        record = tool_record_from_result("get_income_statement", {}, {
            "success": True,
            "items": [
                {"line_item": "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "value": 111.0, "units": "USD", "period_end": "2025-01-26"},
                {"line_item": "Revenues", "value": 222.0, "units": "USD",
                 "period_end": "2025-01-26"},
            ]})
        claim = ParsedClaim(claim_type="sec", ticker="NVDA", metric="revenue",
                            operator="eq", value=111.0, period="fiscal year 2025")

        observation = resolve_trusted_observation(
            claim, [record], expected_period_start="2025-01-01",
            expected_period_end="2025-12-31")

        assert observation.value == 111.0

    def test_no_in_period_candidate_still_yields_nothing(self):
        """The guard's purpose survives: if every candidate is from the wrong
        year, there is no evidence, not a best guess."""
        assert self._resolve(expected_period_start="2030-01-01",
                             expected_period_end="2030-12-31") is None

"""Tests for BaseVerificationAgent verdict override and tolerance logic."""

from unittest.mock import MagicMock
from finvet.agents.base import BaseVerificationAgent
from finvet.config.constants import (
    TOLERANCE_DEFAULT,
    TOLERANCE_LARGE_VALUE_THRESHOLD,
    TOLERANCE_MARKET,
    TOLERANCE_NEWS,
    TOLERANCE_SEC_LARGE,
    TOLERANCE_SEC_SMALL,
)


class ConcreteAgent(BaseVerificationAgent):
    """Minimal concrete agent for testing base class methods."""

    def __init__(self, agent_type="sec"):
        # Skip __init__ (avoids LLM creation) and set fields directly
        self.agent_type = agent_type
        self.tools = []
        self.system_prompt = "test"
        self.max_iterations = 5
        self.tool_map = {}

    def _get_source_description(self) -> str:
        return "Test Source"


class TestGetTolerance:

    def test_market_tolerance(self):
        agent = ConcreteAgent(agent_type="market")
        assert agent._get_tolerance(100.0) == TOLERANCE_MARKET

    def test_sec_large_value_tolerance(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(2_000_000_000) == TOLERANCE_SEC_LARGE

    def test_sec_small_value_tolerance(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(500_000_000) == TOLERANCE_SEC_SMALL

    def test_sec_boundary_value(self):
        agent = ConcreteAgent(agent_type="sec")
        # Exactly $1B should use small tolerance (not >=)
        assert agent._get_tolerance(TOLERANCE_LARGE_VALUE_THRESHOLD) == TOLERANCE_SEC_LARGE

    def test_news_tolerance(self):
        agent = ConcreteAgent(agent_type="news")
        assert agent._get_tolerance(50.0) == TOLERANCE_NEWS

    def test_unknown_agent_default(self):
        agent = ConcreteAgent(agent_type="unknown")
        assert agent._get_tolerance(100.0) == TOLERANCE_DEFAULT

    def test_none_value_sec(self):
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(None) == TOLERANCE_SEC_SMALL


class TestBuildContext:

    def test_minimal_context(self):
        agent = ConcreteAgent()
        state = {"claim_raw": "Apple revenue was $94B"}
        context = agent._build_context(state)
        assert "Apple revenue was $94B" in context
        assert "# Your Task" in context

    def test_context_with_parsed_claim(self):
        from finvet.models.claim import ParsedClaim
        agent = ConcreteAgent()
        parsed = ParsedClaim(
            claim_type="sec",
            ticker="AAPL",
            value=94_000_000_000,
            operator="eq",
            period="FY2024",
        )
        state = {"claim_raw": "test", "parsed_claim": parsed}
        context = agent._build_context(state)
        assert "AAPL" in context
        assert "94,000,000,000" in context
        assert "FY2024" in context

    def test_context_with_memory(self):
        agent = ConcreteAgent()
        state = {
            "claim_raw": "test",
            "memory_context": {
                "claim": "prior claim",
                "verdict": "SUPPORTS",
                "confidence": 0.90,
                "similarity": 0.96,
                "summary": "Previously verified",
            },
        }
        context = agent._build_context(state)
        assert "Prior Verification" in context
        assert "prior claim" in context
        assert "MUST verify independently" in context

    def test_context_with_canonical_period(self):
        from finvet.models.claim import CanonicalPeriod
        agent = ConcreteAgent()
        period = CanonicalPeriod(
            period_type="quarterly",
            start_date="2024-07-01",
            end_date="2024-09-28",
            fiscal_year=2024,
            fiscal_quarter="Q4",
        )
        state = {"claim_raw": "test", "canonical_period": period}
        context = agent._build_context(state)
        assert "quarterly" in context
        assert "2024-07-01" in context
        assert "Q4" in context


class TestEqToleranceCoversRealRoundingSpread:
    """The SEC-large equality tolerance must cover how people actually round.

    Measured across 255 real-sourced eq rows (claimed vs SEC-filed value):
    median 0.004%, p95 1.034%, max 3.067%. The original 1.0% threshold sat
    *below* p95 — true claims like "$185 billion" against a filed
    $182.8B-per-rounding were being REFUTED on rounding alone. 1.5% covers the
    observed p95 with margin while staying far under the 3.067% outlier, so
    genuinely wrong claims still fail.
    """

    def _override(self, claimed, retrieved):
        from finvet.agents.base import VerdictOutput
        from finvet.models.claim import ParsedClaim
        agent = ConcreteAgent(agent_type="sec")
        parsed = ParsedClaim(claim_type="sec", ticker="T",
                             value=claimed, operator="eq")
        verdict_output = VerdictOutput(
            verdict="NOT_ENOUGH_INFO", confidence=0.5,
            reasoning="test", retrieved_value=retrieved,
        )
        verdict, confidence, diff = agent._apply_override(
            verdict_output, {"parsed_claim": parsed}, []
        )
        return verdict, diff

    def test_p95_rounding_spread_is_supported(self):
        """1.2% difference on a >$1B claim — inside the measured p95 band."""
        verdict, diff = self._override(185_000_000_000.0, 182_780_000_000.0)
        assert 1.0 < diff < 1.5  # the band the old tolerance wrongly refuted
        assert verdict == "SUPPORTS"

    def test_exact_p95_case_is_supported(self):
        """The measured p95 itself: 1.034% must pass."""
        verdict, diff = self._override(100_000_000_000.0, 98_966_000_000.0)
        assert verdict == "SUPPORTS"

    def test_genuinely_wrong_claim_still_refuted(self):
        """3% off is the outlier region, not rounding — must stay REFUTES."""
        verdict, diff = self._override(100_000_000_000.0, 97_000_000_000.0)
        assert verdict == "REFUTES"

    def test_small_value_tolerance_unchanged(self):
        """Sub-$1B claims keep the 2.0% threshold; only SEC-large moved."""
        agent = ConcreteAgent(agent_type="sec")
        assert agent._get_tolerance(500_000_000) == 2.0


class TestOverrideReadsOperator:
    """Stage 05: the override reads the contract name. The mirror guarantees
    operator == comparison during EXPAND, so this is behaviourally identical
    for the five original comparators — pinned here so the CONTRACT-phase
    removal of `comparison` cannot silently disable the safety net."""

    def _override(self, claimed, retrieved, **claim_kwargs):
        from finvet.agents.base import VerdictOutput
        from finvet.models.claim import ParsedClaim
        agent = ConcreteAgent(agent_type="sec")
        parsed = ParsedClaim(claim_type="sec", ticker="T", value=claimed,
                             **claim_kwargs)
        verdict_output = VerdictOutput(
            verdict="NOT_ENOUGH_INFO", confidence=0.5,
            reasoning="test", retrieved_value=retrieved,
        )
        return agent._apply_override(verdict_output, {"parsed_claim": parsed}, [])

    def test_operator_only_construction_still_drives_the_override(self):
        verdict, _, _ = self._override(100e9, 120e9, operator="gt")
        assert verdict == "SUPPORTS"          # 120 > 100

    def test_approx_widens_the_tolerance(self):
        """Measured across 36 approx rows: p95 spread 1.25%, max 2.14%.
        2.0x the SEC-large tolerance (1.5% -> 3.0%) covers 97% of them.
        2.5% off: inside approx tolerance, outside plain eq."""
        verdict, _, diff = self._override(100e9, 97.5e9, operator="approx")
        assert 1.5 < diff < 3.0
        assert verdict == "SUPPORTS"

    def test_approx_still_refutes_beyond_the_widened_band(self):
        verdict, _, _ = self._override(100e9, 90e9, operator="approx")
        assert verdict == "REFUTES"

    def test_range_behaves_as_approx_on_the_midpoint(self):
        """Gold stores the band midpoint in value; the band itself is not in
        the schema. Unvalidated assumption (zero range rows carry a filed
        value) — implemented as approx and excluded from accuracy gates."""
        verdict, _, _ = self._override(30e9, 30.6e9, operator="range")
        assert verdict == "SUPPORTS"

    def test_unknown_operator_fails_closed(self):
        """An operator the override does not recognise must not silently skip
        the safety net (today's behaviour): with both numbers in hand but no
        way to compare them, the deterministic layer declines to verify
        rather than letting the LLM verdict pass unchecked. Reachable only
        through schema drift, which is exactly when it matters."""
        from finvet.agents.base import VerdictOutput
        agent = ConcreteAgent(agent_type="sec")
        parsed = MagicMock()
        parsed.value = 100e9
        parsed.operator = "between"           # drift: not one of the seven
        parsed.operator = "between"
        verdict_output = VerdictOutput(
            verdict="SUPPORTS", confidence=0.9,
            reasoning="test", retrieved_value=100e9,
        )
        verdict, confidence, _ = agent._apply_override(
            verdict_output, {"parsed_claim": parsed}, []
        )
        assert verdict == "NOT_ENOUGH_INFO"
        assert confidence <= 0.5


class TestBuildContextCarriesTheContract:
    """Stage 05, reader 4: the agent is told the resolved metric instead of
    being left to infer it from prose; the dead currency line goes."""

    def _context(self, **claim_kwargs):
        from finvet.models.claim import ParsedClaim
        agent = ConcreteAgent()
        parsed = ParsedClaim(claim_type="sec", ticker="AAPL", **claim_kwargs)
        return agent._build_context({"claim_raw": "test", "parsed_claim": parsed})

    def test_metric_is_stated_to_the_agent(self):
        context = self._context(metric="operating_cash_flow")
        assert "operating_cash_flow" in context
        assert "Metric" in context

    def test_null_metric_stays_silent(self):
        """The fall-through policy for absent and derived metrics: say
        nothing, and the agent infers from claim text exactly as it did
        before the field existed."""
        context = self._context()
        assert "Metric" not in context

    def test_currency_no_longer_reaches_the_prompt(self):
        """Post-CONTRACT the guarantee is structural: the field is gone, so
        the model refuses it and no context line can exist."""
        import pytest
        from pydantic import ValidationError
        from finvet.models.claim import ParsedClaim
        with pytest.raises(ValidationError):
            ParsedClaim(claim_type="sec", ticker="AAPL", currency="USD")
        assert "Currency" not in self._context()


class TestRetrievedValueFallbackIsMetricGuided:
    """Stage 07c. The fallback used to pick the tool-result number CLOSEST to
    the claim — selecting whichever figure best agreed with what it was
    checking, a confirmation bias directly under the deterministic override.
    Now: the claim's metric selects by XBRL concept, and with no metric to
    guide it the fallback returns None — NOT_ENOUGH_INFO is the honest
    answer, not the friendliest number in the pile."""

    RESULT = ("success=True statement_type='income' items=[{'line_item': "
              "'RevenueFromContractWithCustomerExcludingAssessedTax', "
              "'value': 391035000000.0, 'period': '2024-09-28'}, "
              "{'line_item': 'CostOfGoodsAndServicesSold', "
              "'value': 210352000000.0, 'period': '2024-09-28'}, "
              "{'line_item': 'NetIncomeLoss', 'value': 93736000000.0, "
              "'period': '2024-09-28'}]")

    def _extract(self, metric, claimed):
        from finvet.agents.base import BaseVerificationAgent
        from finvet.models.claim import ParsedClaim
        parsed = ParsedClaim(claim_type="sec", ticker="AAPL", metric=metric,
                             value=claimed, operator="eq")
        detail = [{"tool": "get_income_statement", "success": True,
                   "result": self.RESULT}]
        return BaseVerificationAgent._extract_retrieved_value(detail, parsed)

    def test_metric_selects_by_concept_not_by_agreement(self):
        """Claimed $210B — the old code would return CostOfGoodsSold
        (agrees perfectly); the metric says revenue, so revenue it is."""
        got = self._extract("revenue", 210_352_000_000.0)
        assert got == 391_035_000_000.0

    def test_net_income_metric_finds_its_concept(self):
        assert self._extract("net_income", 90e9) == 93_736_000_000.0

    def test_no_metric_means_no_guess(self):
        """The bias retired: without guidance the fallback declines, and the
        pipeline says NOT_ENOUGH_INFO instead of confirming the claim with
        whichever number sat nearest to it."""
        assert self._extract(None, 210_352_000_000.0) is None

    def test_metric_whose_concept_is_absent_declines(self):
        assert self._extract("capex", 15e9) is None


class TestToolResultPreviewCoversTheStatement:
    """Real-data finding from stage-07 verification: the 1000-char result
    preview cut a 14-item income statement before NetIncomeLoss (9th item),
    so the metric-guided fallback could see revenue but not net income in the
    SAME persisted output. Fail-closed made that a None, never a wrong
    number — but the window must cover a full statement."""

    def _detail_for(self, n_items=14):
        from langchain_core.messages import AIMessage, ToolMessage
        agent = ConcreteAgent()
        items = ", ".join(
            "{'line_item': 'Filler%dConcept', 'value': %d.0, "
            "'period': '2024-09-28'}" % (i, 10**9 + i) for i in range(n_items - 1))
        content = ("success=True statement_type='income' items=[" + items +
                   ", {'line_item': 'NetIncomeLoss', 'value': 93736000000.0, "
                   "'period': '2024-09-28'}]")
        assert len(content) > 1000   # the old window must genuinely cut it
        msgs = [AIMessage(content="", tool_calls=[
                    {"name": "get_income_statement", "args": {}, "id": "t1"}]),
                ToolMessage(content=content, tool_call_id="t1")]
        _, detail, _ = agent._extract_tool_info(msgs)
        return detail

    def test_late_listed_concept_survives_the_preview(self):
        from finvet.agents.base import BaseVerificationAgent
        from finvet.models.claim import ParsedClaim
        parsed = ParsedClaim(claim_type="sec", ticker="AXP",
                             metric="net_income", value=9e10, operator="eq")
        got = BaseVerificationAgent._extract_retrieved_value(
            self._detail_for(), parsed)
        assert got == 93_736_000_000.0

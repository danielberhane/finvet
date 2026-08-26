"""A claim under review shows what evidence was gathered before it paused.

`Apple was fined 500 million euros by the European Commission in 2024` runs the
News agent, which calls `corroborate_with_filing` — the delegation fires, the
SEC agent reads the filing, and the result is recorded in state. The claim then
escalates, and `_generate_hitl_response` builds a metadata block that omits
`data_sources` entirely. So the A2A badge never renders, and the feature looks
broken when it worked.

The same omission hides retrieved RAG chunks and XBRL provenance on any claim
that pauses. A reviewer opening that claim is the person who most needs to see
what was found; they were shown the least.

`execution.py` already records this exact defect one path over: "streaming HITL
omitted the data_sources the synchronous path sent". This is that fix applied
to the response builder rather than the route.

The provenance block is built by one function used by both paths, so a
successful run and a paused one cannot describe the same evidence differently.
"""

import pytest


def _state(**overrides):
    base = {
        "request_id": "req_x",
        "claim_raw": "Apple was fined 500 million euros",
        "hitl_triggers": ["low_confidence"],
        "agent_evidence": {
            "agent": "news", "verdict": "NOT_ENOUGH_INFO", "confidence": 0.3,
            "tools_called": ["search_financial_news", "corroborate_with_filing"],
            "reasoning": "found coverage", "provenance": [],
        },
    }
    base.update(overrides)
    return base


def _corroboration():
    return {"direction": "news_to_sec", "status": "CORROBORATES",
            "trigger_mode": "agent", "verdict": "SUPPORTS", "confidence": 0.9,
            "retrieved_value": 500_000_000.0, "sources": [], "provenance": []}


def _hitl(state):
    from finvet.graph.nodes.response_generator import _generate_hitl_response

    return _generate_hitl_response(state)["final_response"]


class TestOnePlaceBuildsProvenance:
    """Extracted so the paused and successful paths cannot drift."""

    def test_the_builder_exists_and_is_pure(self):
        from finvet.graph.nodes.response_generator import build_data_sources

        assert build_data_sources({}, {}) == {}

    def test_it_reports_xbrl_when_a_statement_tool_ran(self):
        from finvet.graph.nodes.response_generator import build_data_sources

        sources = build_data_sources(
            {}, {"tools_called": ["get_income_statement"]})

        assert sources["xbrl"]["used"] is True

    def test_it_reports_a2a_when_delegation_ran(self):
        from finvet.graph.nodes.response_generator import build_data_sources

        sources = build_data_sources(
            {"corroboration_result": _corroboration()}, {})

        assert sources["a2a"]["used"] is True
        assert sources["a2a"]["status"] == "CORROBORATES"


class TestAPausedClaimShowsItsEvidence:

    def test_a2a_survives_the_escalation(self):
        """The defect. The delegation ran and the badge never appeared."""
        response = _hitl(_state(corroboration_result=_corroboration()))
        sources = response["metadata"].get("data_sources") or {}

        assert sources.get("a2a", {}).get("used") is True, (
            "corroboration ran and was dropped from the paused response")
        assert sources["a2a"]["status"] == "CORROBORATES"

    def test_rag_chunks_survive_the_escalation(self):
        response = _hitl(_state(rag_chunks_retrieved=[
            {"section": "risk_factors", "filing_type": "10-K",
             "period_end": "2025-09-27", "evidence_id": "abc",
             "chunk_text": "x", "search_query": "q"}]))
        sources = response["metadata"].get("data_sources") or {}

        assert sources.get("rag", {}).get("used") is True

    def test_xbrl_provenance_survives_the_escalation(self):
        response = _hitl(_state(agent_evidence={
            "agent": "sec", "verdict": "NOT_ENOUGH_INFO", "confidence": 0.3,
            "tools_called": ["get_income_statement"], "reasoning": "r"}))
        sources = response["metadata"].get("data_sources") or {}

        assert sources.get("xbrl", {}).get("used") is True

    def test_a_run_with_no_evidence_reports_none(self):
        """The control. An empty dict, not a missing key — a reader can tell
        'nothing was used' from 'nobody looked'."""
        response = _hitl(_state())

        assert response["metadata"]["data_sources"] == {}

    def test_the_pending_contract_is_otherwise_unchanged(self):
        """Everything the UI and the audit trail already read must survive."""
        response = _hitl(_state(corroboration_result=_corroboration()))

        assert response["status"] == "pending_review"
        assert response["verdict"] == "PENDING"
        assert response["confidence"] == 0.0
        assert response["metadata"]["hitl_required"] is True
        assert response["metadata"]["hitl_triggers"] == ["low_confidence"]
        assert response["metadata"]["disposition"] == "pending_review"
        assert "preliminary_analysis" in response


class TestTheSuccessPathIsUnchanged:
    """The extraction must not alter what a completed run reports."""

    def _metadata(self, state, evidence):
        from finvet.graph.nodes.response_generator import _format_metadata

        return _format_metadata(state, evidence)

    def test_a_completed_run_still_reports_its_sources(self):
        metadata = self._metadata(
            {"corroboration_result": _corroboration()},
            {"tools_called": ["get_income_statement"], "agent": "sec"})

        assert metadata["data_sources"]["xbrl"]["used"] is True
        assert metadata["data_sources"]["a2a"]["used"] is True

    @pytest.mark.parametrize("key", [
        "agent", "tools_called", "limitation", "trusted_observation"])
    def test_the_existing_metadata_keys_survive(self, key):
        metadata = self._metadata({}, {"agent": "sec", "tools_called": []})

        assert key in metadata


class TestTheBuilderTheRoutesActuallyUse:
    """There are two pending-response builders, and only one reaches a client.

    `_generate_hitl_response` lives in the graph node; `build_pending_response`
    lives in `api/execution.py` and is what `/verify` and `/verify-stream`
    return. Its docstring says it exists so "both routes describe a pending
    review identically" -- and it had drifted from the node's version, carrying
    four metadata keys where the node carried six.

    Fixing the node alone changed nothing a user could see. This class drives
    the builder on the path to the client, which is the only one that settles
    whether the A2A badge renders.
    """

    def _pending(self, state):
        from finvet.api.execution import build_pending_response

        return build_pending_response("req_x", "a claim", state)

    def test_a2a_reaches_the_client(self):
        response = self._pending(_state(corroboration_result=_corroboration()))
        sources = response["metadata"].get("data_sources") or {}

        assert sources.get("a2a", {}).get("used") is True, (
            "the delegation ran and the client was told nothing about it")
        assert sources["a2a"]["status"] == "CORROBORATES"
        assert sources["a2a"]["trigger_mode"] == "agent"

    def test_rag_reaches_the_client(self):
        response = self._pending(_state(rag_chunks_retrieved=[
            {"section": "risk_factors", "filing_type": "10-K",
             "period_end": "2025-09-27", "evidence_id": "abc",
             "chunk_text": "x", "search_query": "q"}]))

        assert (response["metadata"]["data_sources"]
                .get("rag", {}).get("used")) is True

    def test_an_empty_run_reports_an_empty_map(self):
        response = self._pending(_state())

        assert response["metadata"]["data_sources"] == {}

    def test_the_two_builders_agree_on_provenance(self):
        """The property the duplication keeps breaking. Whatever else differs,
        both must describe the same evidence the same way."""
        state = _state(corroboration_result=_corroboration())

        from_node = _hitl(state)["metadata"]["data_sources"]
        from_route = self._pending(state)["metadata"]["data_sources"]

        assert from_node == from_route

    def test_the_existing_pending_contract_is_unchanged(self):
        response = self._pending(_state())

        assert response["status"] == "pending_review"
        assert response["verdict"] == "PENDING"
        assert response["hitl_triggers"] == ["low_confidence"]
        assert response["metadata"]["hitl_required"] is True
        assert response["metadata"]["agent"] == "news"

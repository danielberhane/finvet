"""Tests for the claim_parsed audit event — stage 04b.

The parse is the pipeline's most consequential single decision and, before
this, the only node leaving no audit record. The event goes through
get_audit_logger().log_event() — the path that actually persists — and NOT
the state["audit_events"] pattern, which is write-only: three nodes append
to it and nothing ever reads it (0 rows in the database for their event
types across 415 verifications).
"""

import json
from unittest.mock import MagicMock, patch

from finvet.graph.nodes.claim_parser import claim_parser

RAW = {"claim_type": "sec", "ticker": "JPM", "value": 158.1e9,
       "comparison": "eq", "period": "fiscal 2024", "currency": "USD",
       "reject_reason": None}


def _run(raw=RAW):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(raw),
                                        response_metadata={})
    audit = MagicMock()
    with patch("finvet.graph.nodes.claim_parser.create_llm", return_value=llm), \
         patch("finvet.graph.nodes.claim_parser.get_audit_logger",
               return_value=audit):
        result = claim_parser({"claim_raw": "JPMorgan reported revenue of "
                               "$158.1 billion for fiscal 2024.",
                               "request_id": "req_test04b"})
    return result, audit


class TestClaimParsedEvent:

    def test_one_event_is_emitted_through_the_working_path(self):
        _, audit = _run()
        calls = [c for c in audit.log_event.call_args_list
                 if c.kwargs.get("event_type") == "claim_parsed"]
        assert len(calls) == 1
        assert calls[0].kwargs["request_id"] == "req_test04b"

    def test_event_carries_the_seven_resolved_fields(self):
        """'Why did it pick that metric?' must be answerable from the trail."""
        _, audit = _run()
        data = [c for c in audit.log_event.call_args_list
                if c.kwargs.get("event_type") == "claim_parsed"][0].kwargs["data"]
        fields = data["fields"]
        assert set(fields) == {"claim_type", "ticker", "metric", "operator",
                               "value", "period", "reject_reason"}
        assert fields["claim_type"] == "sec"
        assert fields["operator"] == "eq"

    def test_event_carries_raw_output_and_decisions(self):
        _, audit = _run()
        data = [c for c in audit.log_event.call_args_list
                if c.kwargs.get("event_type") == "claim_parsed"][0].kwargs["data"]
        assert data["raw"]["comparison"] == "eq"      # pre-normalisation
        assert data["decisions"]["reject"] == "none"
        assert data["decisions"]["metric"] == "absent"

    def test_normalisation_decisions_are_visible(self):
        """A coerced reject shows its coercion in the trail."""
        raw = dict(RAW, reject_reason="incomplete")
        _, audit = _run(raw)
        data = [c for c in audit.log_event.call_args_list
                if c.kwargs.get("event_type") == "claim_parsed"][0].kwargs["data"]
        assert data["decisions"]["reject"] == "coerced_reject"
        assert data["fields"]["claim_type"] == "reject"
        assert data["fields"]["ticker"] is None       # §8 nulling recorded

    def test_parse_still_returns_the_claim(self):
        result, _ = _run()
        assert result["parsed_claim"].ticker == "JPM"

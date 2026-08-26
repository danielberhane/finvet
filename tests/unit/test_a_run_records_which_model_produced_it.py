"""A result that does not say which model produced it cannot be compared to another.

FinVet's central claim is that a deterministic comparison overrules the model.
Testing that claim means varying the model and diffing the outcomes -- and a
diff needs both sides to be attributable. Two things prevented that:

  1. `AuditExecution` recorded no model at all. A stored run was silent about
     the parser, agent and verdict models that produced it, so a run under one
     model and a run under another were indistinguishable after the fact.
  2. `claim_parser` wrote `"parser": "deepseek"` as a literal into the
     `claim_parsed` event. Under the configured default that string happens to
     be correct, which is why it survived; it becomes a falsehood the moment
     `LLM_PARSER__MODEL` names anything else, and a falsehood in the audit
     trail is worse than a gap.

The field is named `llm_config`, not `model_config`: `model_config` is
Pydantic v2's reserved configuration attribute, used in this codebase by
`config/settings.py`, `models/claim.py` and `api/models.py`. A data field of
that name would shadow it in any Pydantic model that ever mirrors these rows.

`llm_config` is carried *inside* the hashed envelope rather than beside it.
`build_execution_envelope` states the invariant it exists to enforce: "a field
stored beside the checksum but absent from it is a field the checksum does not
cover, which is exactly the defect this replaces."
"""

from finvet.audit.integrity import (
    ENVELOPE_SCHEMA_VERSION,
    build_execution_envelope,
    compare_execution_projection,
    compute_execution_checksum,
    verify_execution_checksum,
)


def _envelope(**overrides):
    base = dict(
        request_id="req_1", claim_text="c", terminal_status=None,
        verdict="SUPPORTS", confidence=0.9, agents_run=["sec"], events=[],
        final_response=None, data_sources=None,
        llm_config={"parser": {"model": "deepseek-chat"}},
    )
    base.update(overrides)
    return build_execution_envelope(**base)


class TestTheEnvelopeCarriesTheModel:

    def test_the_hashed_envelope_includes_llm_config(self):
        assert _envelope()["llm_config"] == {"parser": {"model": "deepseek-chat"}}

    def test_changing_the_model_changes_the_checksum(self):
        """The point of carrying it inside: it cannot be edited silently."""
        a = compute_execution_checksum(_envelope())
        b = compute_execution_checksum(
            _envelope(llm_config={"parser": {"model": "minimax-m2.5"}}))

        assert a != b

    def test_the_schema_version_was_bumped(self):
        """integrity.py: 'changing the fields below requires bumping this.'"""
        assert ENVELOPE_SCHEMA_VERSION >= 2
        assert _envelope()["schema_version"] == ENVELOPE_SCHEMA_VERSION


class TestDriftIsDetected:
    """The row keeps llm_config denormalized beside `full_trace`; the two must
    be compared, or the column is free to drift while the digest still verifies."""

    def test_a_matching_row_reports_no_mismatch(self):
        envelope = _envelope()
        row = {"request_id": "req_1", "claim_text": "c", "verdict": "SUPPORTS",
               "confidence": 0.9, "agents_run": ["sec"], "data_sources": None,
               "llm_config": {"parser": {"model": "deepseek-chat"}},
               "full_trace": envelope}

        assert compare_execution_projection(row, []) == []

    def test_an_edited_column_is_caught(self):
        envelope = _envelope()
        row = {"request_id": "req_1", "claim_text": "c", "verdict": "SUPPORTS",
               "confidence": 0.9, "agents_run": ["sec"], "data_sources": None,
               "llm_config": {"parser": {"model": "gpt-4o"}},  # edited
               "full_trace": envelope}

        assert "llm_config" in compare_execution_projection(row, [])


class TestOlderRowsStillVerify:
    """Rows written under schema 1 have no llm_config. Verification recomputes
    from the stored envelope, so they must keep verifying untouched."""

    def test_a_schema_1_envelope_still_verifies(self):
        legacy = {"schema_version": 1, "request_id": "req_0", "claim": "c",
                  "terminal_status": None, "verdict": "SUPPORTS",
                  "confidence": 0.9, "agents_run": [], "events": [],
                  "final_response": None, "data_sources": None}
        checksum = compute_execution_checksum(legacy)

        assert verify_execution_checksum(legacy, checksum) is True

    def test_a_legacy_row_reports_no_llm_config_drift(self):
        legacy = {"schema_version": 1, "request_id": "req_0", "claim": "c",
                  "terminal_status": None, "verdict": None, "confidence": None,
                  "agents_run": [], "events": [], "final_response": None,
                  "data_sources": None}
        row = {"request_id": "req_0", "claim_text": "c", "verdict": None,
               "confidence": None, "agents_run": [], "data_sources": None,
               "llm_config": None, "full_trace": legacy}

        assert compare_execution_projection(row, []) == []


class TestTheParserNamesTheConfiguredModel:
    """Driven through the graph node, not a hand-built dict: the defect was in
    what the producer emitted, and a fixture would assert the shape I
    remembered rather than the shape the node writes (CLAUDE.md gotcha 6)."""

    def _run_parser(self, monkeypatch, model_name):
        # `nodes/__init__.py` re-exports the function under the module's own
        # name, so `finvet.graph.nodes.claim_parser` resolves to the function
        # and shadows the submodule. The registry still holds the module.
        import importlib
        node = importlib.import_module("finvet.graph.nodes.claim_parser")
        from finvet.config.settings import settings

        captured = {}

        class _Logger:
            def log_event(self, event_type, request_id, data, **kw):
                if event_type == "claim_parsed":
                    captured.update(data)

        monkeypatch.setattr(node, "get_audit_logger", lambda: _Logger())
        monkeypatch.setattr(settings.llm_parser, "model", model_name)
        monkeypatch.setattr(node, "create_llm", lambda purpose: _StubLLM())
        return captured

    def test_it_reports_the_configured_model(self, monkeypatch):
        captured = self._run_parser(monkeypatch, "minimax-m2.5")

        from finvet.graph.nodes.claim_parser import claim_parser
        claim_parser({"claim_raw": "Apple's revenue was $391 billion in 2024",
                      "request_id": "req_1"})

        assert captured.get("parser") == "minimax-m2.5", (
            "the audit named a model that did not run")

    def test_it_does_not_hardcode_deepseek(self, monkeypatch):
        captured = self._run_parser(monkeypatch, "some-other-model")

        from finvet.graph.nodes.claim_parser import claim_parser
        claim_parser({"claim_raw": "Apple's revenue was $391 billion in 2024",
                      "request_id": "req_1"})

        assert captured.get("parser") != "deepseek"


class _StubLLM:
    """Returns a parseable claim without reaching an API."""

    def invoke(self, messages):
        class _R:
            content = (
                '{"claim_type": "sec", "ticker": "AAPL", "metric": "revenue",'
                ' "operator": "eq", "value": 391000000000, "period": "2024",'
                ' "reject_reason": null}')
        return _R()

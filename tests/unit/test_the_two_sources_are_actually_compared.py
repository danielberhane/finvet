"""A decline is not a disagreement.

Live trace, `Apple was fined ... worth 1 trillion dollars`:

    news LLM read the articles          -> REFUTES
    deterministic guard overruled it    -> NOT_ENOUGH_INFO
    SEC agent read 5 filing passages    -> REFUTES, retrieved 500,000,000
    a2a status recorded                 -> FOUND_UNCERTIFIED
    final verdict                       -> REFUTES @ 0.95

The verdict is right and rests on a figure Python pulled out of the 10-K. The
*label* is wrong: FOUND_UNCERTIFIED says "passages were read, nothing could be
certified" in a record that carries a certified 500,000,000.

The cause is what the two sides of the comparison were. `classify_status` was
handed the news agent's **final** verdict -- a structural decline meaning "I
have no way to certify a fine amount", because `fine_amount` has no XBRL
concept and news tools return prose. A decline cannot agree or disagree with
anything, so every delegation collapsed to NO_MATCHING_DISCLOSURE and
CORROBORATES / CONTRADICTS were unreachable however decisive the filing was.

The news agent's *reading* is kept separately as `llm_original_verdict`
(base.py:445). That is what the press said, and it is the independent signal
the comparison needs.

Two things this must not do:

- **Take the verdict from the reading.** It is model output. It may set the
  label and the escalation, never the answer.
- **Let the parent borrow the target's observation.** Then both sides rest on
  one number and always agree -- the defect `reclassify_corroboration`'s own
  docstring describes, where "a filing that flatly contradicted the news was
  recorded as CORROBORATES".
"""

from finvet.models.a2a import (
    A2A_CONTRADICTS,
    A2A_CORROBORATES,
    A2A_FOUND_UNCERTIFIED,
    A2A_NO_MATCHING_DISCLOSURE,
)


class _Claim:
    claim_type = "news"
    metric = "fine_amount"
    ticker = "AAPL"
    value = 1_000_000_000_000.0
    operator = "eq"
    period = None


def _a2a(verdict, retrieved, chunks=(("c1",),)):
    """What the delegation returns before classification."""
    return {
        "success": True, "status": "PENDING_CLASSIFICATION",
        "verdict": verdict, "confidence": 0.95, "retrieved_value": retrieved,
        "reasoning": "filing states EUR 500 million",
        "provenance": [{
            "tool": "search_filing_text",
            "result": {"success": True,
                       "chunks": [{"chunk_id": c[0]} for c in chunks]},
        }],
    }


def _run(monkeypatch, *, news_reading, news_final, a2a):
    """Drive the real node. The news agent's two facts are kept apart the way
    `execute` leaves them: what it read, and what it may certify."""
    from finvet.graph.nodes import domain_agents as node

    evidence = {
        "agent": "news",
        "verdict": news_final,               # after the deterministic guard
        "llm_original_verdict": news_reading,  # what the press said
        "confidence": 0.5,
        "override_applied": news_reading != news_final,
        "provenance": [{"tool": "corroborate_with_filing",
                        "args": {"finding": "f"}, "result": a2a}],
    }
    monkeypatch.setattr(node, "_run_agent",
                        lambda *a, **k: {"agent_evidence": evidence})
    monkeypatch.setattr(node, "_unsupported_claim", lambda s: None)

    out = node.run_news_agent({"parsed_claim": _Claim(), "request_id": "r",
                               "claim_raw": "c"})
    return out["agent_evidence"], out.get("corroboration_result") or {}


class TestTheReadingIsWhatGetsCompared:

    def test_agreement_is_recorded_as_agreement(self, monkeypatch):
        """The live case. Press said REFUTES, filing said REFUTES."""
        _, corr = _run(monkeypatch, news_reading="REFUTES",
                       news_final="NOT_ENOUGH_INFO",
                       a2a=_a2a("REFUTES", 500_000_000.0))

        assert corr["status"] == A2A_CORROBORATES

    def test_a_real_conflict_is_finally_reachable(self, monkeypatch):
        """Press implies the claim stands; the filing refutes it. This status
        has never once been produced in production."""
        _, corr = _run(monkeypatch, news_reading="SUPPORTS",
                       news_final="NOT_ENOUGH_INFO",
                       a2a=_a2a("REFUTES", 500_000_000.0))

        assert corr["status"] == A2A_CONTRADICTS

    def test_the_verdict_is_never_taken_from_the_reading(self, monkeypatch):
        """The reading is model output. It may label, never decide."""
        evidence, _ = _run(monkeypatch, news_reading="SUPPORTS",
                           news_final="NOT_ENOUGH_INFO",
                           a2a=_a2a("REFUTES", 500_000_000.0))

        assert evidence["verdict"] == "REFUTES", (
            "the answer must come from the filing, not from the press reading")
        assert evidence["retrieved_value"] == 500_000_000.0


class TestUncertifiedOutcomesAreUnchanged:

    def test_passages_read_but_nothing_certified(self, monkeypatch):
        """retrieved_value None means the filing figure was never certified."""
        _, corr = _run(monkeypatch, news_reading="REFUTES",
                       news_final="NOT_ENOUGH_INFO",
                       a2a=_a2a("NOT_ENOUGH_INFO", None))

        assert corr["status"] == A2A_FOUND_UNCERTIFIED

    def test_an_empty_filing_search_still_reports_no_disclosure(self, monkeypatch):
        _, corr = _run(monkeypatch, news_reading="REFUTES",
                       news_final="NOT_ENOUGH_INFO",
                       a2a=_a2a("NOT_ENOUGH_INFO", None, chunks=()))

        assert corr["status"] == A2A_NO_MATCHING_DISCLOSURE

    def test_a_reading_the_guard_never_touched_is_used_as_is(self, monkeypatch):
        """On the deterministic-fallback path llm_original_verdict is None and
        the final verdict is already deterministic."""
        from finvet.graph.nodes import domain_agents as node

        evidence = {
            "agent": "news", "verdict": "REFUTES",
            "llm_original_verdict": None, "confidence": 0.9,
            "provenance": [{"tool": "corroborate_with_filing", "args": {},
                            "result": _a2a("REFUTES", 500_000_000.0)}],
        }
        monkeypatch.setattr(node, "_run_agent",
                            lambda *a, **k: {"agent_evidence": evidence})
        monkeypatch.setattr(node, "_unsupported_claim", lambda s: None)

        out = node.run_news_agent({"parsed_claim": _Claim(), "request_id": "r",
                                   "claim_raw": "c"})

        assert out["corroboration_result"]["status"] == A2A_CORROBORATES


class TestAConflictReachesAPerson:
    """Every CONTRADICTS is queued.

    I first gated this on "the filing already settled it", reasoning that a
    reviewer could not improve on an authoritative answer. That was wrong
    twice over. CONTRADICTS requires the SEC side to be decisive, which for a
    numeric fine claim means it holds a trusted observation, which is exactly
    what makes the verdict get adopted -- so the gate suppressed the trigger
    every time and made `source_disagreement` unreachable again.

    And the principle did not transfer. `declined_with_reason` suppresses
    review where no tool serves the metric and there is genuinely nothing to
    weigh. Here two sources report different numbers for the same event, which
    a person can act on: the press may be wrong, or the filing may be stale."""

    def _guard(self, corroboration_status, verdict_source):
        from finvet.graph.nodes.output_guardrails import output_guardrails

        state = {
            "request_id": "r", "claim_raw": "c", "verdict": "REFUTES",
            "confidence": 0.95,
            "corroboration_result": {"status": corroboration_status,
                                     "source_agent": "news",
                                     "target_agent": "sec",
                                     "verdict": "REFUTES"},
            "agent_evidence": {"verdict": "REFUTES", "confidence": 0.95,
                               "reasoning": "the filing states EUR 500 million",
                               "verdict_source": verdict_source},
        }
        return output_guardrails(state)

    def test_a_conflict_is_queued_even_when_the_filing_settled_it(self):
        """The case that matters: press and filing disagree on a real number."""
        out = self._guard(A2A_CONTRADICTS, "delegated_filing")

        assert "source_disagreement" in (out.get("hitl_triggers") or [])

    def test_a_conflict_is_queued_when_the_filing_did_not_settle_it(self):
        out = self._guard(A2A_CONTRADICTS, None)

        assert "source_disagreement" in (out.get("hitl_triggers") or [])

    def test_agreement_never_queues(self):
        out = self._guard(A2A_CORROBORATES, None)

        assert "source_disagreement" not in (out.get("hitl_triggers") or [])


class TestAnUnsoundReadingIsNotResurrected:
    """A claim naming no value takes the *qualitative* decline, not the numeric
    fail-closed. There the reading was thrown out because refuting on absence
    is a fallacy (D13, `non_corroboration_is_not_contradiction`) -- a judgment
    that the reading was unsound, not merely uncertified.

    Feeding it back as a comparison side would revive a verdict the system
    deliberately rejected, and could escalate on it. The reading stands in only
    where the claim named a value."""

    class _Valueless:
        claim_type = "news"
        metric = "fine_amount"
        ticker = "AAPL"
        value = None
        operator = None
        period = None

    def test_a_valueless_claims_reading_is_not_compared(self, monkeypatch):
        from finvet.graph.nodes import domain_agents as node

        evidence = {
            "agent": "news", "verdict": "NOT_ENOUGH_INFO",
            "llm_original_verdict": "REFUTES",   # declined as unsound
            "confidence": 0.5, "limitation": "non_corroboration_is_not_contradiction",
            "provenance": [{"tool": "corroborate_with_filing", "args": {},
                            "result": _a2a("SUPPORTS", 500_000_000.0)}],
        }
        monkeypatch.setattr(node, "_run_agent",
                            lambda *a, **k: {"agent_evidence": evidence})
        monkeypatch.setattr(node, "_unsupported_claim", lambda s: None)

        out = node.run_news_agent({"parsed_claim": self._Valueless(),
                                   "request_id": "r", "claim_raw": "c"})

        assert out["corroboration_result"]["status"] != A2A_CONTRADICTS, (
            "a reading the system rejected as unsound was used to declare "
            "the sources in conflict")


class TestTheStatusIsARealStatus:
    """FOUND_UNCERTIFIED was added as a constant annotated `A2AStatus` without
    being added to the Literal, so the annotation was false and
    `A2AResult(status=...)` rejected it. It survived because
    `reclassify_corroboration` works on plain dicts and skips validation --
    every A2AResult construction site happens to use one of the older
    statuses, which is the only reason nothing crashed.

    mypy reported it: "Incompatible types in assignment (expression has type
    Literal['FOUND_UNCERTIFIED'], variable has type Literal[...])".
    """

    def test_it_is_a_declared_status(self):
        import typing

        from finvet.models.a2a import A2A_FOUND_UNCERTIFIED, A2AStatus

        assert A2A_FOUND_UNCERTIFIED in typing.get_args(A2AStatus)

    def test_a_result_carrying_it_can_be_built(self):
        """The latent crash, pinned."""
        from finvet.models.a2a import A2A_FOUND_UNCERTIFIED, A2AResult

        result = A2AResult(
            success=True, status=A2A_FOUND_UNCERTIFIED,
            direction="news_to_sec", source_agent="news", target_agent="sec",
            verdict="NOT_ENOUGH_INFO", confidence=0.5, reasoning="",
            retrieved_value=None, temporal_scope="unknown")

        assert result.status == "FOUND_UNCERTIFIED"

    def test_every_status_constant_is_in_the_literal(self):
        """The class of bug, not just this instance."""
        import typing

        import finvet.models.a2a as a2a

        declared = set(typing.get_args(a2a.A2AStatus))
        constants = {name: value for name, value in vars(a2a).items()
                     if name.startswith("A2A_") and isinstance(value, str)}
        undeclared = {n: v for n, v in constants.items() if v not in declared}

        assert not undeclared, f"status constants outside A2AStatus: {undeclared}"

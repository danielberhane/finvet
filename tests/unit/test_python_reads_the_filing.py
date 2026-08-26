"""Python reads the filing, so the filing can decide.

A live run on "Apple was fined by the European Commission over its App Store
practices worth 1 trillion dollars" returned Human Review at 0.0 confidence.
The model said REFUTES and was overruled, because `fine_amount` has no XBRL
concept and filing prose may never certify a number.

The guard is right about the *model*, and was being applied to the *source*.
XBRL is trusted because Python pulls the value, not because it is structured.
An SEC filing is the authoritative record of what a company disclosed; what
cannot be trusted is a model reading a number out of a paragraph.

So Python reads it. Apple's 10-K, `legal_proceedings`:

    "On April 23, 2025, the Commission fined the Company EUR 500 million in the
     Article 5(4) Investigation and issued a cease and desist order..."

Against a claimed $1 trillion that is a 2000x gap, and the verdict is REFUTES
from the primary source with the passage quotable by `evidence_id`.

Extraction is allowed only where it is unambiguous, and the fixture proves both
sides of that rule with **real corpus text**, not text written for this test:

    legal_proceedings chunk -> exactly one amount   (EUR 500 million)
    mda chunk               -> seven amounts, none a fine -> declines

Everything else fails closed to today's behaviour.
"""

import json
import pathlib

import pytest

from finvet.tools.filing_amounts import extract_amount

FIXTURE = json.loads(
    (pathlib.Path(__file__).parent.parent / "fixtures"
     / "aapl_filing_chunks.json").read_text())


def _chunk(name, **overrides):
    """A chunk shaped the way `search_filing_text` returns them."""
    raw = FIXTURE[name]
    chunk = {
        "section": raw["section"],
        # The tool wraps stored text in delimiters before the model sees it;
        # extraction has to survive that.
        "chunk_text": f"<filing_excerpt>\n{raw['text']}\n</filing_excerpt>",
        "evidence_id": raw["evidence_id"],
        "content_sha256": raw["evidence_id"],
        "filing_type": raw["filing_type"],
        "period_end": raw["period_end"],
    }
    chunk.update(overrides)
    return chunk


class TestItReadsTheAmountTheFilingStates:

    def test_the_fine_is_extracted(self):
        found = extract_amount(_chunk("legal_proceedings"))

        assert found is not None
        assert found.value == 500_000_000.0
        assert found.currency == "EUR"

    def test_it_is_quotable_back_to_the_filing(self):
        """A figure that decides a verdict must be findable in its source."""
        found = extract_amount(_chunk("legal_proceedings"))

        assert found.evidence_id == FIXTURE["legal_proceedings"]["evidence_id"]
        assert found.extraction_method == "deterministic"

    def test_the_model_is_not_consulted(self):
        """No argument carries model output; the text is the only input."""
        import inspect

        params = inspect.signature(extract_amount).parameters
        assert list(params) == ["chunk"]


class TestItDeclinesWhereExtractionWouldBeGuessing:
    """The load-bearing half. A wrong number here is worse than no number."""

    def test_a_chunk_with_many_amounts_declines(self):
        """Real MD&A text: $44.1B, $43.8B, EUR 14.2B, $15.4B, $21.0B, $3.9B --
        none of them a fine. Picking one would be a coin toss."""
        assert extract_amount(_chunk("mda")) is None

    def test_the_wrong_section_declines(self):
        """Fines are disclosed in legal proceedings. The same sentence
        appearing elsewhere is not the disclosure."""
        assert extract_amount(_chunk("legal_proceedings", section="mda")) is None

    def test_fine_language_without_an_amount_declines(self):
        assert extract_amount(_chunk(
            "legal_proceedings",
            chunk_text="The Commission fined the Company an undisclosed sum.")) is None

    def test_an_amount_without_fine_language_declines(self):
        """A number in legal proceedings is not automatically a penalty."""
        assert extract_amount(_chunk(
            "legal_proceedings",
            chunk_text="The Company recorded €500 million of deferred revenue.")) is None

    def test_two_penalty_amounts_decline(self):
        assert extract_amount(_chunk(
            "legal_proceedings",
            chunk_text=("The Commission fined the Company €500 million, and "
                        "separately fined the Company €1.8 billion."))) is None

    @pytest.mark.parametrize("text", ["", "   ", "no numbers at all here"])
    def test_empty_or_numberless_text_declines(self, text):
        assert extract_amount(_chunk("legal_proceedings", chunk_text=text)) is None


class TestScaleAndCurrency:

    @pytest.mark.parametrize("text,expected", [
        ("The Commission fined the Company €500 million.", 500_000_000.0),
        ("The Commission fined the Company €1.8 billion.", 1_800_000_000.0),
        ("The Commission fined the Company $2.5 billion.", 2_500_000_000.0),
    ])
    def test_scale_words_are_applied(self, text, expected):
        found = extract_amount(_chunk("legal_proceedings", chunk_text=text))

        assert found is not None and found.value == expected

    def test_currency_is_recorded_not_converted(self):
        """Release A never converts. The currency is carried so a comparison
        can decline when it would need a rate."""
        found = extract_amount(_chunk("legal_proceedings",
                                      chunk_text="fined the Company $500 million."))

        assert found.currency == "USD"


class TestTheFilingDecidesTheClaim:
    """SEC is authoritative. When the delegated filing lookup settled the
    number deterministically, the News agent adopts that verdict instead of
    reporting NOT_ENOUGH_INFO beside a filing that answered the question.

    The licence is `retrieved_value`, which since the fail-closed fix is
    non-None *only* when Python compared a trusted observation. A model's
    reading of prose leaves it None, so this cannot adopt a model's number.
    """

    def _run(self, monkeypatch, a2a):
        from finvet.graph.nodes import domain_agents as node

        evidence = {
            "agent": "news", "verdict": "NOT_ENOUGH_INFO", "confidence": 0.5,
            "provenance": [{"tool": "corroborate_with_filing",
                            "args": {"finding": "f"}, "result": a2a}],
        }
        monkeypatch.setattr(node, "_run_agent",
                            lambda *a, **k: {"agent_evidence": evidence})
        monkeypatch.setattr(node, "_unsupported_claim", lambda s: None)

        class _Claim:
            claim_type = "news"
            metric = "fine_amount"
            ticker = "AAPL"
            value = 1_000_000_000_000.0
            operator = "eq"
            period = None

        out = node.run_news_agent({"parsed_claim": _Claim(), "request_id": "r",
                                   "claim_raw": "c"})
        return out["agent_evidence"]

    def _a2a(self, verdict, retrieved):
        return {"success": True, "status": "PENDING_CLASSIFICATION",
                "verdict": verdict, "confidence": 0.95,
                "retrieved_value": retrieved, "reasoning": "filing says EUR 500m",
                "provenance": [{"tool": "search_filing_text",
                                "result": {"success": True,
                                           "chunks": [{"chunk_id": "c1"}]}}]}

    def test_a_trusted_filing_refutation_is_adopted(self, monkeypatch):
        evidence = self._run(monkeypatch, self._a2a("REFUTES", 500_000_000.0))

        assert evidence["verdict"] == "REFUTES", (
            "the filing settled the number and the answer still said "
            "not enough info")

    def test_the_filings_number_is_carried(self, monkeypatch):
        evidence = self._run(monkeypatch, self._a2a("REFUTES", 500_000_000.0))

        assert evidence["retrieved_value"] == 500_000_000.0

    def test_an_uncertified_delegation_is_not_adopted(self, monkeypatch):
        """retrieved_value None means no trusted observation existed. The
        nested agent's verdict then rests on prose and may not decide."""
        evidence = self._run(monkeypatch, self._a2a("REFUTES", None))

        assert evidence["verdict"] == "NOT_ENOUGH_INFO"

    def test_a_non_decisive_delegation_is_not_adopted(self, monkeypatch):
        evidence = self._run(monkeypatch, self._a2a("NOT_ENOUGH_INFO", None))

        assert evidence["verdict"] == "NOT_ENOUGH_INFO"


class TestCurrencyIsReadNotAssumed:
    """`ParsedClaim` has no currency field, so a EUR filing figure could not be
    compared against a claim without guessing a rate -- and the guard that
    prevented the guess also blocked "Apple was fined 500 million euros", where
    the claim states its currency in plain text. Read it instead."""

    def test_the_claims_currency_is_recognised(self):
        from finvet.tools.filing_amounts import currency_in_text

        assert currency_in_text("Apple was fined 500 million euros") == "EUR"
        assert currency_in_text("a fine of 500 million dollars") == "USD"
        assert currency_in_text("fined €500 million") == "EUR"
        assert currency_in_text("fined $2 billion") == "USD"

    def test_an_unstated_currency_is_not_invented(self):
        from finvet.tools.filing_amounts import currency_in_text

        assert currency_in_text("Apple was fined 500 million") is None
        assert currency_in_text("") is None

    def test_two_currencies_in_one_claim_are_ambiguous(self):
        from finvet.tools.filing_amounts import currency_in_text

        assert currency_in_text("fined €500 million, about $570 million") is None

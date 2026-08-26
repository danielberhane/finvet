"""A claim about what a filing says is answerable, and reaches a verdict.

Filing retrieval was built, indexed over 1,398 chunks and measured at the tool
boundary — and had never once produced evidence in a released response: 0 of
322 recorded executions carried a `rag` data source. Two things stood in the
way, and neither was in the retrieval code.

The first was period scoping (D17). The second is this: the parser classified
"Apple's annual report discusses risks from supplier concentration" as
`non_financial` and rejected it before any agent ran. An issuer's own report is
a verifiable source and its narrative sections are retrievable text, so that
classification was simply wrong.

Nothing was loosened to fix it. `verification_strategy_for` already returned
`filing_rag` for a sec claim with a null metric — the strategy existed and was
wired. `METRIC_WHITELIST` is vendored from the parser project and conformance
tested, so it is untouched. One rule (R2b) was added to parser_system.txt.

The trust boundary is unchanged and that is the point: these claims name no
number, so there is nothing for the numeric guard to demand. A claim that does
name a number still needs an XBRL fact or a market quote, and filing prose still
cannot become one.

Opt in with `-m integration`; skipped when the API is not reachable.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

TIMEOUT = 400


@pytest.fixture(scope="module")
def api_base_url():
    url = os.environ.get("FINVET_API_URL", "http://localhost:8000").rstrip("/")
    try:
        httpx.get(f"{url}/health", timeout=5)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"API not reachable at {url}: {exc}")
    return url


def _verify(url, claim):
    response = httpx.post(f"{url}/verify", json={"claim": claim}, timeout=TIMEOUT)
    assert response.status_code == 200, response.text
    return response.json()


def _sources(body):
    return ((body.get("metadata") or {}).get("data_sources") or {})


class TestAFilingClaimReachesAVerdict:
    """The whole point: an answer, not a badge on a queued review."""

    @pytest.mark.parametrize("claim", [
        "Apple's annual report discusses risks from supplier concentration",
        "Nvidia's 10-K describes dependence on a limited number of suppliers",
    ], ids=["aapl-suppliers", "nvda-suppliers"])
    def test_it_is_supported_by_retrieved_filing_text(self, api_base_url, claim):
        body = _verify(api_base_url, claim)

        assert body["verdict"] == "SUPPORTS", (
            f"got {body['verdict']} ({body.get('status')}) — this claim was "
            f"rejected as non_financial before R2b")
        assert body["status"] == "success", (
            "a claim the filing settles must terminate, not queue a reviewer")

    def test_the_verdict_cites_the_filing_it_read(self, api_base_url):
        body = _verify(
            api_base_url,
            "Apple's annual report discusses risks from supplier concentration")
        rag = _sources(body).get("rag") or {}

        assert rag.get("used") is True, "no RAG provenance reached the response"
        assert rag.get("chunks_retrieved", 0) > 0
        evidence = (rag.get("evidence") or [{}])[0]
        assert evidence.get("evidence_id"), "a passage with no content hash"
        assert evidence.get("filing_type"), "a passage with no filing identity"
        assert evidence.get("period_end"), "a passage with no period"

    def test_the_agent_actually_searched_the_filing(self, api_base_url):
        body = _verify(
            api_base_url,
            "Apple's annual report discusses risks from supplier concentration")

        assert "search_filing_text" in (
            (body.get("metadata") or {}).get("tools_called") or [])


class TestTheRejectBoundaryHolds:
    """R2b widens what counts as a filing claim. It must not widen what counts
    as a claim at all — these are the cases it could plausibly have swallowed."""

    @pytest.mark.parametrize("claim,reason", [
        ("The Eiffel Tower is located in Paris", "non_financial"),
        ("Apple's CEO enjoys sailing on weekends", "non_financial"),
        ("Should I buy Tesla stock right now?", "advice_seeking"),
        ("What was Apple's revenue in 2024?", "question"),
        ("Apple's revenue will reach $500 billion in 2030", "future_prediction"),
        ("Tesla is a better investment than Ford", "opinion_subjective"),
    ], ids=["eiffel", "ceo-hobby", "advice", "question", "forecast", "opinion"])
    def test_a_non_claim_is_still_rejected(self, api_base_url, claim, reason):
        body = _verify(api_base_url, claim)

        assert body["verdict"] == "REJECTED", (
            f"R2b swallowed something it should not have: {claim!r}")
        assert (body.get("metadata") or {}).get("reject_reason") == reason

    def test_a_company_fact_no_filing_addresses_is_not_a_filing_claim(
            self, api_base_url):
        """The line R2b draws, stated in the prompt as 'the test is whether a
        filing could settle it'. A CEO's hobbies are about the company and are
        in no filing."""
        body = _verify(api_base_url, "Apple's CEO enjoys sailing on weekends")

        assert body["verdict"] == "REJECTED"
        assert not _sources(body).get("rag")


class TestTheTrustBoundaryIsUnchanged:
    """R2b routes claims that need no number. It must not have made filing
    prose into a source of numbers."""

    def test_a_numeric_claim_still_rests_on_xbrl(self, api_base_url):
        body = _verify(
            api_base_url,
            "Apple's total revenue was $391 billion in fiscal year 2024")

        assert body["verdict"] == "SUPPORTS"
        observation = (body.get("metadata") or {}).get("trusted_observation") or {}
        assert observation.get("concept"), (
            "the verdict must still rest on an identified XBRL fact")

    def test_filing_text_is_still_supporting_evidence_only(self):
        from finvet.models.evidence import SUPPORTING_EVIDENCE_TOOLS

        assert "search_filing_text" in SUPPORTING_EVIDENCE_TOOLS

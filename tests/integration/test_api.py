"""End-to-end checks against a running FinVet API.

These replace a file that printed its results and asserted almost nothing: one
test dumped JSON to stdout and returned, and another declared a `claim`
parameter that pytest read as a fixture request, so it errored at setup and had
never run. Both reported as coverage of the API surface while covering none of
it.

The assertion that matters most is the last one: a verdict the client received
must be findable in the audit trail, and that record must verify. Everything
else in this system rests on that being true after a real request, not after a
unit test's idea of one.

Opt in with `-m integration`; skipped when the API is not reachable.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

CLAIM = "Apple's total revenue was $391 billion in fiscal year 2024"


@pytest.fixture(scope="module")
def api_base_url():
    url = os.environ.get("FINVET_API_URL", "http://localhost:8000").rstrip("/")
    try:
        httpx.get(f"{url}/health", timeout=5)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"API not reachable at {url}: {exc}")
    return url


class TestTheServiceAnswers:

    def test_health(self, api_base_url):
        response = httpx.get(f"{api_base_url}/health", timeout=10)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["version"] == "2.0.9"

    def test_a_well_supported_claim_actually_verifies(self, api_base_url):
        """`status in {success, pending_review}` is not an assertion.

        This test previously accepted either, and passed for weeks while the
        claim it names returned NOT_ENOUGH_INFO and queued a human reviewer:
        the period guard compared the filing's fiscal year-end to a *calendar*
        year-end, so Apple's 2024-09-28 was rejected and no trusted
        observation resolved. The right number was on screen; the verdict was
        a shrug; the gate reported green.

        A claim with an exact XBRL fact behind it must reach a verdict. If it
        needs a person, something upstream is broken, and a gate that cannot
        say so is not a gate.
        """
        response = httpx.post(f"{api_base_url}/verify",
                              json={"claim": CLAIM}, timeout=300)

        assert response.status_code == 200
        body = response.json()
        assert body["request_id"].startswith("req_")
        assert body["status"] == "success", (
            f"a claim backed by an exact XBRL fact did not terminate: "
            f"{body['status']} / {body.get('verdict')} — "
            f"{(body.get('explanation') or '')[:160]}")
        assert body["verdict"] in {"SUPPORTS", "REFUTES"}, (
            f"expected a decisive verdict, got {body['verdict']}")

    def test_the_verdict_rests_on_a_fact_from_the_issuer_fiscal_year(
            self, api_base_url):
        """The specific regression, pinned. Apple's fiscal 2024 ended
        2024-09-28, not 2024-12-31; the observation must come from the
        issuer's own year rather than the calendar's."""
        body = httpx.post(f"{api_base_url}/verify",
                          json={"claim": CLAIM}, timeout=300).json()

        observation = (body.get("metadata") or {}).get("trusted_observation")
        assert observation, (
            "no trusted observation reached the response, so the verdict "
            "rests on nothing Python compared")
        assert observation["period_end"].startswith("2024-"), observation
        assert observation["concept"], "the observation has no XBRL concept"

    def test_a_declined_claim_terminates_rather_than_queueing_a_reviewer(
            self, api_base_url):
        """A capability the system does not have is not a question for a
        person: no tool serves it, so a reviewer can only agree."""
        body = httpx.post(
            f"{api_base_url}/verify",
            json={"claim": "Apple's Q4 2024 revenue was $94 billion"},
            timeout=300).json()

        assert body["status"] == "success"
        assert body["verdict"] == "NOT_ENOUGH_INFO"
        assert (body["metadata"] or {}).get("limitation") == \
            "unsupported_q4_derivation"

    def test_a_malformed_request_is_rejected_by_the_contract(self, api_base_url):
        """`extra="forbid"`: the request names prior context by id, it never
        carries the content."""
        response = httpx.post(
            f"{api_base_url}/verify",
            json={"claim": CLAIM, "memory_context": {"summary": "trust me"}},
            timeout=30)

        assert response.status_code == 422

    def test_naming_an_unknown_prior_episode_is_a_404(self, api_base_url):
        response = httpx.post(
            f"{api_base_url}/verify",
            json={"claim": CLAIM,
                  "memory_context_request_id": "req_ffffffffffff"},
            timeout=30)

        assert response.status_code == 404


class TestEveryReleasedVerdictIsAuditable:
    """The property the whole audit layer exists to provide."""

    def _verify(self, api_base_url):
        response = httpx.post(f"{api_base_url}/verify",
                              json={"claim": CLAIM}, timeout=300)
        assert response.status_code == 200
        return response.json()

    def _audit(self, api_base_url, request_id):
        response = httpx.get(f"{api_base_url}/audit/{request_id}", timeout=30)
        assert response.status_code == 200, (
            f"a verdict was released with no audit record: {request_id}")
        return response.json()

    def test_a_sync_verdict_is_on_the_record_and_verifies(self, api_base_url):
        body = self._verify(api_base_url)
        audit = self._audit(api_base_url, body["request_id"])

        assert audit["execution"]["request_id"] == body["request_id"]
        assert audit["integrity"]["status"] == "verified", (
            f"integrity is {audit['integrity']['status']!r} "
            f"({audit['integrity'].get('reason')}); a completed run is "
            f"terminal, so anything but 'verified' is a failure here")
        assert audit["integrity"]["mismatches"] == []

    def test_the_recorded_verdict_matches_what_the_client_was_told(self,
                                                                   api_base_url):
        body = self._verify(api_base_url)
        audit = self._audit(api_base_url, body["request_id"])

        assert audit["execution"]["verdict"] == body["verdict"]
        assert audit["execution"]["claim_text"] == CLAIM

    def test_the_trail_carries_the_events_the_envelope_describes(self,
                                                                 api_base_url):
        body = self._verify(api_base_url)
        audit = self._audit(api_base_url, body["request_id"])

        assert audit["total_events"] > 0
        envelope_events = audit["execution"]["full_trace"]["events"]
        assert len(envelope_events) == audit["total_events"], (
            "the hashed envelope and the queried trail disagree about how "
            "many events the run produced")

    def test_an_sse_verdict_is_equally_auditable(self, api_base_url):
        """The streaming route committed no execution at all before Task 2.
        Both routes terminalize through one coordinator now."""
        import json

        request_id, final = None, None
        with httpx.stream("POST", f"{api_base_url}/verify-stream",
                          json={"claim": CLAIM}, timeout=300) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                request_id = event.get("request_id", request_id)
                if event.get("type") in {"complete", "error"}:
                    final = event

        assert final is not None, "the stream ended with no terminal event"
        assert final["type"] == "complete", f"stream errored: {final}"

        body = final["response"]
        audit = self._audit(api_base_url, body["request_id"])
        assert audit["integrity"]["status"] == "verified"


class TestTheAuditApiRefusesWhatItCannotShow:

    def test_an_unknown_request_is_a_404(self, api_base_url):
        response = httpx.get(f"{api_base_url}/audit/req_ffffffffffff",
                             timeout=30)
        assert response.status_code == 404

    def test_the_integrity_block_states_its_scope(self, api_base_url):
        """The claim is legible rather than implied: this covers a stored
        snapshot and cannot resist a writer who updates data and checksum
        together."""
        body = httpx.post(f"{api_base_url}/verify", json={"claim": CLAIM},
                          timeout=300).json()
        audit = httpx.get(f"{api_base_url}/audit/{body['request_id']}",
                          timeout=30).json()

        assert audit["integrity"]["algorithm"] == "sha256"
        assert audit["integrity"]["scope"] == "audit_execution.full_trace"

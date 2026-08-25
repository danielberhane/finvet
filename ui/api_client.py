"""HTTP wrapper for the FinVet API."""

import json
import os
import requests

API_BASE_URL = os.environ.get("FINVET_API_URL", "http://localhost:8000")


def health_check():
    """Check API health. Returns (connected: bool, version: str)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json().get("version", "")
    except Exception:
        pass
    return False, ""


def get_pending_reviews():
    """Fetch pending HITL reviews. Returns list or empty list on failure."""
    try:
        resp = requests.get(f"{API_BASE_URL}/reviews", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def get_reviews_detailed():
    """Fetch pending reviews with full details. Returns (reviews, error_msg)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/reviews", timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"Failed to fetch pending reviews (Status: {resp.status_code})"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to FinVet API. Make sure the server is running on port 8000."
    except Exception as e:
        return None, f"Error loading pending reviews: {e}"


def submit_review(request_id, payload):
    """Submit a HITL review decision. Returns (response, error_msg)."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/review/{request_id}",
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"Failed to submit review: {resp.text}"
    except Exception as e:
        return None, f"Error submitting review: {e}"


def verify_claim(claim_text, memory_context_request_id=None):
    """Run claim verification. Returns (response_obj, error_type).

    error_type is None on success, or one of: 'guardrail', 'api_error',
    'timeout', 'connection'.
    """
    payload = {"claim": claim_text}
    # Only the id crosses the API boundary. The server reads the episode from
    # its own store, so nothing this client composes reaches the agent prompt.
    if memory_context_request_id:
        payload["memory_context_request_id"] = memory_context_request_id
    try:
        resp = requests.post(
            f"{API_BASE_URL}/verify",
            json=payload,
            timeout=120
        )
        return resp, None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, "connection"


def verify_claim_stream(claim_text, memory_context_request_id=None):
    """Run claim verification with SSE streaming.

    Yields dicts with type: 'progress', 'complete', 'done', or 'error'.
    Falls back to non-streaming verify_claim() on connection failure.
    """
    payload = {"claim": claim_text}
    # Only the id crosses the API boundary. The server reads the episode from
    # its own store, so nothing this client composes reaches the agent prompt.
    if memory_context_request_id:
        payload["memory_context_request_id"] = memory_context_request_id
    try:
        resp = requests.post(
            f"{API_BASE_URL}/verify-stream",
            json=payload,
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    yield event
                except json.JSONDecodeError:
                    continue
    except Exception:
        # Fallback to non-streaming
        resp, error_type = verify_claim(claim_text, memory_context_request_id)
        if error_type:
            yield {"type": "error", "message": error_type}
        elif resp and resp.status_code == 200:
            yield {"type": "complete", "response": resp.json()}
            yield {"type": "done"}
        elif resp and resp.status_code == 400:
            yield {"type": "guardrail", "response": resp.json()}
        else:
            yield {"type": "error", "message": f"API error: {resp.status_code if resp else 'no response'}"}


def memory_check(claim_text):
    """Check for similar past verifications. Returns list of matches or None."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/memory-check",
            json={"claim": claim_text},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("matches", [])
    except Exception:
        pass
    return None


def memory_accept(original_request_id, claim, similarity):
    """Log acceptance of a cached memory result."""
    try:
        requests.post(
            f"{API_BASE_URL}/memory-accept",
            json={
                "original_request_id": original_request_id,
                "claim": claim,
                "similarity": similarity,
            },
            timeout=5
        )
    except Exception:
        pass


def list_audit_executions(
    verdict=None, agent=None, date_from=None, date_to=None,
    data_source=None, limit=50, offset=0
):
    """Fetch audit executions with optional filters. Returns (data, error_msg)."""
    params = {"limit": limit, "offset": offset}
    if verdict:
        params["verdict"] = verdict
    if agent:
        params["agent"] = agent
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if data_source:
        params["data_source"] = data_source
    try:
        resp = requests.get(f"{API_BASE_URL}/audit", params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Failed to fetch audit records (Status: {resp.status_code})"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to FinVet API. Make sure the server is running on port 8000."
    except Exception as e:
        return None, f"Error loading audit records: {e}"


def get_audit_detail(request_id):
    """Fetch full audit trail for one request. Returns (data, error_msg)."""
    try:
        resp = requests.get(f"{API_BASE_URL}/audit/{request_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        if resp.status_code == 404:
            return None, f"No audit trail found for {request_id}"
        return None, f"Failed to fetch audit detail (Status: {resp.status_code})"
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to FinVet API. Make sure the server is running on port 8000."
    except Exception as e:
        return None, f"Error loading audit detail: {e}"

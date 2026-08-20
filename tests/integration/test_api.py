"""Integration checks against a running FinVet API on :8000.

Opt in with:  pytest -m integration
"""

import pytest
import httpx

pytestmark = pytest.mark.integration
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_verify_claim(claim: str):
    """Test claim verification."""
    print("Testing claim verification...")
    print(f"Claim: {claim}")
    print()

    response = httpx.post(
        f"{BASE_URL}/verify",
        json={"claim": claim, "user_id": "test_user"},
        timeout=60.0  # Allow up to 60 seconds
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*60)
        print(f"VERDICT: {result['verdict']}")
        print(f"CONFIDENCE: {result['confidence']:.2f} ({result['confidence_label']})")
        print("="*60)
        print(f"\nSUMMARY: {result['summary']}")
        print(f"\nEXPLANATION: {result['explanation']}")

        if result.get('sources'):
            print(f"\nSOURCES ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. {source.get('description', 'Unknown')} - {source.get('url', 'N/A')}")

        if result.get('disclosures'):
            print("\nDISCLOSURES:")
            for disclosure in result['disclosures']:
                print(f"  - {disclosure}")

        if result.get('metadata'):
            meta = result['metadata']
            print("\nMETADATA:")
            print(f"  Agents used: {', '.join(meta.get('agents_used', []))}")
            print(f"  Execution time: {meta.get('execution_time_ms', 0)}ms")
            print(f"  Tokens used: {meta.get('total_tokens_used', 0)}")

    else:
        print(f"Error: {response.text}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test health
    test_health()

    # Test claims
    claims = [
        "Tesla Q3 2024 revenue was $25.18 billion",
        "Apple Q4 2023 revenue was $119.58 billion",
        "Microsoft beat earnings estimates by 15%",
    ]

    for claim in claims:
        test_verify_claim(claim)

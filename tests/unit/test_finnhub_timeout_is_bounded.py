"""A slow market feed must not consume the agent's whole iteration budget.

Finnhub returned read timeouts during a benchmark run. The client waited 30s
per call, the tool reported an error, the agent tried again, and three claims
took 600s each -- half an hour of a 100-row run spent waiting on one feed.

The timeout is the multiplier: whatever it is, the agent can spend it once per
iteration. Keeping it well under the per-claim budget is what bounds the cost.
"""

from finvet.config.constants import AGENT_MAX_ITERATIONS, FINNHUB_TIMEOUT_SECONDS
from finvet.mcp.finnhub import FinnhubClient


class TestTheClientDoesNotWaitLong:

    def test_the_configured_timeout_is_used(self):
        """Read from the client rather than the constant: the constant being
        right means nothing if the client still hardcodes its own."""
        client = FinnhubClient(api_key="k", mock_mode=False)

        assert client._client.timeout.read == FINNHUB_TIMEOUT_SECONDS

    def test_a_full_retry_loop_stays_within_the_claim_budget(self):
        """The agent may call a failing tool once per iteration, so the worst
        case is the timeout times the budget. At 30s that was 150s of waiting
        for one feed, before any model latency."""
        worst_case = FINNHUB_TIMEOUT_SECONDS * AGENT_MAX_ITERATIONS

        assert worst_case <= 60, (
            f"a dead feed can cost {worst_case}s of pure waiting per claim")

"""A run must be attributable to the model that actually served it.

`scripts/run_golden.py` posts claims to the API over HTTP and then recorded
`active_llm_config()` -- read from its **own** process. That answers "what was
this shell launched with", not "what ran the claims", and the two came apart:
a benchmark launched with the MiniMax env against an API still holding the
DeepSeek one stamped 56 DeepSeek rows `MiniMax-M2.7`. Nothing in the artifact
contradicted it, so the cross-model comparison ran on it and was wrong.

The fix is that only the serving process may say what it serves. These drive
the route callable rather than asserting on a hand-built dict, because the
defect was precisely that a plausible dict was believed.
"""

import asyncio

from finvet.api.models import HealthResponse
from finvet.api.routes.health import health


class TestHealthNamesTheModelBehindEachRole:

    def test_every_role_is_reported(self):
        body = asyncio.run(health())

        assert set(body["llm"]) == {"parser", "agent", "verdict"}, (
            "a run may point roles at different providers, so one model name "
            "for the whole process is not enough to attribute it")

    def test_each_role_carries_model_and_endpoint(self):
        """A model name alone is ambiguous: the same name served by a hosted
        API and by a local gateway are not the same run."""
        body = asyncio.run(health())

        for role, cfg in body["llm"].items():
            assert cfg.get("model"), f"{role} reported no model"
            assert "base_url" in cfg, f"{role} reported no endpoint"

    def test_the_response_model_accepts_it(self):
        """`response_model=HealthResponse` silently drops undeclared keys, which
        would strip the attribution back out on the way to the client."""
        body = asyncio.run(health())

        assert HealthResponse(**body).llm == body["llm"]


class TestNoCredentialIsExposed:

    def test_health_carries_no_api_key(self):
        """/health is unauthenticated. It reports configuration, never secrets."""
        body = asyncio.run(health())

        flat = repr(body).lower()
        for banned in ("api_key", "sk-", "secret", "token"):
            assert banned not in flat, f"/health leaked {banned!r}"

"""The capability profile Release A actually ships.

Claim memory stays in the repository and stays off. It is the one subsystem
whose output is prior *model* output rather than a source: reusing a cached
verdict, or feeding a past summary back into an agent, moves something the
system said into the position of something the system found. That may be worth
studying; it is not worth shipping on by default, and the difference has to be
enforced by the default rather than described in a document.

These tests pin the default and the paths that depend on it. Enabling the
feature explicitly is still supported — `ENABLE_CLAIM_MEMORY=true` — and is
what makes this "experimental" rather than "removed".
"""

import pytest

from finvet.config.settings import Settings

# Three secrets are required and normally come from .env. Supply them so these
# tests measure the flags they are about and nothing else.
_REQUIRED = {"deepseek_api_key": "x", "tavily_api_key": "x",
             "postgres_password": "x"}


def _settings(**overrides):
    return Settings(_env_file=None, **{**_REQUIRED, **overrides})


class TestClaimMemoryIsOffByDefault:

    def test_claim_memory_is_experimental_and_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("ENABLE_CLAIM_MEMORY", raising=False)
        assert _settings().enable_claim_memory is False

    def test_it_can_still_be_enabled_explicitly(self, monkeypatch):
        """Disabled by default, not removed. The distinction is the whole
        meaning of "experimental"."""
        monkeypatch.delenv("ENABLE_CLAIM_MEMORY", raising=False)
        assert _settings(enable_claim_memory=True).enable_claim_memory is True

    def test_the_env_var_still_switches_it_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CLAIM_MEMORY", "true")
        assert _settings().enable_claim_memory is True

    def test_the_implementation_is_still_present(self):
        """Off, not deleted: the module and both routes remain importable."""
        import importlib

        importlib.import_module("finvet.memory.store_service")
        routes = importlib.import_module("finvet.api.routes.memory")
        assert hasattr(routes, "memory_check")
        assert hasattr(routes, "memory_accept")


class TestTheDisabledPathIsSilentNotBroken:
    """With memory off, the endpoints answer normally with nothing to offer.

    An error here would push a disabled experiment into the user's way on every
    verification.
    """

    def test_disabled_claim_memory_returns_no_matches(self, monkeypatch):
        from finvet.api import deps
        from finvet.api.models import MemoryCheckRequest
        from finvet.api.routes.memory import memory_check

        monkeypatch.setattr(deps, "claim_memory", None)
        assert memory_check(
            MemoryCheckRequest(claim="Apple revenue was $391B")) == {"matches": []}

    def test_a_named_prior_episode_is_a_404_not_a_silent_pass(self):
        """Asking to reuse a specific verification while memory is disabled is
        a request the server cannot honour. Proceeding without the context
        would verify a different question than the one submitted."""
        from fastapi import HTTPException

        from finvet.api.execution import resolve_memory_context

        with pytest.raises(HTTPException) as excinfo:
            resolve_memory_context(None, "req_0123456789ab")
        assert excinfo.value.status_code == 404

    def test_no_context_requested_is_simply_no_context(self):
        from finvet.api.execution import resolve_memory_context

        assert resolve_memory_context(None, None) is None


class TestTheDefaultUiOffersNoMemoryChoice:
    """An empty result must route straight to verification.

    The decision lives inside a Streamlit callback, so the branch is pinned by
    reading it rather than by mocking Streamlit internals. The rendered result
    is a Task 10 manual check.
    """

    def test_an_empty_match_list_is_falsy(self):
        """What the client returns when the API says {"matches": []}, and what
        the UI branch tests."""
        assert not []

    def test_the_view_routes_an_empty_result_to_verification(self):
        from pathlib import Path

        source = Path("ui/views/verify.py").read_text()

        assert "matches = memory_check(claim)" in source
        # The branch order matters: a non-empty list shows the card, anything
        # else goes straight to the pipeline.
        assert "elif matches:" in source, (
            "the view does not distinguish an empty result from a populated "
            "one, so a disabled feature could still render its card")
        assert "_run_verification(claim)" in source

    def test_the_client_turns_an_empty_payload_into_an_empty_list(self):
        """Not None: None means the API could not be reached, which the view
        treats differently."""
        from pathlib import Path

        source = Path("ui/api_client.py").read_text()
        assert 'resp.json().get("matches", [])' in source


def _without_the_optional_extra(monkeypatch):
    """Simulate a default install: the optional package cannot be imported."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langgraph.store.postgres"):
            raise ImportError("No module named 'langgraph.store.postgres'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _reimport_main(monkeypatch):
    import importlib
    import sys

    for mod in [m for m in list(sys.modules) if m.startswith("finvet.main")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    return importlib.import_module("finvet.main")


class TestTheApiBootsWithoutTheOptionalExtra:
    """`langgraph-checkpoint-postgres` is an optional dependency, and claim
    memory ships disabled. A module-scope import would stop the API booting on
    a default install for a feature that install does not use.

    Both directions matter: absent-and-disabled must boot silently, and
    enabled-but-absent must degrade to disabled rather than crash — that
    degradation is what the architecture documentation promises.
    """

    @staticmethod
    def _set_memory(monkeypatch, enabled):
        """`settings` is a module-level singleton read at import time, so the
        env var is already baked in. Patch the attribute the branch reads."""
        from finvet.config import settings as settings_module

        monkeypatch.setattr(settings_module.settings, "enable_claim_memory",
                            enabled, raising=False)

    def test_it_imports_with_memory_disabled_and_the_package_absent(
            self, monkeypatch):
        self._set_memory(monkeypatch, False)
        _without_the_optional_extra(monkeypatch)

        main = _reimport_main(monkeypatch)      # must not raise

        assert main.claim_store is None

    def test_enabling_it_without_the_package_degrades_to_disabled(
            self, monkeypatch, caplog):
        import logging

        self._set_memory(monkeypatch, True)
        _without_the_optional_extra(monkeypatch)

        with caplog.at_level(logging.WARNING):
            main = _reimport_main(monkeypatch)  # must not raise

        assert main.claim_store is None
        assert "memory" in caplog.text.lower(), (
            "degrading silently hides a configuration the operator asked for")

    def test_the_import_is_not_at_module_scope(self):
        """The specific hazard: moving the dependency to an optional group
        while the import stays unconditional stops the API booting."""
        from pathlib import Path

        source = Path("src/finvet/main.py").read_text()
        header = source.split("if settings.enable_claim_memory")[0]

        assert "from langgraph.store.postgres import PostgresStore" not in header, (
            "PostgresStore is imported before the feature gate")

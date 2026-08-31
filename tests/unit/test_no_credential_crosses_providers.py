"""One provider's key must never be sent to another provider's endpoint.

`create_llm` read the key as:

    os.environ.get(config.api_key_env, "") or settings.deepseek_api_key

The fallback exists so a `.env` carrying only `DEEPSEEK_API_KEY` works, which is
harmless while every role points at DeepSeek. It stops being harmless the moment
a role is pointed somewhere else:

    LLM_AGENT__BASE_URL=http://<gateway>/v1
    LLM_AGENT__API_KEY_ENV=LITELLM_API_KEY     # and LITELLM_API_KEY unset

The agent then posts the **DeepSeek credential to the MiniMax gateway**. The
symptom is an upstream 401, which reads as a network or allowlist problem, and
the actual cause -- a credential sent to a third party -- is invisible.

Two things follow. A missing key must raise, naming the variable to set. And the
error must arrive before the request, not as somebody else's 401.

`finvet-min-max-test-new/src/finvet/llm/factory.py` already made this choice:
"a silent substitution sends one provider's credential to another provider's
base_url, which surfaces only as a confusing upstream 401."
"""

import pytest

from finvet.llm.factory import create_llm


class TestAMissingKeyRaisesRatherThanSubstituting:

    def test_the_deepseek_key_is_not_sent_to_another_endpoint(self, monkeypatch):
        """The case that motivated this: a gateway configured, its key absent."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret")
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setattr("finvet.llm.factory.settings.llm_agent.api_key_env",
                            "LITELLM_API_KEY")
        monkeypatch.setattr("finvet.llm.factory.settings.llm_agent.base_url",
                            "http://gateway.internal/v1")

        with pytest.raises(RuntimeError) as exc:
            create_llm("agent")

        assert "LITELLM_API_KEY" in str(exc.value), (
            "the error must name the variable to set")
        assert "sk-deepseek-secret" not in str(exc.value), (
            "a credential must never appear in an error message")

    def test_the_message_names_the_role_and_the_model(self, monkeypatch):
        """A reader has three roles to check; the error should say which."""
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        monkeypatch.setattr("finvet.llm.factory.settings.llm_parser.api_key_env",
                            "LITELLM_API_KEY")

        with pytest.raises(RuntimeError) as exc:
            create_llm("parser")

        message = str(exc.value)
        assert "parser" in message
        assert "deepseek-chat" in message      # whatever model was configured

    @pytest.mark.parametrize("purpose", ["parser", "agent", "verdict"])
    def test_every_role_is_guarded(self, purpose, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        with pytest.raises(RuntimeError):
            create_llm(purpose)


class TestAnEmptyKeyIsAMissingKey:

    def test_an_empty_string_does_not_pass_as_a_credential(self, monkeypatch):
        """`ChatOpenAI` accepts "" and fails later at the provider, which is the
        same confusing 401 by another route."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")

        with pytest.raises(RuntimeError):
            create_llm("agent")


class TestTheNormalPathStillWorks:

    def test_a_configured_key_builds_a_client(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-present")

        llm = create_llm("agent")

        assert llm.model_name == "deepseek-chat"

    def test_a_per_role_key_is_read_from_its_own_variable(self, monkeypatch):
        """Switching one role to another provider is the supported path and
        must keep working -- it is what the guard exists to make safe."""
        monkeypatch.setenv("LITELLM_API_KEY", "sk-gateway")
        monkeypatch.setattr("finvet.llm.factory.settings.llm_parser.api_key_env",
                            "LITELLM_API_KEY")
        monkeypatch.setattr("finvet.llm.factory.settings.llm_parser.model",
                            "MiniMax-M2.7")

        llm = create_llm("parser")

        assert llm.model_name == "MiniMax-M2.7"

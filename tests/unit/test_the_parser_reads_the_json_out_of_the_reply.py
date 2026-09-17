"""The parser node finds the JSON object in whatever the model wrapped it in.

The node used to strip markdown fences and hand the remainder to json.loads,
which is exactly what DeepSeek returns and nothing else. The next parser is a
fine-tuned open-weight model served locally, and Qwen-family bases emit a
<think>...</think> block before the object unless the chat template is told
not to (observed in the claim-parser project: 0% JSON validity until
enable_thinking=False). A sentence of prose either side is the other common
wrapper. Both are tolerated now; a reply with no decodable object still
raises ParsingError, and nothing inside the object is repaired.

Drives the node, not the helper: a fake model returns the wrapped reply and
the assertion is on the ParsedClaim the node emits.
"""

import importlib

import pytest

from finvet.utils.exceptions import ParsingError

# The package re-exports the node *function* under the same name; the module
# is what monkeypatch needs.
node = importlib.import_module("finvet.graph.nodes.claim_parser")

_APPLE = ('{"claim_type": "sec", "ticker": "AAPL", "metric": "revenue", '
          '"operator": "eq", "value": 391000000000, "period": "fiscal 2024", '
          '"reject_reason": null}')


class _Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"total_tokens": 42}


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return _Reply(self._content)


@pytest.fixture
def parse(monkeypatch):
    class _Audit:
        def log_event(self, **kwargs):
            pass
    monkeypatch.setattr(node, "get_audit_logger", lambda: _Audit())

    def _run(content):
        monkeypatch.setattr(node, "create_llm", lambda role: _FakeLLM(content))
        out = node.claim_parser({"request_id": "r",
                                 "claim_raw": "Apple's fiscal 2024 revenue was $391 billion"})
        return out["parsed_claim"]
    return _run


class TestWrappersAreTolerated:

    def test_bare_json(self, parse):
        assert parse(_APPLE).ticker == "AAPL"

    def test_markdown_fences(self, parse):
        assert parse(f"```json\n{_APPLE}\n```").value == 391e9

    def test_a_leading_think_block(self, parse):
        reply = "<think>\nThe claim names Apple and a revenue figure.\n</think>\n" + _APPLE
        assert parse(reply).metric == "revenue"

    def test_prose_around_the_object(self, parse):
        assert parse(f"Sure! Here is the parse:\n{_APPLE}\nHope this helps.").period == "fiscal 2024"

    def test_nested_braces_do_not_cut_the_object_short(self, parse):
        """raw_decode, not a regex: a brace inside a string is not the end."""
        reply = _APPLE.replace('"fiscal 2024"', '"fiscal 2024 {as filed}"')
        assert parse(reply).period == "fiscal 2024 {as filed}"


class TestNothingIsRepaired:

    def test_no_object_at_all_still_fails(self, parse):
        with pytest.raises(ParsingError):
            parse("I could not parse that claim.")

    def test_a_broken_object_still_fails(self, parse):
        with pytest.raises(ParsingError):
            parse('{"claim_type": "sec", "ticker": ')

    def test_a_json_array_is_not_an_object(self, parse):
        with pytest.raises(ParsingError):
            parse("[1, 2, 3]")

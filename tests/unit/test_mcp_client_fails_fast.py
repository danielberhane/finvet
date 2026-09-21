"""An MCP server that has stopped answering is asked once, not once per tool.

Two findings from one night (issue #4). With the SEC EDGAR MCP down, every
tool call waited the full 60 s timeout and the agent kept calling: seven
timeouts, seven minutes, for the answer the first timeout already gave. And
after the server was restarted, the client kept sending the session id the
server had forgotten, so every call was an instant 404 until the API was
restarted too.

The transport is faked at the HTTP layer so the real `MCPClient` runs.
"""

import json
import sys

import httpx
import pytest

sys.path.insert(0, "src")

from finvet.mcp.mcp_client import MCPClient, MCPError  # noqa: E402


def _rpc(request: httpx.Request) -> dict:
    return json.loads(request.content)


def _tool_result(request_id, payload) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": request_id,
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}}
    return httpx.Response(200, json=body)


def _init_ok(session_id: str) -> httpx.Response:
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": "init", "result": {}},
                          headers={"mcp-session-id": session_id})


class _Server:
    """Scripted MCP server: records every request, answers per `handle`."""

    def __init__(self, handle):
        self.requests = []
        self._handle = handle
        self.transport = httpx.MockTransport(self._respond)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handle(request, len(self.requests))

    def tool_calls(self):
        return [r for r in self.requests if _rpc(r).get("method") == "tools/call"]


def _client(server: _Server, timeout: float = 5.0) -> MCPClient:
    client = MCPClient("http://mcp.test", timeout=timeout)
    client._client = httpx.Client(transport=server.transport,
                                  timeout=httpx.Timeout(timeout))
    return client


class TestATimedOutServerIsNotAskedAgain:

    def _dead_server(self):
        def handle(request, n):
            rpc = _rpc(request)
            if rpc.get("method") == "initialize":
                return _init_ok("s1")
            if rpc.get("method") == "notifications/initialized":
                return httpx.Response(202)
            raise httpx.ReadTimeout("server never answered", request=request)
        return _Server(handle)

    def test_the_second_call_fails_without_contacting_the_server(self):
        server = self._dead_server()
        client = _client(server)

        with pytest.raises(MCPError, match="timed out"):
            client.call_tool("get_company_info", {"identifier": "INTC"})
        asked_before = len(server.tool_calls())

        with pytest.raises(MCPError, match="timed out"):
            client.call_tool("get_recent_filings", {"cik": "50863"})

        assert asked_before == 1
        assert len(server.tool_calls()) == 1, (
            "a server that just timed out was asked again")

    def test_the_server_is_probed_again_once_the_window_has_passed(self, monkeypatch):
        server = self._dead_server()
        client = _client(server, timeout=5.0)
        now = [1000.0]
        monkeypatch.setattr("time.monotonic", lambda: now[0])

        with pytest.raises(MCPError):
            client.call_tool("get_company_info", {"identifier": "INTC"})
        now[0] += 5.0 + 1

        with pytest.raises(MCPError):
            client.call_tool("get_company_info", {"identifier": "INTC"})

        assert len(server.tool_calls()) == 2


class TestAStaleSessionIsReinitialised:

    def test_a_404_reinitialises_and_retries_once(self):
        state = {"session": None}      # restarted: knows no session yet

        def handle(request, n):
            rpc = _rpc(request)
            if rpc.get("method") == "initialize":
                state["session"] = "new"
                return _init_ok("new")
            if rpc.get("method") == "notifications/initialized":
                return httpx.Response(202)
            if request.headers.get("mcp-session-id") != state["session"]:
                return httpx.Response(404, json={"jsonrpc": "2.0", "id": "server-error",
                                                 "error": {"code": -32001,
                                                           "message": "Session not found"}})
            return _tool_result(rpc["id"], {"success": True, "cik": "0000050863"})

        server = _Server(handle)
        client = _client(server)
        client.session_id = "old"          # what the server has forgotten

        result = client.call_tool("get_company_info", {"identifier": "INTC"})

        assert result == {"success": True, "cik": "0000050863"}
        assert client.session_id == "new"
        assert len(server.tool_calls()) == 2
        inits = [r for r in server.requests if _rpc(r).get("method") == "initialize"]
        assert len(inits) == 1

    def test_a_404_after_reinitialising_is_reported_not_retried_forever(self):
        def handle(request, n):
            rpc = _rpc(request)
            if rpc.get("method") == "initialize":
                return _init_ok("s2")
            if rpc.get("method") == "notifications/initialized":
                return httpx.Response(202)
            return httpx.Response(404, json={"jsonrpc": "2.0", "id": "server-error",
                                             "error": {"code": -32001,
                                                       "message": "Session not found"}})

        server = _Server(handle)
        client = _client(server)
        client.session_id = "old"

        with pytest.raises(MCPError, match="404"):
            client.call_tool("get_company_info", {"identifier": "INTC"})

        assert len(server.tool_calls()) == 2


# --- through the producer -------------------------------------------------

from typing import List  # noqa: E402

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from finvet.graph.nodes import domain_agents  # noqa: E402
from finvet.mcp.sec_edgar import SECEdgarClient  # noqa: E402
from finvet.models.claim import ParsedClaim  # noqa: E402
from finvet.tools import sec_tools  # noqa: E402


class _ScriptedModel(BaseChatModel):
    """Says what DeepSeek said that night: the same tool, again and again."""

    script: List[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _asks_for(company, n):
    return AIMessage(content="", tool_calls=[
        {"id": f"call{n}", "name": "get_company_info",
         "args": {"ticker_or_name": company}}])


class TestARunAgainstADeadServer:

    def test_asks_the_server_once_and_still_fails_closed(self, monkeypatch):
        def handle(request, n):
            rpc = _rpc(request)
            if rpc.get("method") == "initialize":
                return _init_ok("s1")
            if rpc.get("method") == "notifications/initialized":
                return httpx.Response(202)
            raise httpx.ReadTimeout("server never answered", request=request)
        server = _Server(handle)
        edgar = SECEdgarClient(base_url="http://mcp.test")
        edgar._mcp = _client(server)
        sec_tools._set_client(edgar)

        model = _ScriptedModel(script=[
            _asks_for("INTC", 1), _asks_for("INTC", 2), _asks_for("Intel", 3),
            AIMessage(content="I could not retrieve the filing."),
        ])
        monkeypatch.setattr("finvet.agents.base.create_llm", lambda role: model)

        state = {
            "request_id": "req_test",
            "claim_raw": "Intel's total revenue was 250 billion in fiscal year 2023",
            "parsed_claim": ParsedClaim(
                claim_type="sec", ticker="INTC", metric="revenue", operator="eq",
                value=250_000_000_000.0, period="fiscal year 2023", reject_reason=None),
            "total_tokens_used": 0,
        }

        out = domain_agents.run_sec_agent(state)
        evidence = out["agent_evidence"]

        assert evidence["tools_called"] == ["get_company_info"] * 3
        assert len(server.tool_calls()) == 1, (
            f"the dead server was asked {len(server.tool_calls())} times")
        assert evidence["verdict"] == "NOT_ENOUGH_INFO"
        assert evidence["confidence"] <= 0.5

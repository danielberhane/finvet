"""Generic MCP client for streamable-http transport."""

import json
import uuid
from typing import Any, Dict, Optional

import httpx

from .. import __version__
from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MCPError(Exception):
    """Raised when an MCP server returns an error or is unreachable."""
    pass


class MCPClient:
    """Reusable JSON-RPC client for any MCP server using streamable-http transport.

    Handles session initialization, session-id tracking, and SSE response parsing.
    All calls are synchronous (httpx.Client).
    """

    def __init__(self, base_url: str, timeout: Optional[float] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = settings.sec_mcp_timeout_s if timeout is None else timeout
        self.session_id: Optional[str] = None
        self._client = httpx.Client(timeout=httpx.Timeout(self.timeout))

    def initialize(self) -> None:
        """Send JSON-RPC 'initialize' + 'notifications/initialized' to start a session."""
        payload = {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "finvet", "version": __version__},
            },
        }
        try:
            resp = self._client.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Accept": "text/event-stream, application/json"},
            )
            resp.raise_for_status()
            self.session_id = resp.headers.get("mcp-session-id")
            if self.session_id:
                logger.info(f"MCP session initialized: {self.session_id}")
            else:
                logger.warning(f"No session ID returned from {self.base_url}")

            # Send initialized notification
            notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            headers = {"Accept": "text/event-stream, application/json"}
            if self.session_id:
                headers["mcp-session-id"] = self.session_id
            self._client.post(f"{self.base_url}/mcp", json=notif, headers=headers)

        except httpx.ConnectError:
            raise MCPError(f"Cannot connect to MCP server at {self.base_url}")
        except httpx.TimeoutException:
            raise MCPError(f"Timeout connecting to MCP server at {self.base_url}")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool via JSON-RPC tools/call and return the parsed result."""
        if self.session_id is None:
            self.initialize()

        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        headers = {"Accept": "text/event-stream, application/json"}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id

        try:
            resp = self._client.post(
                f"{self.base_url}/mcp", json=payload, headers=headers
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise MCPError(f"Cannot connect to MCP server at {self.base_url}")
        except httpx.TimeoutException:
            raise MCPError(f"Tool call '{name}' timed out ({self.timeout}s)")
        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP {e.response.status_code} from MCP server: {e.response.text}")

        return self._parse_response(resp.text, request_id, name)

    def _parse_response(self, text: str, request_id: str, tool_name: str) -> Any:
        """Parse SSE or plain JSON-RPC response."""
        # Try SSE first (lines starting with "data:")
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if event.get("id") != request_id:
                continue
            if "error" in event:
                err = event["error"]
                raise MCPError(f"MCP error: {err.get('message', 'Unknown')} (code: {err.get('code')})")
            return self._extract_content(event.get("result"))

        # Fall back to plain JSON-RPC
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            logger.exception(f"Unparseable MCP response from '{tool_name}'")
            raise MCPError(f"Unparseable response from '{tool_name}': {text[:200]}")
        if "error" in event:
            err = event["error"]
            raise MCPError(f"MCP error: {err.get('message', 'Unknown')} (code: {err.get('code')})")
        return self._extract_content(event.get("result"))

    @staticmethod
    def _extract_content(result: Any) -> Any:
        """Unwrap MCP tool result content array → parsed dict/str."""
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return content
        return result

    def close(self) -> None:
        self._client.close()

"""The sync/async bridge in ``McpSyncClient``.

The client runs its MCP session on a private thread and event loop, so the
failure modes worth pinning are lifecycle ones: a handshake that fails must
raise instead of hanging, ``close`` must be safe to call twice, and a session
that dies must fail in-flight callers rather than leaving them blocked until
the timeout.

The MCP SDK is an optional dependency, so these tests substitute a fake
transport and a fake ``ClientSession`` — which also lets them exercise the
error paths a live server would rarely produce on demand.
"""

from __future__ import annotations

import contextlib
import sys
from unittest.mock import patch

import pytest

from tradingagents.brokers.errors import BrokerConnectionError, BrokerNotConfiguredError
from tradingagents.brokers.mcp_client import McpSyncClient


class FakeTool:
    def __init__(self, name, description="", input_schema=None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


class FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class FakeCallToolResult:
    def __init__(self, structured=None):
        self.content = []
        self.structuredContent = structured
        self.isError = False


class FakeSession:
    """Minimal stand-in for ``mcp.ClientSession``."""

    instances: list[FakeSession] = []

    def __init__(self, read_stream, write_stream, **_kwargs):
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.initialized = False
        self.exited = False
        self.calls = []
        FakeSession.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        self.exited = True
        return False

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return FakeListToolsResult([FakeTool("get_positions")])

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        if name == "boom":
            raise RuntimeError("tool exploded")
        return FakeCallToolResult(structured={"echo": name})


@contextlib.asynccontextmanager
async def _fake_transport(*_args, **_kwargs):
    yield ("read", "write")


@contextlib.asynccontextmanager
async def _failing_transport(*_args, **_kwargs):
    raise OSError("connection refused")
    yield  # pragma: no cover - unreachable, keeps this an async generator


@contextlib.contextmanager
def _patched(transport=_fake_transport, session=FakeSession):
    FakeSession.instances.clear()
    with (
        patch("tradingagents.brokers.mcp_client._open_transport", transport),
        patch("tradingagents.brokers.mcp_client._require_sdk", return_value=session),
    ):
        yield


@pytest.fixture()
def client():
    instance = McpSyncClient("https://example.invalid/mcp", timeout=5.0)
    yield instance
    instance.close()


@pytest.mark.unit
def test_connect_performs_the_handshake(client):
    with _patched():
        client.connect()
        assert FakeSession.instances[0].initialized is True


@pytest.mark.unit
def test_connect_is_idempotent(client):
    with _patched():
        client.connect()
        client.connect()
        assert len(FakeSession.instances) == 1


@pytest.mark.unit
def test_list_tools_returns_specs(client):
    with _patched():
        tools = client.list_tools()
    assert [tool.name for tool in tools] == ["get_positions"]
    assert tools[0].input_schema == {"type": "object", "properties": {}}


@pytest.mark.unit
def test_call_tool_connects_lazily_and_unwraps(client):
    with _patched():
        assert client.call_tool("get_positions", {"a": 1}) == {"echo": "get_positions"}
        assert FakeSession.instances[0].calls == [("get_positions", {"a": 1})]


@pytest.mark.unit
def test_tool_exceptions_surface_in_the_calling_thread(client):
    # The job runs on another thread; the error must not be swallowed there.
    with _patched(), pytest.raises(RuntimeError, match="tool exploded"):
        client.call_tool("boom")


@pytest.mark.unit
def test_session_survives_a_failed_call(client):
    with _patched():
        with pytest.raises(RuntimeError):
            client.call_tool("boom")
        assert client.call_tool("get_positions") == {"echo": "get_positions"}
        assert len(FakeSession.instances) == 1


@pytest.mark.unit
def test_transport_failure_raises_rather_than_hanging(client):
    with (
        _patched(transport=_failing_transport),
        pytest.raises(BrokerConnectionError, match="connection refused"),
    ):
        client.connect()


@pytest.mark.unit
def test_close_exits_the_session_and_is_idempotent(client):
    with _patched():
        client.connect()
        session = FakeSession.instances[0]
        client.close()
        client.close()
    assert session.exited is True


@pytest.mark.unit
def test_reconnect_after_close_opens_a_fresh_session(client):
    with _patched():
        client.connect()
        client.close()
        client.connect()
        assert len(FakeSession.instances) == 2


@pytest.mark.unit
def test_missing_mcp_package_explains_the_extra(client):
    # Shadowing with None makes `from mcp import ...` raise ImportError.
    with (
        patch.dict(sys.modules, {"mcp": None}),
        pytest.raises(BrokerNotConfiguredError, match="tradingagents\\[robinhood\\]"),
    ):
        client.connect()

"""A synchronous MCP client over Streamable HTTP.

TradingAgents' graph is synchronous (LangGraph ``invoke``) while the MCP Python
SDK is async-only, so this module bridges the two. It owns one daemon thread
running its own event loop; the transport and ``ClientSession`` context
managers are entered once inside a single "serving" coroutine, and every later
call is handed to that same coroutine through a queue.

Keeping all session I/O in the task that opened the contexts is deliberate:
anyio cancel scopes must be exited by the task that entered them, so awaiting
the session from an unrelated task is what produces the "cancel scope in a
different task" errors this design avoids.

``mcp`` is an optional dependency (``pip install "tradingagents[robinhood]"``);
it is imported lazily so the core install stays lean and the rest of the
framework keeps working without it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .errors import (
    BrokerConnectionError,
    BrokerError,
    BrokerNotConfiguredError,
    BrokerToolError,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0

# Keys servers commonly use to wrap a payload that is really just a list.
_ENVELOPE_KEYS = ("results", "result", "data", "items")


@dataclass(frozen=True)
class ToolSpec:
    """One tool as advertised by ``tools/list``."""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    @property
    def properties(self) -> Mapping[str, Any]:
        props = self.input_schema.get("properties")
        return props if isinstance(props, Mapping) else {}

    @property
    def required(self) -> tuple[str, ...]:
        req = self.input_schema.get("required")
        return tuple(str(name) for name in req) if isinstance(req, (list, tuple)) else ()


def _open_transport(url: str, headers: dict[str, str], timeout: float):
    """Return an async context manager yielding ``(read_stream, write_stream)``.

    The Streamable HTTP entry point was renamed between MCP SDK 1.x
    (``streamablehttp_client``, which accepts ``headers`` directly) and 2.x
    (``streamable_http_client``, which takes a preconfigured httpx client), so
    both spellings are supported rather than pinning users to one SDK line.
    """
    try:
        import mcp.client.streamable_http as transport_mod
    except ImportError as exc:  # pragma: no cover - exercised via _require_sdk
        raise BrokerNotConfiguredError(
            "The 'mcp' package is required for broker support. "
            'Install it with: pip install "tradingagents[robinhood]"'
        ) from exc

    modern = getattr(transport_mod, "streamable_http_client", None)
    if modern is not None:
        http_client = transport_mod.create_mcp_http_client(headers=headers or None)
        return modern(url, http_client=http_client)

    legacy = getattr(transport_mod, "streamablehttp_client", None)
    if legacy is None:  # pragma: no cover - no known SDK version lacks both
        raise BrokerNotConfiguredError(
            "The installed 'mcp' package exposes no Streamable HTTP client; "
            'upgrade it with: pip install -U "mcp>=1.9"'
        )
    return legacy(url, headers=headers or None, timeout=timedelta(seconds=timeout))


def _require_sdk():
    """Import and return ``ClientSession``, or explain how to install it."""
    try:
        from mcp import ClientSession
    except ImportError as exc:
        raise BrokerNotConfiguredError(
            "The 'mcp' package is required for broker support. "
            'Install it with: pip install "tradingagents[robinhood]"'
        ) from exc
    return ClientSession


class McpSyncClient:
    """Blocking wrapper around one Streamable HTTP MCP session.

    Connects lazily on first use and stays connected until :meth:`close`, so a
    run that reads the account and later places an order pays the handshake
    cost once. Usable as a context manager.
    """

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.url = url
        self._headers = dict(headers or {})
        self._timeout = float(timeout)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._ready = threading.Event()
        self._connected = False
        self._startup_error: BaseException | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Open the session if it is not already open. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._connected = False
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._run, name="tradingagents-mcp", daemon=True
            )
            self._thread.start()

        # The serving coroutine sets _ready either once initialize() returns or
        # when it gives up, so this wait covers both success and failure paths.
        if not self._ready.wait(self._timeout):
            self.close()
            raise BrokerConnectionError(
                f"Timed out after {self._timeout}s connecting to the MCP server at {self.url}"
            )
        if not self._connected:
            error = self._startup_error
            self.close()
            # Preserve the taxonomy: "the mcp extra is not installed" is a
            # configuration problem, not a transport one, and callers that
            # distinguish them should not have to parse the message.
            if isinstance(error, BrokerError):
                raise error
            raise BrokerConnectionError(
                f"Could not connect to the MCP server at {self.url}: {error}"
            ) from error

    def close(self) -> None:
        """Shut the session and its event loop down. Idempotent."""
        with self._lock:
            thread, loop, queue = self._thread, self._loop, self._queue
            self._thread = None
        if thread is None:
            return
        if loop is not None and queue is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # loop already finished
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))
        thread.join(timeout=self._timeout)
        self._loop = None
        self._queue = None
        self._connected = False

    def __enter__(self) -> McpSyncClient:
        self.connect()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # MCP operations
    # ------------------------------------------------------------------
    def list_tools(self) -> list[ToolSpec]:
        """Return every tool the server advertises."""

        async def job(session):
            result = await session.list_tools()
            return [
                ToolSpec(
                    name=tool.name,
                    description=getattr(tool, "description", None) or "",
                    input_schema=dict(getattr(tool, "inputSchema", None) or {}),
                )
                for tool in result.tools
            ]

        return self._submit(job)

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        """Call ``name`` and return its payload as plain Python data."""

        async def job(session):
            return await session.call_tool(name, dict(arguments or {}))

        logger.debug("MCP call_tool %s(%s)", name, dict(arguments or {}))
        return unwrap_result(name, self._submit(job))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _submit(self, job: Callable[[Any], Awaitable[Any]]) -> Any:
        self.connect()
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            raise BrokerConnectionError("The MCP client is not connected")

        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (job, future))
        except RuntimeError as exc:
            raise BrokerConnectionError("The MCP client event loop is closed") from exc

        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise BrokerConnectionError(
                f"Timed out after {self._timeout}s waiting for the MCP server at {self.url}"
            ) from exc

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 — reported to the calling thread
            if self._startup_error is None:
                self._startup_error = exc
        finally:
            self._ready.set()

    async def _serve(self) -> None:
        client_session = _require_sdk()
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        failure: BaseException | None = None
        try:
            # 1.x yields (read, write, get_session_id); 2.x yields (read, write).
            async with (
                _open_transport(self.url, self._headers, self._timeout) as streams,
                client_session(streams[0], streams[1]) as session,
            ):
                await session.initialize()
                self._connected = True
                self._ready.set()
                await self._pump(session)
        except BaseException as exc:  # noqa: BLE001 — reported to the calling thread
            failure = exc
            if not self._connected:
                self._startup_error = exc
        finally:
            self._connected = False
            self._ready.set()
            self._reject_pending(failure)

    async def _pump(self, session) -> None:
        """Run queued jobs one at a time until a ``None`` sentinel arrives."""
        assert self._queue is not None
        while True:
            job, future = await self._queue.get()
            if job is None:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(await job(session))
            except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
                future.set_exception(exc)

    def _reject_pending(self, failure: BaseException | None) -> None:
        """Fail any queued jobs so callers get an error instead of hanging."""
        queue = self._queue
        if queue is None:
            return
        error = failure or BrokerConnectionError("The MCP session closed")
        while not queue.empty():
            try:
                _job, future = queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - guarded by empty()
                break
            if future is not None and not future.done():
                future.set_exception(error)


def unwrap_result(tool_name: str, result: Any) -> Any:
    """Reduce an MCP ``CallToolResult`` to plain Python data.

    Prefers ``structuredContent`` when the server supplies it, then falls back
    to the text blocks — parsed as JSON when they look like JSON, since most
    servers return a JSON document as a single text block.
    """
    if getattr(result, "isError", False):
        raise BrokerToolError(
            f"{tool_name} failed: {result_text(result) or 'the server reported an error'}"
        )

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping) and structured:
        return _unwrap_envelope(dict(structured))

    text = result_text(result)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return text
    return _unwrap_envelope(parsed) if isinstance(parsed, Mapping) else parsed


def _unwrap_envelope(payload: dict[str, Any]) -> Any:
    """Unwrap ``{"result": [...]}``-style single-key envelopes."""
    if len(payload) != 1:
        return payload
    key, value = next(iter(payload.items()))
    if str(key).lower() in _ENVELOPE_KEYS and isinstance(value, (list, dict)):
        return value
    return payload


def result_text(result: Any) -> str:
    """Join the text blocks of a tool result into one string."""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)

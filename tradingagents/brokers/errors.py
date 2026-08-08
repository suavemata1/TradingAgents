"""Broker error taxonomy.

Mirrors the shape of :mod:`tradingagents.dataflows.errors`: one hierarchy so
callers react by *behavior* rather than by broker. Everything the brokerage
layer can fail with derives from :class:`BrokerError`, and the graph catches
that base type to keep a broker outage from killing an otherwise-complete
analysis run.

    BrokerError
    ├── BrokerNotConfiguredError   missing credentials / optional dependency
    ├── BrokerConnectionError      transport or handshake failure
    ├── BrokerCapabilityError      server does not expose a required tool
    └── BrokerToolError            the server ran the tool and reported failure
"""

from __future__ import annotations


class BrokerError(Exception):
    """Base for any condition where the brokerage layer could not proceed."""


class BrokerNotConfiguredError(BrokerError):
    """Credentials, endpoint, or the optional ``mcp`` dependency are missing."""


class BrokerConnectionError(BrokerError):
    """The MCP transport failed, timed out, or the handshake was rejected."""


class BrokerCapabilityError(BrokerError):
    """The connected server does not expose a tool for a required capability.

    Carries the capability name and the tool names the server *did* advertise
    so the message tells the user exactly what to pin via ``broker_tool_map``
    instead of leaving them to guess.
    """

    def __init__(self, capability: str, available: tuple[str, ...] = ()):
        self.capability = capability
        self.available = tuple(available)
        msg = f"The broker MCP server exposes no {capability!r} tool"
        if self.available:
            msg += f" (advertised tools: {', '.join(sorted(self.available))})"
            msg += (
                f". Pin it explicitly with config "
                f'broker_tool_map={{"{capability}": "<tool name>"}}.'
            )
        super().__init__(msg)


class BrokerToolError(BrokerError):
    """A tool call reached the server but came back as an error result."""

"""Brokerage integration.

Optional and off by default: TradingAgents ships as research software and
places no orders unless you enable execution explicitly (see
:mod:`tradingagents.brokers.execution` for the safety gates).

Requires the ``mcp`` extra: ``pip install "tradingagents[robinhood]"``.
"""

from .discovery import CAPABILITIES, Capabilities, discover
from .errors import (
    BrokerCapabilityError,
    BrokerConnectionError,
    BrokerError,
    BrokerNotConfiguredError,
    BrokerToolError,
)
from .execution import (
    ARMED_ENV_VAR,
    MODE_DRY_RUN,
    MODE_LIVE,
    MODE_OFF,
    ExecutionResult,
    OrderIntent,
    build_order_intent,
    execute_decision,
    resolve_execution_mode,
)
from .mcp_client import McpSyncClient, ToolSpec
from .robinhood import DEFAULT_MCP_URL, AccountSnapshot, Position, RobinhoodBroker

__all__ = [
    "ARMED_ENV_VAR",
    "CAPABILITIES",
    "DEFAULT_MCP_URL",
    "MODE_DRY_RUN",
    "MODE_LIVE",
    "MODE_OFF",
    "AccountSnapshot",
    "BrokerCapabilityError",
    "BrokerConnectionError",
    "BrokerError",
    "BrokerNotConfiguredError",
    "BrokerToolError",
    "Capabilities",
    "ExecutionResult",
    "McpSyncClient",
    "OrderIntent",
    "Position",
    "RobinhoodBroker",
    "ToolSpec",
    "build_order_intent",
    "discover",
    "execute_decision",
    "resolve_execution_mode",
]

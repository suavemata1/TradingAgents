"""Robinhood brokerage access over the hosted MCP server.

The server lives at ``https://agent.robinhood.com/mcp/trading`` and speaks
Streamable HTTP MCP with a bearer token. Everything below is written against
*capabilities* rather than a fixed tool list (see
:mod:`tradingagents.brokers.discovery`), and every payload reader tolerates
the field-name variation you get from a vendor API you do not control: a
renamed key degrades one field to ``None`` instead of raising.

Nothing in this module places an order on its own — order placement is
reachable only through :mod:`tradingagents.brokers.execution`, which owns the
safety gates.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .discovery import (
    ACCOUNT,
    PLACE_ORDER,
    POSITIONS,
    QUOTE,
    Capabilities,
    discover,
    map_arguments,
    missing_required,
    normalize_key,
)
from .errors import BrokerCapabilityError, BrokerNotConfiguredError
from .mcp_client import DEFAULT_TIMEOUT, McpSyncClient

logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://agent.robinhood.com/mcp/trading"

# Checked in order; the first non-empty one wins.
TOKEN_ENV_VARS = ("ROBINHOOD_MCP_TOKEN", "ROBINHOOD_API_TOKEN", "ROBINHOOD_TOKEN")

# Keys that wrap a list of records in an object response.
_COLLECTION_KEYS = (
    "positions", "holdings", "results", "data", "items", "orders", "accounts",
)


def _pick(record: Mapping[str, Any], *aliases: str) -> Any:
    """Return the first non-empty value among ``aliases`` (name-insensitive)."""
    normalized = {normalize_key(key): value for key, value in record.items()}
    for alias in aliases:
        value = normalized.get(normalize_key(alias))
        if value not in (None, "", [], {}):
            return value
    return None


def _as_float(value: Any) -> float | None:
    """Parse a number that may arrive as ``12.5``, ``"12.5"`` or ``"$1,234.50"``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _as_records(payload: Any) -> tuple[dict[str, Any], ...]:
    """Normalize a *collection* payload into a tuple of record dicts.

    Handles both a bare list and the ``{"results": [...]}`` envelopes servers
    wrap them in. Use this only where a list is what was asked for; see
    :func:`_as_record` for single-object reads.
    """
    if isinstance(payload, Mapping):
        for key in _COLLECTION_KEYS:
            value = _pick(payload, key)
            if _is_list(value):
                return tuple(dict(item) for item in value if isinstance(item, Mapping))
        return (dict(payload),)
    if _is_list(payload):
        return tuple(dict(item) for item in payload if isinstance(item, Mapping))
    return ()


def _as_record(payload: Any) -> dict[str, Any]:
    """Normalize a *single-object* payload (an account, a quote) into one dict.

    Deliberately stricter than :func:`_as_records`: a multi-key object is taken
    as the record itself even when one of its values is a list, because an
    account payload that happens to embed ``holdings`` is an account — not a
    list of holdings. Only an unambiguous single-key envelope is unwrapped.
    """
    if isinstance(payload, Mapping):
        if len(payload) == 1:
            only = next(iter(payload.values()))
            if isinstance(only, Mapping):
                return dict(only)
            if _is_list(only):
                first = next((item for item in only if isinstance(item, Mapping)), None)
                return dict(first) if first is not None else {}
        return dict(payload)
    if _is_list(payload):
        first = next((item for item in payload if isinstance(item, Mapping)), None)
        return dict(first) if first is not None else {}
    return {}


@dataclass(frozen=True)
class Position:
    """One open position. ``quantity`` is signed: negative means short."""

    symbol: str
    quantity: float
    market_value: float | None = None
    average_price: float | None = None

    def to_line(self) -> str:
        parts = [f"{self.symbol}: {self.quantity:g} shares"]
        if self.market_value is not None:
            parts.append(f"market value ${self.market_value:,.2f}")
        if self.average_price is not None:
            parts.append(f"avg cost ${self.average_price:,.2f}")
        return " — ".join(parts)


@dataclass(frozen=True)
class AccountSnapshot:
    """Point-in-time account state used for prompts and position sizing."""

    buying_power: float | None = None
    portfolio_value: float | None = None
    cash: float | None = None
    positions: tuple[Position, ...] = ()

    def position_for(self, symbol: str) -> Position | None:
        target = str(symbol).upper()
        for position in self.positions:
            if position.symbol.upper() == target:
                return position
        return None

    def to_markdown(self, symbol: str | None = None) -> str:
        """Render the snapshot for injection into an agent prompt."""
        lines = ["**Live brokerage account (Robinhood)**"]
        for label, value in (
            ("Buying power", self.buying_power),
            ("Portfolio value", self.portfolio_value),
            ("Cash", self.cash),
        ):
            if value is not None:
                lines.append(f"- {label}: ${value:,.2f}")

        if symbol:
            held = self.position_for(symbol)
            lines.append(
                f"- Current position in {symbol.upper()}: {held.to_line()}"
                if held
                else f"- Current position in {symbol.upper()}: none"
            )

        if self.positions:
            lines.append(f"- Open positions ({len(self.positions)}):")
            lines.extend(f"  - {position.to_line()}" for position in self.positions)
        else:
            lines.append("- Open positions: none")
        return "\n".join(lines)


def _parse_position(record: Mapping[str, Any]) -> Position | None:
    symbol = _pick(record, "symbol", "ticker", "instrument_symbol", "instrument")
    quantity = _as_float(
        _pick(record, "quantity", "qty", "shares", "quantity_owned", "size", "units")
    )
    if not symbol or quantity is None:
        return None
    return Position(
        symbol=str(symbol).upper(),
        quantity=quantity,
        market_value=_as_float(
            _pick(record, "market_value", "equity", "total_value", "value")
        ),
        average_price=_as_float(
            _pick(record, "average_price", "average_buy_price", "avg_cost", "cost_basis")
        ),
    )


class RobinhoodBroker:
    """Capability-driven client for the Robinhood trading MCP server."""

    def __init__(
        self,
        client: McpSyncClient,
        *,
        tool_map: Mapping[str, str] | None = None,
        order_extra_args: Mapping[str, Any] | None = None,
        arg_map: Mapping[str, tuple[str, ...]] | None = None,
    ):
        self._client = client
        self._tool_map = dict(tool_map or {})
        self._order_extra_args = dict(order_extra_args or {})
        self._arg_map = {key: tuple(value) for key, value in (arg_map or {}).items()}
        self._capabilities: Capabilities | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RobinhoodBroker:
        """Build a broker from a TradingAgents config dict.

        Raises :class:`BrokerNotConfiguredError` when no API token is present;
        callers that treat a missing broker as "skip it" should catch
        :class:`~tradingagents.brokers.errors.BrokerError`.
        """
        token = next(
            (os.environ[var] for var in TOKEN_ENV_VARS if os.environ.get(var, "").strip()),
            None,
        )
        if not token:
            raise BrokerNotConfiguredError(
                "No Robinhood MCP credentials found. Set one of: "
                + ", ".join(TOKEN_ENV_VARS)
            )

        url = str(config.get("broker_mcp_url") or DEFAULT_MCP_URL)
        client = McpSyncClient(
            url,
            headers={"Authorization": f"Bearer {token.strip()}"},
            timeout=float(config.get("broker_timeout") or DEFAULT_TIMEOUT),
        )
        return cls(
            client,
            tool_map=config.get("broker_tool_map") or {},
            order_extra_args=config.get("broker_order_extra_args") or {},
            arg_map=config.get("broker_arg_map") or {},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RobinhoodBroker:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    def capabilities(self) -> Capabilities:
        """Discover (once) what this server exposes."""
        if self._capabilities is None:
            self._capabilities = discover(self._client.list_tools(), self._tool_map)
            logger.info("Broker capabilities: %s", self._capabilities.summary())
        return self._capabilities

    def _call(self, capability: str, intent: Mapping[str, Any], **kwargs) -> Any:
        spec = self.capabilities().spec(capability)
        arguments = map_arguments(
            spec, intent, aliases=self._arg_map, extra=kwargs.pop("extra", None)
        )
        unsatisfied = missing_required(spec, arguments)
        if unsatisfied:
            raise BrokerCapabilityError(
                f"{capability} (tool {spec.name!r} requires "
                f"{', '.join(unsatisfied)}, which could not be filled — supply "
                f"them via config broker_order_extra_args)",
                self.capabilities().tool_names,
            )
        return self._client.call_tool(spec.name, arguments)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_positions(self) -> tuple[Position, ...]:
        payload = self._call(POSITIONS, {})
        parsed = (_parse_position(record) for record in _as_records(payload))
        return tuple(position for position in parsed if position is not None)

    def get_account(self, symbol: str | None = None) -> AccountSnapshot:
        """Fetch balances plus open positions as a single snapshot.

        Positions are optional: a server that exposes balances but no positions
        tool still yields a usable snapshot rather than failing the whole read.
        """
        record = _as_record(self._call(ACCOUNT, {}))

        positions: tuple[Position, ...] = ()
        if self.capabilities().has(POSITIONS):
            positions = self.get_positions()
        elif record:
            # Some servers return holdings inline with the account payload.
            inline = _pick(record, "positions", "holdings")
            if inline is not None:
                parsed = (_parse_position(item) for item in _as_records(inline))
                positions = tuple(item for item in parsed if item is not None)

        return AccountSnapshot(
            buying_power=_as_float(
                _pick(
                    record, "buying_power", "purchasing_power",
                    "cash_available_for_trading", "available_cash", "available_funds",
                )
            ),
            portfolio_value=_as_float(
                _pick(record, "portfolio_value", "total_equity", "account_value", "equity")
            ),
            cash=_as_float(_pick(record, "cash", "cash_balance", "settled_cash", "uninvested_cash")),
            positions=positions,
        )

    def get_quote(self, symbol: str) -> float | None:
        """Return the last traded price, or None when unavailable."""
        if not self.capabilities().has(QUOTE):
            return None
        record = _as_record(self._call(QUOTE, {"symbol": symbol}))
        if not record:
            return None
        return _as_float(
            _pick(record, "last_trade_price", "last_price", "price", "mark_price", "ask_price")
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float | None = None,
        notional: float | None = None,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str | None = None,
    ) -> Any:
        """Submit an order. Callers are responsible for the safety gates.

        This is intentionally low-level and unguarded — the decision to place a
        real order belongs to :func:`tradingagents.brokers.execution.execute_decision`,
        which is the only caller in this codebase.
        """
        if quantity is None and notional is None:
            raise ValueError("place_order requires either quantity or notional")

        return self._call(
            PLACE_ORDER,
            {
                "symbol": str(symbol).upper(),
                "side": str(side).lower(),
                "quantity": quantity,
                "notional": notional,
                "order_type": str(order_type).lower(),
                "limit_price": limit_price,
                "time_in_force": time_in_force,
            },
            extra=self._order_extra_args,
        )

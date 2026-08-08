"""Runtime capability discovery for a brokerage MCP server.

Tool names and argument spellings are deliberately *not* hardcoded. The hosted
Robinhood server is not reachable from CI, and a vendor-hosted server can
change its schema without a release on our side, so the broker asks the server
what it exposes (``tools/list``) and matches the answer against capability
patterns. Whatever the matcher gets wrong is pinnable through the
``broker_tool_map`` and ``broker_arg_map`` config keys, so a schema surprise is
a config change rather than a code change.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import BrokerCapabilityError
from .mcp_client import ToolSpec

# Capability names used throughout the broker layer.
POSITIONS = "positions"
ACCOUNT = "account"
QUOTE = "quote"
PLACE_ORDER = "place_order"
CANCEL_ORDER = "cancel_order"
LIST_ORDERS = "list_orders"

CAPABILITIES: tuple[str, ...] = (
    POSITIONS, ACCOUNT, QUOTE, PLACE_ORDER, CANCEL_ORDER, LIST_ORDERS,
)

# Ordered rules — the first capability whose pattern matches a tool's
# normalized name claims it. Order encodes precedence, which is what keeps
# "cancel_order" from reading as an order *listing* and "get_buying_power"
# from reading as a bare "buy" order tool.
_RULES: tuple[tuple[str, str], ...] = (
    (CANCEL_ORDER, r"cancel"),
    (PLACE_ORDER, r"(place|create|submit|enter|execute)\w*order|order\w*(place|create|submit)|^(buy|sell)(_(stock|shares|equity|order))?$"),
    (LIST_ORDERS, r"order"),
    (POSITIONS, r"position|holding"),
    # "equity" is deliberately absent — it reads as an asset class as often as
    # an account balance ("place_equity_order", "get_equity_quote").
    (ACCOUNT, r"account|buying\w*power|balance|portfolio|cash"),
    (QUOTE, r"quote|price|market\w*data|last\w*trade"),
)

# Canonical order/query fields -> the parameter names servers actually use.
# Matched case- and separator-insensitively against the tool's input schema.
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument", "instrument_id", "stock", "security", "asset"),
    "side": ("side", "action", "direction", "order_side", "transaction_type"),
    "quantity": ("quantity", "qty", "shares", "share_count", "size", "units"),
    "notional": (
        "notional", "notional_amount", "amount_in_dollars", "dollar_amount",
        "dollars", "cash_amount", "amount",
    ),
    "order_type": ("order_type", "type", "execution_type"),
    "limit_price": ("limit_price", "price", "limit", "limit_amount"),
    "stop_price": ("stop_price", "stop", "trigger_price"),
    "time_in_force": ("time_in_force", "tif", "duration"),
    "account_id": ("account_id", "account_number", "account", "account_uuid"),
    "order_id": ("order_id", "id", "order_uuid", "order_number"),
}


def normalize_key(value: Any) -> str:
    """Collapse a name to lowercase alphanumerics for tolerant comparison.

    ``"Buying Power"``, ``"buying_power"`` and ``"buyingPower"`` all normalize
    to ``"buyingpower"``, so alias tables stay short.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalize_tool_name(name: str) -> str:
    """Lowercase a tool name and collapse separators to single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def classify_tool(name: str) -> str | None:
    """Return the capability ``name`` looks like, or None if nothing matches."""
    normalized = _normalize_tool_name(name)
    for capability, pattern in _RULES:
        if re.search(pattern, normalized):
            return capability
    return None


@dataclass(frozen=True)
class Capabilities:
    """What the connected server can do, and which tool provides each ability."""

    tools: tuple[ToolSpec, ...]
    mapping: Mapping[str, str]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    def has(self, capability: str) -> bool:
        return capability in self.mapping

    def spec(self, capability: str) -> ToolSpec:
        """Return the ``ToolSpec`` backing ``capability``.

        Raises :class:`BrokerCapabilityError` — naming the tools the server did
        advertise — rather than returning None, so a missing ability surfaces
        as an actionable message instead of an ``AttributeError`` later.
        """
        name = self.mapping.get(capability)
        if name is None:
            raise BrokerCapabilityError(capability, self.tool_names)
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise BrokerCapabilityError(capability, self.tool_names)

    def summary(self) -> str:
        """One-line human-readable capability report, for logs and errors."""
        resolved = ", ".join(
            f"{cap}={self.mapping[cap]}" for cap in CAPABILITIES if cap in self.mapping
        )
        missing = [cap for cap in CAPABILITIES if cap not in self.mapping]
        text = resolved or "none"
        if missing:
            text += f" (unavailable: {', '.join(missing)})"
        return text


def discover(
    tools: list[ToolSpec] | tuple[ToolSpec, ...],
    overrides: Mapping[str, str] | None = None,
) -> Capabilities:
    """Map capabilities onto the server's advertised tools.

    ``overrides`` pins a capability to an exact tool name and wins over pattern
    matching; naming a tool the server does not expose is an error rather than
    a silent fallback, because the whole point of pinning is to stop guessing.
    """
    tools = tuple(tools)
    by_name = {tool.name: tool for tool in tools}
    mapping: dict[str, str] = {}

    for capability, tool_name in (overrides or {}).items():
        if not tool_name:
            continue
        if tool_name not in by_name:
            raise BrokerCapabilityError(str(capability), tuple(by_name))
        mapping[str(capability)] = tool_name

    candidates: dict[str, list[ToolSpec]] = {}
    for tool in tools:
        capability = classify_tool(tool.name)
        if capability is not None and capability not in mapping:
            candidates.setdefault(capability, []).append(tool)

    for capability, matches in candidates.items():
        # Shortest name first: "get_positions" is a better general-purpose
        # match than "get_options_positions" or "get_positions_history".
        best = min(matches, key=lambda tool: (len(tool.name), tool.name))
        mapping[capability] = best.name

    return Capabilities(tools=tools, mapping=mapping)


def _coerce_to_schema(schema: Mapping[str, Any], value: Any) -> Any:
    """Nudge ``value`` toward what the parameter's JSON schema asks for.

    Handles the two mismatches that actually bite in practice: enums that spell
    a value differently (``"buy"`` vs ``"BUY"``), and servers that want numbers
    as strings or strings as numbers.
    """
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and enum:
        target = normalize_key(value)
        for option in enum:
            if normalize_key(option) == target:
                return option
        return value

    declared = schema.get("type")
    types = declared if isinstance(declared, (list, tuple)) else [declared]
    types = [t for t in types if isinstance(t, str)]

    if "string" in types and not isinstance(value, str):
        # Avoid "1e-05" / "1.0" surprises for whole numbers.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str) and ("number" in types or "integer" in types):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if "integer" in types and number.is_integer() else number
    if "integer" in types and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def map_arguments(
    spec: ToolSpec,
    intent: Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Translate canonical intent fields onto ``spec``'s real parameter names.

    Fields the tool does not accept are dropped rather than passed through:
    an unexpected argument is the most common reason a strict server rejects an
    otherwise-valid order. ``extra`` is merged last and verbatim, which is the
    escape hatch for server-specific required arguments (an account id, say)
    that no alias table could anticipate.
    """
    merged_aliases = {**_ARG_ALIASES, **(aliases or {})}
    supplied = {key: value for key, value in intent.items() if value is not None}

    properties = spec.properties
    if not properties:
        # No published schema — send the canonical names and let the server
        # validate rather than dropping everything we have.
        return {**supplied, **dict(extra or {})}

    lookup = {normalize_key(name): name for name in properties}
    args: dict[str, Any] = {}
    for field_name, value in supplied.items():
        target = None
        for candidate in (field_name, *merged_aliases.get(field_name, ())):
            target = lookup.get(normalize_key(candidate))
            if target is not None:
                break
        if target is None:
            continue
        args[target] = _coerce_to_schema(properties[target], value)

    args.update(dict(extra or {}))
    return args


def missing_required(spec: ToolSpec, arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """Return required parameters of ``spec`` that ``arguments`` does not set."""
    return tuple(name for name in spec.required if name not in arguments)

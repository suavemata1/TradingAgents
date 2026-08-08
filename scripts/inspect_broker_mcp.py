"""Introspect a brokerage MCP server and report what TradingAgents resolved.

Run this once against your own account before enabling execution. It performs
the MCP handshake, prints every tool the server advertises with its input
schema, shows which capability each tool was matched to, and — with
``--dry-run-order`` — shows the exact arguments an order would be sent with.

It is strictly read-only unless you pass ``--place-order``, which is gated
behind the same two switches as a normal run (see
``tradingagents/brokers/execution.py``).

Usage:
    ROBINHOOD_MCP_TOKEN=... python scripts/inspect_broker_mcp.py
    ROBINHOOD_MCP_TOKEN=... python scripts/inspect_broker_mcp.py --account
    ROBINHOOD_MCP_TOKEN=... python scripts/inspect_broker_mcp.py --dry-run-order AAPL Buy

Why this exists: the hosted Robinhood server is not reachable from CI, so the
broker discovers tool names and argument spellings at runtime rather than
hardcoding them. This script shows you what that discovery actually produced,
which is the fastest way to spot a mapping that needs pinning via
``broker_tool_map`` / ``broker_arg_map`` / ``broker_order_extra_args``.
"""

from __future__ import annotations

import argparse
import json
import sys

from tradingagents.brokers.discovery import CAPABILITIES, map_arguments, missing_required
from tradingagents.brokers.errors import BrokerError
from tradingagents.brokers.execution import (
    ARMED_ENV_VAR,
    build_order_intent,
    execute_decision,
    resolve_execution_mode,
)
from tradingagents.brokers.robinhood import RobinhoodBroker
from tradingagents.default_config import DEFAULT_CONFIG


def _print_tools(capabilities) -> None:
    print(f"\n=== Tools advertised ({len(capabilities.tools)}) ===")
    reverse = {tool: cap for cap, tool in capabilities.mapping.items()}
    for tool in sorted(capabilities.tools, key=lambda t: t.name):
        matched = reverse.get(tool.name)
        label = f"  [{matched}]" if matched else "  [unmatched]"
        print(f"\n{tool.name}{label}")
        if tool.description:
            print(f"    {tool.description.strip().splitlines()[0]}")
        properties = dict(tool.properties)
        if properties:
            required = set(tool.required)
            for name, schema in properties.items():
                flag = " (required)" if name in required else ""
                kind = schema.get("type", "?")
                enum = schema.get("enum")
                kind = f"{kind} {list(enum)}" if enum else kind
                print(f"    - {name}: {kind}{flag}")
        else:
            print("    (no published input schema)")


def _print_capability_map(capabilities) -> None:
    print("\n=== Capability map ===")
    for capability in CAPABILITIES:
        tool = capabilities.mapping.get(capability)
        print(f"  {capability:<14} -> {tool if tool else '(unavailable)'}")
    unmatched = [
        tool.name for tool in capabilities.tools
        if tool.name not in set(capabilities.mapping.values())
    ]
    if unmatched:
        print(f"\n  Unmatched tools: {', '.join(sorted(unmatched))}")
        print("  Pin any of these with config broker_tool_map if a capability is wrong.")


def _print_account(broker, symbol: str | None) -> None:
    print("\n=== Account ===")
    account = broker.get_account(symbol)
    print(account.to_markdown(symbol))


def _print_dry_run(broker, config, symbol: str, rating: str) -> None:
    print(f"\n=== Dry-run order for {symbol.upper()} at rating {rating!r} ===")
    account = broker.get_account(symbol)
    intent = build_order_intent(rating, symbol, account, config)
    if intent is None:
        print("  No order: this rating implies no trade for the current account state.")
        return
    print(f"  Intent: {intent.describe()}")

    spec = broker.capabilities().spec("place_order")
    arguments = map_arguments(
        spec,
        {
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "notional": intent.notional,
            "order_type": intent.order_type,
            "limit_price": intent.limit_price,
            "time_in_force": intent.time_in_force,
        },
        aliases=config.get("broker_arg_map") or {},
        extra=config.get("broker_order_extra_args") or {},
    )
    print(f"  Would call: {spec.name}")
    print(f"  With arguments: {json.dumps(arguments, indent=4, default=str)}")

    unsatisfied = missing_required(spec, arguments)
    if unsatisfied:
        print(f"\n  !! Unfilled required arguments: {', '.join(unsatisfied)}")
        print("     Supply them via config broker_order_extra_args.")
    else:
        print("\n  All required arguments are satisfied.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", default=None, help="Override the MCP endpoint (default: config value)"
    )
    parser.add_argument("--account", action="store_true", help="Also fetch and print the account")
    parser.add_argument(
        "--dry-run-order",
        nargs=2,
        metavar=("SYMBOL", "RATING"),
        help="Show the exact order call for a symbol and 5-tier rating",
    )
    parser.add_argument(
        "--place-order",
        nargs=2,
        metavar=("SYMBOL", "RATING"),
        help=(
            "Actually route the rating through the execution gate. Still a dry "
            f"run unless execution_mode=live and {ARMED_ENV_VAR}=1."
        ),
    )
    args = parser.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    if args.url:
        config["broker_mcp_url"] = args.url

    print(f"Connecting to {config['broker_mcp_url']} ...")
    try:
        broker = RobinhoodBroker.from_config(config)
    except BrokerError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    try:
        capabilities = broker.capabilities()
        print("Connected.")
        _print_tools(capabilities)
        _print_capability_map(capabilities)

        symbol = None
        if args.dry_run_order:
            symbol = args.dry_run_order[0]
        elif args.place_order:
            symbol = args.place_order[0]

        if args.account or symbol:
            _print_account(broker, symbol)

        if args.dry_run_order:
            _print_dry_run(broker, config, *args.dry_run_order)

        if args.place_order:
            symbol, rating = args.place_order
            config["execution_enabled"] = True
            mode = resolve_execution_mode(config)
            print(f"\n=== Execution ({mode}) ===")
            result = execute_decision(rating, symbol, config, broker=broker)
            print(result.to_markdown())
    except BrokerError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        broker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

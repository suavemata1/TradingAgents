"""Robinhood MCP broker: capability discovery, argument mapping, payload
parsing, rating-to-order sizing, and the two-switch execution gate.

The hosted MCP server is never contacted — a fake ``McpSyncClient`` stands in
for it — so these run offline and without credentials. That is also the point
of the design under test: the broker is written against whatever tools a server
advertises at runtime, so the fakes here deliberately use *different* tool
names and argument spellings from one test to the next.
"""

import os
import unittest
from unittest import mock

import pytest

from tradingagents.brokers.discovery import (
    classify_tool,
    discover,
    map_arguments,
    missing_required,
)
from tradingagents.brokers.errors import BrokerCapabilityError, BrokerToolError
from tradingagents.brokers.execution import (
    ARMED_ENV_VAR,
    MODE_DRY_RUN,
    MODE_LIVE,
    MODE_OFF,
    build_order_intent,
    execute_decision,
    resolve_execution_mode,
)
from tradingagents.brokers.mcp_client import ToolSpec, unwrap_result
from tradingagents.brokers.robinhood import AccountSnapshot, Position, RobinhoodBroker


def _spec(name, properties=None, required=(), **kwargs):
    schema = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    return ToolSpec(name=name, input_schema=schema, **kwargs)


class FakeClient:
    """Stands in for ``McpSyncClient``, recording the calls it receives."""

    def __init__(self, tools, responses=None):
        self._tools = list(tools)
        self._responses = dict(responses or {})
        self.calls = []
        self.closed = False

    def list_tools(self):
        return list(self._tools)

    def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        response = self._responses.get(name)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


# ----------------------------------------------------------------------
# Capability discovery
# ----------------------------------------------------------------------
@pytest.mark.unit
class ClassificationTests(unittest.TestCase):
    def test_order_verbs_map_to_placement(self):
        for name in (
            "place_order", "create_order", "submit_equity_order",
            "placeOrder", "execute_order", "buy", "sell_shares",
        ):
            self.assertEqual(classify_tool(name), "place_order", name)

    def test_cancel_wins_over_generic_order_match(self):
        # "cancel_order" contains "order"; precedence must not read it as a listing.
        self.assertEqual(classify_tool("cancel_order"), "cancel_order")

    def test_order_listings_are_not_placement(self):
        for name in ("get_orders", "list_orders", "order_history"):
            self.assertEqual(classify_tool(name), "list_orders", name)

    def test_buying_power_is_an_account_read_not_a_buy_order(self):
        # The regression this guards: "buying_power" starts with "buy".
        self.assertEqual(classify_tool("get_buying_power"), "account")

    def test_positions_and_quotes(self):
        self.assertEqual(classify_tool("get_positions"), "positions")
        self.assertEqual(classify_tool("list_holdings"), "positions")
        self.assertEqual(classify_tool("get_quote"), "quote")

    def test_unknown_tools_are_unclassified(self):
        self.assertIsNone(classify_tool("get_weather"))


@pytest.mark.unit
class DiscoveryTests(unittest.TestCase):
    def test_shortest_name_wins_for_a_capability(self):
        caps = discover([
            _spec("get_options_positions"),
            _spec("get_positions"),
            _spec("get_positions_history"),
        ])
        self.assertEqual(caps.mapping["positions"], "get_positions")

    def test_overrides_beat_pattern_matching(self):
        caps = discover(
            [_spec("get_positions"), _spec("weird_custom_tool")],
            overrides={"positions": "weird_custom_tool"},
        )
        self.assertEqual(caps.mapping["positions"], "weird_custom_tool")

    def test_override_naming_an_absent_tool_is_an_error(self):
        # Silently falling back would defeat the purpose of pinning.
        with self.assertRaises(BrokerCapabilityError):
            discover([_spec("get_positions")], overrides={"positions": "nope"})

    def test_missing_capability_error_lists_available_tools(self):
        caps = discover([_spec("get_positions")])
        self.assertFalse(caps.has("place_order"))
        with self.assertRaises(BrokerCapabilityError) as ctx:
            caps.spec("place_order")
        self.assertIn("get_positions", str(ctx.exception))
        self.assertIn("broker_tool_map", str(ctx.exception))

    def test_summary_reports_resolved_and_missing(self):
        summary = discover([_spec("get_positions")]).summary()
        self.assertIn("positions=get_positions", summary)
        self.assertIn("place_order", summary)


# ----------------------------------------------------------------------
# Argument mapping
# ----------------------------------------------------------------------
@pytest.mark.unit
class ArgumentMappingTests(unittest.TestCase):
    def test_canonical_fields_map_onto_server_spellings(self):
        spec = _spec("place_order", {
            "ticker": {"type": "string"},
            "action": {"type": "string"},
            "share_count": {"type": "number"},
        })
        args = map_arguments(spec, {"symbol": "AAPL", "side": "buy", "quantity": 3})
        self.assertEqual(args, {"ticker": "AAPL", "action": "buy", "share_count": 3})

    def test_unsupported_fields_are_dropped_not_forwarded(self):
        # Strict servers reject unexpected arguments, so dropping beats passing through.
        spec = _spec("place_order", {"symbol": {"type": "string"}})
        args = map_arguments(spec, {"symbol": "AAPL", "time_in_force": "gfd"})
        self.assertEqual(args, {"symbol": "AAPL"})

    def test_enum_spelling_is_matched_case_insensitively(self):
        spec = _spec("place_order", {"side": {"type": "string", "enum": ["BUY", "SELL"]}})
        self.assertEqual(map_arguments(spec, {"side": "buy"})["side"], "BUY")

    def test_numbers_are_coerced_to_the_declared_type(self):
        spec = _spec("place_order", {
            "quantity": {"type": "string"},
            "limit_price": {"type": "number"},
        })
        args = map_arguments(spec, {"quantity": 5.0, "limit_price": "12.50"})
        self.assertEqual(args["quantity"], "5")
        self.assertEqual(args["limit_price"], 12.5)

    def test_schemaless_tools_receive_canonical_names(self):
        args = map_arguments(ToolSpec(name="place_order"), {"symbol": "AAPL", "side": "buy"})
        self.assertEqual(args, {"symbol": "AAPL", "side": "buy"})

    def test_extra_args_are_merged_verbatim(self):
        spec = _spec("place_order", {"symbol": {"type": "string"}})
        args = map_arguments(spec, {"symbol": "AAPL"}, extra={"account_id": "abc"})
        self.assertEqual(args["account_id"], "abc")

    def test_config_aliases_extend_the_builtin_table(self):
        spec = _spec("place_order", {"qty_of_shares": {"type": "number"}})
        args = map_arguments(
            spec, {"quantity": 2}, aliases={"quantity": ("qty_of_shares",)}
        )
        self.assertEqual(args, {"qty_of_shares": 2})

    def test_missing_required_is_reported(self):
        spec = _spec(
            "place_order",
            {"symbol": {"type": "string"}, "account_id": {"type": "string"}},
            required=("symbol", "account_id"),
        )
        args = map_arguments(spec, {"symbol": "AAPL"})
        self.assertEqual(missing_required(spec, args), ("account_id",))


# ----------------------------------------------------------------------
# Result unwrapping
# ----------------------------------------------------------------------
class _Block:
    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, content=(), structured=None, is_error=False):
        self.content = list(content)
        self.structuredContent = structured
        self.isError = is_error


@pytest.mark.unit
class UnwrapTests(unittest.TestCase):
    def test_structured_content_is_preferred(self):
        result = _Result(content=[_Block("ignored")], structured={"buying_power": 100})
        self.assertEqual(unwrap_result("t", result), {"buying_power": 100})

    def test_single_key_envelopes_are_unwrapped(self):
        result = _Result(structured={"results": [{"symbol": "AAPL"}]})
        self.assertEqual(unwrap_result("t", result), [{"symbol": "AAPL"}])

    def test_json_text_blocks_are_parsed(self):
        result = _Result(content=[_Block('{"symbol": "AAPL"}')])
        self.assertEqual(unwrap_result("t", result), {"symbol": "AAPL"})

    def test_plain_text_passes_through(self):
        self.assertEqual(unwrap_result("t", _Result(content=[_Block("hello")])), "hello")

    def test_error_results_raise(self):
        result = _Result(content=[_Block("insufficient funds")], is_error=True)
        with self.assertRaises(BrokerToolError) as ctx:
            unwrap_result("place_order", result)
        self.assertIn("insufficient funds", str(ctx.exception))


# ----------------------------------------------------------------------
# Broker payload parsing
# ----------------------------------------------------------------------
@pytest.mark.unit
class BrokerReadTests(unittest.TestCase):
    def _broker(self, responses, tools=None, **kwargs):
        tools = tools or [
            _spec("get_account"),
            _spec("get_positions"),
            _spec("place_order", {
                "symbol": {"type": "string"},
                "side": {"type": "string"},
                "quantity": {"type": "number"},
                "notional": {"type": "number"},
            }),
        ]
        client = FakeClient(tools, responses)
        return RobinhoodBroker(client, **kwargs), client

    def test_account_and_positions_are_parsed(self):
        broker, _ = self._broker({
            "get_account": {"buying_power": "1,500.50", "portfolio_value": 9000},
            "get_positions": [
                {"symbol": "aapl", "quantity": "10", "market_value": 2000,
                 "average_price": 180.0},
                {"symbol": "MSFT", "qty": 4},
            ],
        })
        account = broker.get_account("AAPL")
        self.assertEqual(account.buying_power, 1500.50)
        self.assertEqual(account.portfolio_value, 9000.0)
        self.assertEqual(len(account.positions), 2)

        held = account.position_for("aapl")
        self.assertEqual(held.symbol, "AAPL")
        self.assertEqual(held.quantity, 10.0)
        self.assertEqual(account.position_for("TSLA"), None)

    def test_unparseable_position_records_are_skipped_not_fatal(self):
        broker, _ = self._broker({
            "get_account": {"buying_power": 100},
            "get_positions": [{"nonsense": True}, {"symbol": "AAPL", "quantity": 1}],
        })
        self.assertEqual(len(broker.get_account().positions), 1)

    def test_unknown_balance_fields_degrade_to_none(self):
        broker, _ = self._broker({"get_account": {"mystery": 1}, "get_positions": []})
        account = broker.get_account()
        self.assertIsNone(account.buying_power)
        self.assertEqual(account.positions, ())

    def test_positions_can_arrive_inline_with_the_account(self):
        # Server exposing no positions tool at all.
        broker, _ = self._broker(
            {"get_account": {"buying_power": 50, "holdings": [{"symbol": "F", "quantity": 3}]}},
            tools=[_spec("get_account")],
        )
        account = broker.get_account()
        self.assertEqual(account.positions[0].symbol, "F")

    def test_place_order_maps_arguments_and_drops_unsupported_ones(self):
        broker, client = self._broker({"place_order": {"id": "o1"}})
        broker.place_order(symbol="aapl", side="buy", notional=250.0)
        name, args = client.calls[-1]
        self.assertEqual(name, "place_order")
        self.assertEqual(args["symbol"], "AAPL")
        self.assertEqual(args["side"], "buy")
        self.assertEqual(args["notional"], 250.0)
        self.assertNotIn("time_in_force", args)  # not in this server's schema

    def test_place_order_forwards_configured_extra_args(self):
        broker, client = self._broker(
            {"place_order": {}},
            tools=[_spec("get_account"), _spec("get_positions"),
                   _spec("place_order", {"symbol": {"type": "string"},
                                         "account_id": {"type": "string"}},
                         required=("symbol", "account_id"))],
            order_extra_args={"account_id": "acct-1"},
        )
        broker.place_order(symbol="AAPL", side="buy", notional=10)
        self.assertEqual(client.calls[-1][1]["account_id"], "acct-1")

    def test_unfillable_required_argument_is_an_actionable_error(self):
        broker, _ = self._broker(
            {"place_order": {}},
            tools=[_spec("place_order", {"symbol": {"type": "string"},
                                         "account_id": {"type": "string"}},
                         required=("symbol", "account_id"))],
        )
        with self.assertRaises(BrokerCapabilityError) as ctx:
            broker.place_order(symbol="AAPL", side="buy", notional=10)
        self.assertIn("account_id", str(ctx.exception))
        self.assertIn("broker_order_extra_args", str(ctx.exception))

    def test_place_order_requires_a_size(self):
        broker, _ = self._broker({"place_order": {}})
        with self.assertRaises(ValueError):
            broker.place_order(symbol="AAPL", side="buy")

    def test_quote_is_optional(self):
        broker, _ = self._broker({"get_account": {}}, tools=[_spec("get_account")])
        self.assertIsNone(broker.get_quote("AAPL"))


# ----------------------------------------------------------------------
# Execution gate
# ----------------------------------------------------------------------
@pytest.mark.unit
class ExecutionModeTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertEqual(resolve_execution_mode({}, env={}), MODE_OFF)

    def test_enabled_defaults_to_dry_run(self):
        self.assertEqual(
            resolve_execution_mode({"execution_enabled": True}, env={}), MODE_DRY_RUN
        )

    def test_live_requires_the_arming_env_var(self):
        config = {"execution_enabled": True, "execution_mode": "live"}
        # Config alone is not enough — this is the whole point of two switches.
        self.assertEqual(resolve_execution_mode(config, env={}), MODE_DRY_RUN)
        self.assertEqual(
            resolve_execution_mode(config, env={ARMED_ENV_VAR: "1"}), MODE_LIVE
        )

    def test_arming_env_var_alone_is_not_enough(self):
        self.assertEqual(
            resolve_execution_mode({}, env={ARMED_ENV_VAR: "1"}), MODE_OFF
        )

    def test_arming_var_accepts_common_truthy_spellings(self):
        config = {"execution_enabled": True, "execution_mode": "live"}
        for value in ("1", "true", "TRUE", "yes", "on"):
            self.assertEqual(
                resolve_execution_mode(config, env={ARMED_ENV_VAR: value}), MODE_LIVE
            )
        for value in ("0", "false", "no", "", "maybe"):
            self.assertEqual(
                resolve_execution_mode(config, env={ARMED_ENV_VAR: value}), MODE_DRY_RUN
            )

    def test_invalid_mode_fails_loudly(self):
        with self.assertRaises(ValueError):
            resolve_execution_mode(
                {"execution_enabled": True, "execution_mode": "yolo"}, env={}
            )


@pytest.mark.unit
class OrderSizingTests(unittest.TestCase):
    CONFIG = {
        "execution_position_fraction": 0.10,
        "execution_max_order_notional": 1000.0,
        "execution_min_order_notional": 1.0,
    }

    def _account(self, buying_power=10_000.0, positions=()):
        return AccountSnapshot(buying_power=buying_power, positions=tuple(positions))

    def test_hold_places_nothing(self):
        self.assertIsNone(build_order_intent("Hold", "AAPL", self._account(), self.CONFIG))

    def test_buy_is_sized_off_buying_power(self):
        intent = build_order_intent("Buy", "AAPL", self._account(), self.CONFIG)
        self.assertEqual(intent.side, "buy")
        self.assertEqual(intent.notional, 1000.0)  # 10_000 * 0.10 * 1.0

    def test_overweight_is_a_half_size_buy(self):
        intent = build_order_intent("Overweight", "AAPL", self._account(), self.CONFIG)
        self.assertEqual(intent.notional, 500.0)

    def test_notional_cap_is_enforced(self):
        config = {**self.CONFIG, "execution_max_order_notional": 250.0}
        intent = build_order_intent("Buy", "AAPL", self._account(), config)
        self.assertEqual(intent.notional, 250.0)

    def test_tiny_orders_are_skipped(self):
        config = {**self.CONFIG, "execution_min_order_notional": 100.0}
        account = self._account(buying_power=50.0)
        self.assertIsNone(build_order_intent("Buy", "AAPL", account, config))

    def test_buy_without_buying_power_is_skipped(self):
        self.assertIsNone(
            build_order_intent("Buy", "AAPL", self._account(buying_power=0.0), self.CONFIG)
        )
        self.assertIsNone(
            build_order_intent("Buy", "AAPL", self._account(buying_power=None), self.CONFIG)
        )

    def test_sell_is_sized_off_the_held_position(self):
        account = self._account(positions=[Position(symbol="AAPL", quantity=10)])
        intent = build_order_intent("Sell", "AAPL", account, self.CONFIG)
        self.assertEqual(intent.side, "sell")
        self.assertEqual(intent.quantity, 10)
        self.assertIsNone(intent.notional)

    def test_underweight_trims_half_the_position(self):
        account = self._account(positions=[Position(symbol="AAPL", quantity=10)])
        intent = build_order_intent("Underweight", "AAPL", account, self.CONFIG)
        self.assertEqual(intent.quantity, 5)

    def test_sell_with_no_position_is_a_noop_not_a_short(self):
        self.assertIsNone(build_order_intent("Sell", "AAPL", self._account(), self.CONFIG))

    def test_share_counts_round_down_unless_fractional_is_allowed(self):
        account = self._account(positions=[Position(symbol="AAPL", quantity=5)])
        self.assertEqual(
            build_order_intent("Underweight", "AAPL", account, self.CONFIG).quantity, 2
        )
        config = {**self.CONFIG, "execution_allow_fractional_shares": True}
        self.assertEqual(
            build_order_intent("Underweight", "AAPL", account, config).quantity, 2.5
        )

    def test_a_partial_trim_below_one_share_is_skipped(self):
        account = self._account(positions=[Position(symbol="AAPL", quantity=1)])
        self.assertIsNone(build_order_intent("Underweight", "AAPL", account, self.CONFIG))

    def test_rating_actions_are_configurable(self):
        config = {**self.CONFIG, "execution_rating_actions": {"Hold": ["buy", 0.25]}}
        intent = build_order_intent("Hold", "AAPL", self._account(), config)
        self.assertEqual(intent.notional, 250.0)

    def test_unmapped_rating_is_a_noop(self):
        self.assertIsNone(build_order_intent("Nonsense", "AAPL", self._account(), self.CONFIG))


class FakeBroker:
    def __init__(self, account=None, order=None, error=None):
        self._account = account or AccountSnapshot(buying_power=10_000.0)
        self._order = order or {"id": "order-1"}
        self._error = error
        self.orders = []
        self.closed = False

    def get_account(self, symbol=None):
        if self._error:
            raise self._error
        return self._account

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return self._order

    def close(self):
        self.closed = True


@pytest.mark.unit
class ExecuteDecisionTests(unittest.TestCase):
    BASE = {
        "execution_enabled": True,
        "execution_position_fraction": 0.10,
        "execution_max_order_notional": 1000.0,
        "execution_min_order_notional": 1.0,
    }

    def test_off_when_disabled(self):
        broker = FakeBroker()
        result = execute_decision("Buy", "AAPL", {}, broker=broker)
        self.assertEqual(result.mode, MODE_OFF)
        self.assertEqual(broker.orders, [])

    def test_dry_run_describes_the_order_without_sending_it(self):
        broker = FakeBroker()
        result = execute_decision("Buy", "AAPL", self.BASE, broker=broker)
        self.assertEqual(result.mode, MODE_DRY_RUN)
        self.assertFalse(result.placed)
        self.assertEqual(broker.orders, [])
        self.assertIsNotNone(result.intent)
        self.assertIn("BUY", result.to_markdown())
        self.assertIn("$1,000.00 notional", result.to_markdown())

    def test_live_without_arming_stays_a_dry_run(self):
        broker = FakeBroker()
        config = {**self.BASE, "execution_mode": "live"}
        env = {key: value for key, value in os.environ.items() if key != ARMED_ENV_VAR}
        with mock.patch.dict(os.environ, env, clear=True):
            result = execute_decision("Buy", "AAPL", config, broker=broker)
        self.assertEqual(result.mode, MODE_DRY_RUN)
        self.assertEqual(broker.orders, [])

    def test_live_and_armed_places_the_order(self):
        broker = FakeBroker()
        config = {**self.BASE, "execution_mode": "live"}
        with mock.patch.dict(os.environ, {ARMED_ENV_VAR: "1"}):
            result = execute_decision("Buy", "AAPL", config, broker=broker)
        self.assertEqual(result.mode, MODE_LIVE)
        self.assertTrue(result.placed)
        self.assertEqual(len(broker.orders), 1)
        self.assertEqual(broker.orders[0]["side"], "buy")
        self.assertEqual(broker.orders[0]["notional"], 1000.0)

    def test_hold_sends_nothing_even_when_armed(self):
        broker = FakeBroker()
        config = {**self.BASE, "execution_mode": "live"}
        with mock.patch.dict(os.environ, {ARMED_ENV_VAR: "1"}):
            result = execute_decision("Hold", "AAPL", config, broker=broker)
        self.assertFalse(result.placed)
        self.assertEqual(broker.orders, [])

    def test_broker_failure_is_reported_not_raised(self):
        # A brokerage outage must not discard a completed analysis run.
        broker = FakeBroker(error=BrokerToolError("account read failed"))
        result = execute_decision("Buy", "AAPL", self.BASE, broker=broker)
        self.assertFalse(result.placed)
        self.assertIn("account read failed", result.detail)

    def test_a_caller_supplied_broker_is_not_closed(self):
        broker = FakeBroker()
        execute_decision("Buy", "AAPL", self.BASE, broker=broker)
        self.assertFalse(broker.closed)


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------
# A server shaped the way a real brokerage plausibly is: every tool name and
# argument spelling differs from our canonical vocabulary, order placement has
# a required account id nothing can infer, and the side is an upper-case enum.
_REALISTIC_TOOLS = [
    _spec("get_account_details", description="Balances and buying power"),
    _spec("get_stock_positions", description="Open equity positions"),
    _spec("get_stock_quote", {"symbol": {"type": "string"}}),
    _spec(
        "place_equity_order",
        {
            "instrument": {"type": "string"},
            "action": {"type": "string", "enum": ["BUY", "SELL"]},
            "dollar_amount": {"type": "number"},
            "shares": {"type": "number"},
            "account_number": {"type": "string"},
        },
        required=("instrument", "action", "account_number"),
    ),
    _spec("cancel_stock_order", {"order_id": {"type": "string"}}),
    _spec("get_stock_orders"),
    _spec("get_watchlist"),
]


@pytest.mark.unit
class EndToEndTests(unittest.TestCase):
    CONFIG = {
        "execution_enabled": True,
        "execution_mode": "live",
        "execution_position_fraction": 0.05,
        "execution_max_order_notional": 1000.0,
        "execution_min_order_notional": 1.0,
    }

    def _broker(self):
        client = FakeClient(_REALISTIC_TOOLS, {
            "get_account_details": {"buying_power": "4,000.00", "portfolio_value": 21000},
            "get_stock_positions": [{"symbol": "AAPL", "quantity": "12"}],
            "place_equity_order": {"id": "order-9"},
        })
        return RobinhoodBroker(client, order_extra_args={"account_number": "RH-123"}), client

    def test_every_capability_resolves_from_unfamiliar_names(self):
        broker, _ = self._broker()
        self.assertEqual(broker.capabilities().mapping, {
            "account": "get_account_details",
            "positions": "get_stock_positions",
            "quote": "get_stock_quote",
            "place_order": "place_equity_order",
            "cancel_order": "cancel_stock_order",
            "list_orders": "get_stock_orders",
        })

    def test_unrelated_tools_stay_unmatched(self):
        broker, _ = self._broker()
        self.assertNotIn("get_watchlist", broker.capabilities().mapping.values())

    def test_buy_rating_reaches_the_server_correctly_translated(self):
        broker, client = self._broker()
        with mock.patch.dict(os.environ, {ARMED_ENV_VAR: "1"}):
            result = execute_decision("Buy", "AAPL", self.CONFIG, broker=broker)

        self.assertTrue(result.placed)
        name, args = client.calls[-1]
        self.assertEqual(name, "place_equity_order")
        self.assertEqual(args, {
            "instrument": "AAPL",       # symbol -> instrument
            "action": "BUY",            # side -> action, coerced to the enum
            "dollar_amount": 200.0,     # notional -> dollar_amount (4000 * 0.05)
            "account_number": "RH-123",  # from broker_order_extra_args
        })

    def test_sell_rating_sends_shares_from_the_held_position(self):
        broker, client = self._broker()
        with mock.patch.dict(os.environ, {ARMED_ENV_VAR: "1"}):
            execute_decision("Sell", "AAPL", self.CONFIG, broker=broker)
        _name, args = client.calls[-1]
        self.assertEqual(args["action"], "SELL")
        self.assertEqual(args["shares"], 12)
        self.assertNotIn("dollar_amount", args)

    def test_unarmed_run_reaches_the_server_only_for_reads(self):
        broker, client = self._broker()
        result = execute_decision("Buy", "AAPL", self.CONFIG, broker=broker)
        self.assertFalse(result.placed)
        self.assertNotIn("place_equity_order", [name for name, _ in client.calls])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

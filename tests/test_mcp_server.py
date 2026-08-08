"""Tests for the Model Context Protocol server surface.

The MCP SDK is an optional extra, so every test here skips when it is absent.
"""

import pytest

pytest.importorskip("mcp", reason="the MCP server needs the optional [mcp] extra")

import anyio  # noqa: E402

from tradingagents.mcp import server as mcp_server  # noqa: E402

pytestmark = pytest.mark.unit


def _tools_by_name(server):
    return {tool.name: tool for tool in anyio.run(server.list_tools)}


def _input_schema(tool):
    """Read a tool's JSON input schema across mcp 1.x / 2.x attribute names."""
    return getattr(tool, "input_schema", None) or tool.inputSchema


@pytest.fixture()
def server():
    return mcp_server.build_server()


class _FakeGraph:
    """Stand-in for TradingAgentsGraph that records how it was constructed."""

    last = None

    def __init__(self, selected_analysts=None, debug=False, config=None, callbacks=None):
        self.selected_analysts = list(selected_analysts or [])
        self.config = config
        self.propagated = None
        self.saved = None
        _FakeGraph.last = self

    def propagate(self, company_name, trade_date, asset_type="stock"):
        self.propagated = (company_name, trade_date, asset_type)
        state = {
            "market_report": "market text",
            "sentiment_report": "sentiment text",
            "news_report": "news text",
            "fundamentals_report": "fundamentals text",
            "investment_plan": "research manager text",
            "trader_investment_plan": "trader text",
            "final_trade_decision": "**Rating**: Overweight",
        }
        return state, "Overweight"

    def save_reports(self, final_state, ticker, save_path=None):
        self.saved = (ticker, save_path)
        return f"/tmp/reports/{ticker}/complete_report.md"


@pytest.fixture()
def fake_graph(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.TradingAgentsGraph", _FakeGraph
    )
    return _FakeGraph


def test_server_exposes_the_data_toolkit_and_the_pipeline(server):
    expected = [*mcp_server.DATA_TOOLS, "list_data_vendors", "run_trading_analysis"]
    assert list(_tools_by_name(server)) == expected


def test_every_data_tool_is_the_one_the_analysts_call():
    """The MCP surface is unwrapped from the agent toolkit, so it cannot drift."""
    from tradingagents.agents.utils import agent_utils

    for name in mcp_server.DATA_TOOLS:
        assert name in agent_utils.__all__


def test_argument_descriptions_survive_into_the_input_schema(server):
    schema = _input_schema(_tools_by_name(server)["get_indicators"])
    properties = schema["properties"]

    assert properties["symbol"]["description"] == "ticker symbol of the company"
    assert properties["look_back_days"]["description"] == "how many days to look back"
    assert properties["look_back_days"]["default"] == 30
    assert schema["required"] == ["symbol", "indicator", "curr_date"]


def test_optional_arguments_stay_optional(server):
    schema = _input_schema(_tools_by_name(server)["get_global_news"])

    assert schema["required"] == ["curr_date"]
    assert schema["properties"]["limit"]["default"] is None


def test_data_tool_call_routes_to_the_configured_vendor(server, monkeypatch):
    calls = []

    def fake_route(method, *args, **kwargs):
        calls.append((method, args, kwargs))
        return "OHLCV rows"

    monkeypatch.setattr(
        "tradingagents.agents.utils.core_stock_tools.route_to_vendor", fake_route
    )

    result = anyio.run(
        server.call_tool,
        "get_stock_data",
        {"symbol": "NVDA", "start_date": "2024-05-01", "end_date": "2024-05-10"},
    )

    assert calls == [("get_stock_data", ("NVDA", "2024-05-01", "2024-05-10"), {})]
    assert result.content[0].text == "OHLCV rows"


def test_list_data_vendors_reports_the_routing_table():
    table = mcp_server.list_data_vendors()

    assert "core_stock_apis" in table
    assert "get_macro_indicators" in table
    assert "polymarket" in table


def test_list_data_vendors_surfaces_tool_level_overrides():
    from tradingagents.dataflows.config import set_config

    set_config({"tool_vendors": {"get_stock_data": "alpha_vantage"}})

    assert "get_stock_data: alpha_vantage" in mcp_server.list_data_vendors()


def test_run_trading_analysis_rejects_a_malformed_date(fake_graph):
    with pytest.raises(ValueError, match="trade_date must be YYYY-MM-DD"):
        mcp_server.run_trading_analysis(ticker="NVDA", trade_date="05/10/2024")


def test_run_trading_analysis_rejects_unknown_analysts(fake_graph):
    with pytest.raises(ValueError, match="Unknown analyst"):
        mcp_server.run_trading_analysis(ticker="NVDA", analysts=["market", "astrology"])


def test_run_trading_analysis_returns_the_rating_and_reports(fake_graph):
    result = mcp_server.run_trading_analysis(ticker="nvda", trade_date="2024-05-10")

    assert result["ticker"] == "NVDA"
    assert result["trade_date"] == "2024-05-10"
    assert result["asset_type"] == "stock"
    assert result["rating"] == "Overweight"
    assert result["decision"] == "**Rating**: Overweight"
    assert result["reports"]["portfolio_manager"] == "**Rating**: Overweight"
    assert result["reports"]["research_manager"] == "research manager text"
    assert result["report_path"] is None
    assert fake_graph.last.propagated == ("NVDA", "2024-05-10", "stock")


def test_run_trading_analysis_can_omit_the_long_reports(fake_graph):
    result = mcp_server.run_trading_analysis(
        ticker="NVDA", trade_date="2024-05-10", include_reports=False
    )

    assert result["reports"] == {}
    # The decision itself is the point of the call, so it is always returned.
    assert result["decision"] == "**Rating**: Overweight"


def test_run_trading_analysis_can_save_the_report_tree(fake_graph):
    result = mcp_server.run_trading_analysis(
        ticker="NVDA", trade_date="2024-05-10", save_reports=True
    )

    assert result["report_path"] == "/tmp/reports/NVDA/complete_report.md"


def test_crypto_runs_drop_the_fundamentals_analyst(fake_graph):
    result = mcp_server.run_trading_analysis(ticker="BTCUSD", trade_date="2024-05-10")

    assert result["ticker"] == "BTC-USD"
    assert result["asset_type"] == "crypto"
    assert result["analysts"] == ["market", "social", "news"]
    assert fake_graph.last.propagated == ("BTC-USD", "2024-05-10", "crypto")


def test_a_crypto_run_of_only_fundamentals_is_rejected(fake_graph):
    with pytest.raises(ValueError, match="No analysts left to run"):
        mcp_server.run_trading_analysis(ticker="BTC-USD", analysts=["fundamentals"])


def test_research_depth_sets_both_debate_budgets(fake_graph):
    mcp_server.run_trading_analysis(
        ticker="NVDA", trade_date="2024-05-10", research_depth=3
    )

    assert fake_graph.last.config["max_debate_rounds"] == 3
    assert fake_graph.last.config["max_risk_discuss_rounds"] == 3


def test_llm_overrides_reach_the_graph_config(fake_graph):
    mcp_server.run_trading_analysis(
        ticker="NVDA",
        trade_date="2024-05-10",
        llm_provider="anthropic",
        deep_think_llm="claude-deep",
        quick_think_llm="claude-quick",
    )

    config = fake_graph.last.config
    assert config["llm_provider"] == "anthropic"
    assert config["deep_think_llm"] == "claude-deep"
    assert config["quick_think_llm"] == "claude-quick"
    # A provider switch must not inherit the previous provider's endpoint.
    assert config["backend_url"] is None


def test_the_run_does_not_mutate_the_shared_default_config(fake_graph):
    from tradingagents.default_config import DEFAULT_CONFIG

    before = DEFAULT_CONFIG["max_debate_rounds"]
    mcp_server.run_trading_analysis(
        ticker="NVDA", trade_date="2024-05-10", research_depth=before + 5
    )

    assert DEFAULT_CONFIG["max_debate_rounds"] == before

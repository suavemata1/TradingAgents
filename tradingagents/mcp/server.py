"""MCP server exposing the TradingAgents toolkit over the Model Context Protocol.

Two kinds of tool are served:

* **Data tools** — the vendor-routed toolkit the analyst agents themselves call
  (``get_stock_data``, ``get_news``, ``get_macro_indicators``, ...). They are
  registered by unwrapping the LangChain tool objects in
  :mod:`tradingagents.agents.utils.agent_utils`, so the MCP surface cannot
  drift from what an in-graph analyst sees. Each returns in seconds and needs
  only the relevant vendor key (Alpha Vantage, FRED), not an LLM key.
* **``run_trading_analysis``** — the full multi-agent graph, returning the
  Portfolio Manager's rating and the per-team reports. It needs an LLM provider
  key and takes minutes, not seconds.

Start it with ``tradingagents-mcp`` (or ``python -m tradingagents.mcp``). The
default transport is stdio, which is what Claude Desktop / Claude Code launch.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import logging
import threading
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, TypedDict

from pydantic import Field

from tradingagents.agents.utils import agent_utils
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import TOOLS_CATEGORIES, VENDOR_METHODS, get_vendor
from tradingagents.dataflows.symbol_utils import crypto_base, normalize_symbol
from tradingagents.default_config import DEFAULT_CONFIG

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - only reached on mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP as _Server
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise ImportError(
            "The TradingAgents MCP server needs the Model Context Protocol SDK. "
            'Install it with: pip install "tradingagents[mcp]"'
        ) from exc

logger = logging.getLogger(__name__)

SERVER_NAME = "tradingagents"

SERVER_INSTRUCTIONS = """\
TradingAgents exposes the market data layer that its analyst agents run on,
plus the full multi-agent analysis pipeline.

Use the individual data tools (prices, indicators, fundamentals, news, macro
series, prediction markets) for quick lookups — they are deterministic vendor
calls that return in seconds. Use get_verified_market_snapshot before making
exact claims about price levels or indicator values.

Use run_trading_analysis only when a full analyst-debate-and-decision run is
actually wanted: it drives several LLM agents through a debate and takes
minutes to complete.

All dates are YYYY-MM-DD. Tickers are Yahoo Finance symbols, so non-US markets
need their exchange suffix (0700.HK, 7203.T, RELIANCE.NS) and crypto is quoted
as BASE-USD (BTC-USD).\
"""

# Data tools re-exported by the agent toolkit, in the order they are advertised.
DATA_TOOLS = (
    "get_stock_data",
    "get_indicators",
    "get_verified_market_snapshot",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
)

ANALYSTS = ("market", "social", "news", "fundamentals")

# Final-state keys worth returning, mapped to the names used in the reports.
REPORT_KEYS = {
    "market_report": "market",
    "sentiment_report": "sentiment",
    "news_report": "news",
    "fundamentals_report": "fundamentals",
    "investment_plan": "research_manager",
    "trader_investment_plan": "trader",
    "final_trade_decision": "portfolio_manager",
}

# ``TradingAgentsGraph`` publishes its config to the process-global dataflows
# config, so two concurrent runs with different vendor/LLM settings would race.
# MCP dispatches sync tools onto a thread pool, so serialise runs here.
_ANALYSIS_LOCK = threading.Lock()


class TradingAnalysis(TypedDict):
    """Structured result of a full multi-agent run."""

    ticker: str
    trade_date: str
    asset_type: str
    analysts: list[str]
    rating: str
    decision: str
    reports: dict[str, str]
    report_path: str | None


def _package_version() -> str:
    """The installed tradingagents version, or empty when running from a checkout."""
    try:
        return version("tradingagents")
    except PackageNotFoundError:  # pragma: no cover - source tree without install
        return ""


def _with_field_descriptions(annotation: Any) -> Any:
    """Turn ``Annotated[T, "text"]`` into ``Annotated[T, Field(description="text")]``.

    The toolkit documents each parameter with a bare string in ``Annotated``,
    which LangChain reads as a field description but pydantic ignores. Promoting
    it to a real ``Field`` keeps that help text in the MCP input schema.
    """
    metadata = getattr(annotation, "__metadata__", None)
    if not metadata:
        return annotation
    described = tuple(
        Field(description=item) if isinstance(item, str) else item for item in metadata
    )
    return Annotated[(annotation.__origin__, *described)]


def _as_plain_function(lc_tool: Any):
    """Expose a LangChain ``@tool`` object as a plain function MCP can introspect."""
    fn = lc_tool.func
    signature = inspect.signature(fn)
    described = signature.replace(
        parameters=[
            parameter.replace(annotation=_with_field_descriptions(parameter.annotation))
            for parameter in signature.parameters.values()
        ]
    )

    def call(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    call.__name__ = lc_tool.name
    call.__doc__ = fn.__doc__
    # Set explicitly rather than via functools.wraps: inspect.signature would
    # otherwise follow __wrapped__ back to the undescribed original.
    call.__signature__ = described
    return call


def list_data_vendors() -> str:
    """List which data vendor currently serves each category of TradingAgents tool.

    Use this to find out where a number came from, or why a tool reports a
    missing API key. "default" means no vendor was pinned, so the whole
    available chain is tried in order.

    Returns:
        str: A markdown table of category, tools, configured vendor chain, and
            the vendors available for that category.
    """
    lines = [
        "| Category | Tools | Configured vendor chain | Available vendors |",
        "| --- | --- | --- | --- |",
    ]
    for category, info in TOOLS_CATEGORIES.items():
        available = sorted({v for tool in info["tools"] for v in VENDOR_METHODS.get(tool, {})})
        lines.append(
            f"| {category} | {', '.join(info['tools'])} | "
            f"{get_vendor(category)} | {', '.join(available)} |"
        )

    overrides = get_config().get("tool_vendors") or {}
    if overrides:
        lines.append("")
        lines.append("Tool-level overrides (these win over the category setting):")
        lines += [f"- {tool}: {vendor}" for tool, vendor in sorted(overrides.items())]
    return "\n".join(lines)


def run_trading_analysis(
    ticker: Annotated[
        str, Field(description="Yahoo Finance symbol, e.g. NVDA, 0700.HK, BTC-USD")
    ],
    trade_date: Annotated[
        str, Field(description="Analysis date in YYYY-MM-DD format; omit for today")
    ] = "",
    analysts: Annotated[
        list[str] | None,
        Field(
            description="Subset of market, social, news, fundamentals; omit to run all "
            "of them. The fundamentals analyst is skipped for crypto."
        ),
    ] = None,
    research_depth: Annotated[
        int, Field(description="Bull/bear and risk debate rounds; higher is slower", ge=1)
    ] = 1,
    llm_provider: Annotated[
        str | None, Field(description="Override the configured LLM provider, e.g. anthropic")
    ] = None,
    deep_think_llm: Annotated[
        str | None, Field(description="Override the deep-thinking model ID")
    ] = None,
    quick_think_llm: Annotated[
        str | None, Field(description="Override the quick-thinking model ID")
    ] = None,
    include_reports: Annotated[
        bool, Field(description="Include the full per-team reports; they are long")
    ] = True,
    save_reports: Annotated[
        bool, Field(description="Also write the markdown report tree under results_dir")
    ] = False,
) -> TradingAnalysis:
    """Run the full TradingAgents multi-agent pipeline and return its decision.

    Analysts gather market, sentiment, news, and fundamentals evidence; bull and
    bear researchers debate it; a trader drafts a plan; the risk team stress-tests
    it; and the Portfolio Manager issues a five-tier rating (Buy, Overweight,
    Hold, Underweight, Sell).

    This drives many LLM calls and takes minutes. It needs a configured LLM
    provider key. For a quick data lookup, call the individual data tools instead.

    Returns:
        TradingAnalysis: The rating, the Portfolio Manager's full decision text,
            and each team's report.
    """
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    trade_date = (trade_date or datetime.now().strftime("%Y-%m-%d")).strip()
    try:
        datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"trade_date must be YYYY-MM-DD, got {trade_date!r}") from exc

    if research_depth < 1:
        raise ValueError(f"research_depth must be >= 1, got {research_depth}")

    requested = [name.strip().lower() for name in (analysts or ANALYSTS) if name.strip()]
    unknown = sorted({name for name in requested if name not in ANALYSTS})
    if unknown:
        raise ValueError(f"Unknown analyst(s) {unknown}. Choose from {list(ANALYSTS)}.")
    selected = list(dict.fromkeys(requested))

    symbol = normalize_symbol(ticker)
    asset_type = "crypto" if crypto_base(symbol) else "stock"
    if asset_type == "crypto":
        # Crypto has no filings to analyse; the CLI drops this analyst too.
        selected = [name for name in selected if name != "fundamentals"]
    if not selected:
        raise ValueError(f"No analysts left to run for {symbol}. Choose from {list(ANALYSTS)}.")

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["max_debate_rounds"] = research_depth
    config["max_risk_discuss_rounds"] = research_depth
    for key, override in (
        ("llm_provider", llm_provider),
        ("deep_think_llm", deep_think_llm),
        ("quick_think_llm", quick_think_llm),
    ):
        if override:
            config[key] = override
    if llm_provider:
        # The default endpoint belongs to whichever provider was configured, so
        # a provider switch must not inherit it (see DEFAULT_CONFIG backend_url).
        config["backend_url"] = None

    logger.info(
        "Running analysis for %s on %s (%s, analysts=%s, depth=%d)",
        symbol, trade_date, asset_type, selected, research_depth,
    )

    with _ANALYSIS_LOCK:
        graph = TradingAgentsGraph(selected_analysts=selected, config=config)
        final_state, rating = graph.propagate(symbol, trade_date, asset_type=asset_type)
        report_path = str(graph.save_reports(final_state, symbol)) if save_reports else None

    reports: dict[str, str] = {}
    if include_reports:
        reports = {
            label: final_state[key]
            for key, label in REPORT_KEYS.items()
            if final_state.get(key)
        }

    return TradingAnalysis(
        ticker=symbol,
        trade_date=trade_date,
        asset_type=asset_type,
        analysts=selected,
        rating=rating,
        decision=final_state.get("final_trade_decision", ""),
        reports=reports,
        report_path=report_path,
    )


def build_server() -> Any:
    """Build the MCP server with the data toolkit and the analysis pipeline registered."""
    kwargs: dict[str, Any] = {"name": SERVER_NAME, "instructions": SERVER_INSTRUCTIONS}
    if "version" in inspect.signature(_Server).parameters:
        # mcp >= 2.0 advertises a server version to the client; 1.x has no such
        # parameter and would reject the keyword.
        kwargs["version"] = _package_version()
    server = _Server(**kwargs)

    for name in DATA_TOOLS:
        lc_tool = getattr(agent_utils, name)
        server.add_tool(
            _as_plain_function(lc_tool),
            name=lc_tool.name,
            description=lc_tool.description,
        )

    server.add_tool(list_data_vendors)
    server.add_tool(run_trading_analysis)
    return server


def main() -> None:
    """Entry point for the ``tradingagents-mcp`` console script."""
    parser = argparse.ArgumentParser(
        prog="tradingagents-mcp",
        description="Serve the TradingAgents toolkit over the Model Context Protocol.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="Transport to serve on (default: stdio, what MCP clients launch).",
    )
    args = parser.parse_args()

    # stdio carries the protocol itself, so logs must go to stderr — which is
    # where basicConfig sends them.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_server().run(transport=args.transport)

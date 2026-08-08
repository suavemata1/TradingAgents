"""Model Context Protocol server for TradingAgents.

Importing this package requires the optional MCP SDK:
``pip install "tradingagents[mcp]"``.
"""

from tradingagents.mcp.server import build_server, main

__all__ = ["build_server", "main"]

"""Allow ``python -m tradingagents.mcp`` alongside the ``tradingagents-mcp`` script."""

from tradingagents.mcp.server import main

if __name__ == "__main__":
    main()

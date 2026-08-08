"""Broker integration wiring: account context reaching the position-taking
agents, and the graph opening/closing a broker only when asked to.

Complements ``test_robinhood_broker.py``, which covers the broker itself.
These assert the *rendered* prompts and the graph's lifecycle, so a refactor
that drops the context or leaks a connection fails here rather than silently
running against an assumed-flat book.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import get_account_context_block
from tradingagents.brokers.errors import BrokerNotConfiguredError
from tradingagents.brokers.robinhood import AccountSnapshot, Position
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.propagation import Propagator

ACCOUNT_TEXT = "**Live brokerage account (Robinhood)**\n- Buying power: $2,500.00"


def _capturing_llm(captured: dict, result):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or result
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for message in prompt:
        parts.append(
            message.get("content", "")
            if isinstance(message, dict)
            else getattr(message, "content", "")
        )
    return "\n".join(str(part) for part in parts)


def _risk_debate_state():
    return {
        "history": "debate",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "count": 0,
    }


# ----------------------------------------------------------------------
# Context block
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_absent_account_context_costs_no_tokens():
    assert get_account_context_block({}) == ""
    assert get_account_context_block({"account_context": "   "}) == ""


@pytest.mark.unit
def test_present_account_context_is_labelled_and_instructive():
    block = get_account_context_block({"account_context": ACCOUNT_TEXT})
    assert ACCOUNT_TEXT in block
    assert "Size your recommendation against this real account state" in block


@pytest.mark.unit
def test_snapshot_markdown_reports_the_symbol_position():
    snapshot = AccountSnapshot(
        buying_power=2500.0,
        portfolio_value=10_000.0,
        positions=(Position(symbol="AAPL", quantity=10, market_value=2000.0),),
    )
    text = snapshot.to_markdown("AAPL")
    assert "Buying power: $2,500.00" in text
    assert "Current position in AAPL" in text
    assert "10 shares" in text


@pytest.mark.unit
def test_snapshot_markdown_states_a_flat_position_explicitly():
    # "none" beats omission: the agent must not read silence as "unknown".
    text = AccountSnapshot(buying_power=100.0).to_markdown("TSLA")
    assert "Current position in TSLA: none" in text


# ----------------------------------------------------------------------
# Prompt injection
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_trader_prompt_carries_account_context():
    from tradingagents.agents.schemas import TraderAction, TraderProposal

    captured = {}
    llm = _capturing_llm(captured, TraderProposal(action=TraderAction.BUY, reasoning="x"))
    create_trader(llm)({
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy",
        "account_context": ACCOUNT_TEXT,
    })
    assert "Buying power: $2,500.00" in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_portfolio_manager_prompt_carries_account_context():
    from tradingagents.agents.schemas import PortfolioDecision

    captured = {}
    decision = PortfolioDecision(
        rating="Buy", executive_summary="summary", investment_thesis="thesis"
    )
    llm = _capturing_llm(captured, decision)
    create_portfolio_manager(llm)({
        "company_of_interest": "NVDA",
        "investment_plan": "plan",
        "trader_investment_plan": "proposal",
        "risk_debate_state": _risk_debate_state(),
        "account_context": ACCOUNT_TEXT,
    })
    assert "Buying power: $2,500.00" in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_prompts_are_unchanged_without_a_broker():
    from tradingagents.agents.schemas import TraderAction, TraderProposal

    captured = {}
    llm = _capturing_llm(captured, TraderProposal(action=TraderAction.BUY, reasoning="x"))
    create_trader(llm)({
        "company_of_interest": "NVDA",
        "investment_plan": "**Recommendation**: Buy",
    })
    assert "Live brokerage account" not in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_initial_state_defaults_account_context_to_empty():
    state = Propagator().create_initial_state("NVDA", "2026-01-15")
    assert state["account_context"] == ""


# ----------------------------------------------------------------------
# Graph lifecycle
# ----------------------------------------------------------------------
def _graph(config_overrides):
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = {**DEFAULT_CONFIG, **config_overrides}
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch("tradingagents.graph.trading_graph.create_llm_client", return_value=client):
        return TradingAgentsGraph(config=config)


@pytest.mark.unit
def test_broker_is_not_opened_by_default():
    graph = _graph({})
    assert graph._broker_requested() is False
    assert graph._open_broker() is None
    assert graph.resolve_account_context("NVDA") == ""
    assert graph.execute_decision("NVDA", "Buy") is None


@pytest.mark.unit
def test_either_switch_requests_a_broker():
    assert _graph({"broker_context_enabled": True})._broker_requested() is True
    assert _graph({"execution_enabled": True})._broker_requested() is True


@pytest.mark.unit
def test_unconfigured_broker_degrades_instead_of_failing_the_run():
    # A missing token must cost the run its account context, not the run.
    graph = _graph({"broker_context_enabled": True})
    with patch(
        "tradingagents.brokers.robinhood.RobinhoodBroker.from_config",
        side_effect=BrokerNotConfiguredError("no token"),
    ):
        assert graph._open_broker() is None
    assert graph.resolve_account_context("NVDA") == ""


@pytest.mark.unit
def test_account_context_is_read_from_the_open_broker():
    graph = _graph({"broker_context_enabled": True})
    broker = MagicMock()
    broker.get_account.return_value = AccountSnapshot(buying_power=2500.0)
    graph._broker = broker

    context = graph.resolve_account_context("NVDA")
    assert "Buying power: $2,500.00" in context
    broker.get_account.assert_called_once_with("NVDA")


@pytest.mark.unit
def test_context_read_failure_does_not_propagate():
    graph = _graph({"broker_context_enabled": True})
    broker = MagicMock()
    broker.get_account.side_effect = BrokerNotConfiguredError("session lost")
    graph._broker = broker
    assert graph.resolve_account_context("NVDA") == ""


@pytest.mark.unit
def test_execute_decision_reuses_the_run_broker():
    graph = _graph({"execution_enabled": True})
    broker = MagicMock()
    graph._broker = broker
    with patch("tradingagents.graph.trading_graph.execute_decision") as executor:
        graph.execute_decision("NVDA", "Buy")
    _args, kwargs = executor.call_args
    assert kwargs["broker"] is broker

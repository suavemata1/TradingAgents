"""Turn a Portfolio Manager rating into a broker order — behind two gates.

TradingAgents is research software. Nothing here places a real order unless
**both** of these are true:

1. ``execution_enabled`` is True *and* ``execution_mode`` is ``"live"`` in the
   run config, and
2. the ``TRADINGAGENTS_EXECUTION_ARMED`` environment variable is truthy.

Two independent switches, one of which cannot live in a committed config file
or a shared ``DEFAULT_CONFIG`` override. A copied config, a stale environment,
or a single mistaken edit therefore cannot by itself send money to a broker.
Every other combination resolves to a dry run that reports the order it
*would* have placed, which is also what you get if the live gate is only
half-open — loudly, via a warning.

Broker failures are non-fatal by design: a completed analysis is worth
returning even when the brokerage leg is down, so failures come back as an
:class:`ExecutionResult` describing what went wrong rather than as an
exception that discards the run.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import BrokerError
from .robinhood import AccountSnapshot, RobinhoodBroker

logger = logging.getLogger(__name__)

ARMED_ENV_VAR = "TRADINGAGENTS_EXECUTION_ARMED"

MODE_OFF = "off"
MODE_DRY_RUN = "dry_run"
MODE_LIVE = "live"

_TRUTHY = ("1", "true", "yes", "on")

# Rating -> (side, fraction of a full-size position). Hold is the no-op.
# Overweight/Underweight are half-size adjustments rather than full entries or
# exits, matching how the 5-tier scale is described to the Portfolio Manager.
DEFAULT_RATING_ACTIONS: dict[str, tuple[str | None, float]] = {
    "Buy": ("buy", 1.0),
    "Overweight": ("buy", 0.5),
    "Hold": (None, 0.0),
    "Underweight": ("sell", 0.5),
    "Sell": ("sell", 1.0),
}


@dataclass(frozen=True)
class OrderIntent:
    """A fully-sized order, before any decision about whether to send it."""

    symbol: str
    side: str
    quantity: float | None = None
    notional: float | None = None
    order_type: str = "market"
    limit_price: float | None = None
    time_in_force: str | None = None

    def describe(self) -> str:
        size = (
            f"${self.notional:,.2f} notional"
            if self.notional is not None
            else f"{self.quantity:g} shares"
        )
        text = f"{self.side.upper()} {size} of {self.symbol} ({self.order_type})"
        if self.limit_price is not None:
            text += f" @ {self.limit_price:,.2f}"
        if self.time_in_force:
            text += f", {self.time_in_force.upper()}"
        return text


@dataclass(frozen=True)
class ExecutionResult:
    """What the execution layer did, or declined to do, and why."""

    mode: str
    rating: str
    detail: str
    intent: OrderIntent | None = None
    placed: bool = False
    order: Any = None

    def to_markdown(self) -> str:
        heading = {
            MODE_OFF: "Execution: disabled",
            MODE_DRY_RUN: "Execution: dry run (no order sent)",
            MODE_LIVE: "Execution: live" if self.placed else "Execution: live (no order sent)",
        }.get(self.mode, f"Execution: {self.mode}")

        lines = [f"**{heading}**", "", f"- Rating: {self.rating}", f"- Outcome: {self.detail}"]
        if self.intent is not None:
            lines.append(f"- Order: {self.intent.describe()}")
        return "\n".join(lines)


def resolve_execution_mode(
    config: Mapping[str, Any], env: Mapping[str, str] | None = None
) -> str:
    """Resolve the effective execution mode from config plus environment.

    Returns ``"off"``, ``"dry_run"``, or ``"live"``. ``"live"`` requires the
    config to ask for it *and* the arming environment variable to be set;
    asking for live without arming downgrades to a dry run with a warning
    rather than failing, so an unattended run still completes its analysis.
    """
    env = os.environ if env is None else env

    if not config.get("execution_enabled"):
        return MODE_OFF

    requested = str(config.get("execution_mode") or MODE_DRY_RUN).strip().lower()
    if requested not in (MODE_DRY_RUN, MODE_LIVE):
        raise ValueError(
            f"execution_mode must be {MODE_DRY_RUN!r} or {MODE_LIVE!r}, got {requested!r}"
        )
    if requested == MODE_DRY_RUN:
        return MODE_DRY_RUN

    if str(env.get(ARMED_ENV_VAR, "")).strip().lower() in _TRUTHY:
        return MODE_LIVE

    logger.warning(
        "execution_mode is 'live' but %s is not set — falling back to a dry run. "
        "Set %s=1 in the environment to arm real order placement.",
        ARMED_ENV_VAR,
        ARMED_ENV_VAR,
    )
    return MODE_DRY_RUN


def _rating_action(rating: str, config: Mapping[str, Any]) -> tuple[str | None, float]:
    """Look up ``(side, weight)`` for a 5-tier rating."""
    actions = {**DEFAULT_RATING_ACTIONS, **(config.get("execution_rating_actions") or {})}
    action = actions.get(rating)
    if action is None:
        logger.warning("Rating %r has no execution mapping; treating it as no-op", rating)
        return None, 0.0
    # Tolerate lists as well as tuples so the mapping survives a JSON round-trip.
    side, weight = tuple(action)
    return (str(side).lower() if side else None), float(weight)


def _round_shares(quantity: float, config: Mapping[str, Any]) -> float:
    """Round a share count down to what the account is allowed to trade."""
    if config.get("execution_allow_fractional_shares"):
        return round(quantity, 6)
    return float(math.floor(quantity))


def build_order_intent(
    rating: str,
    symbol: str,
    account: AccountSnapshot,
    config: Mapping[str, Any],
) -> OrderIntent | None:
    """Size an order for ``rating``, or return None when it implies no trade.

    Buys are sized in notional dollars off buying power, which lets the broker
    handle fractional shares and avoids a quote round-trip. Sells are sized in
    shares off the existing position, because you cannot sell what you do not
    hold — a sell rating with no position is simply a no-op, not an error.
    """
    side, weight = _rating_action(rating, config)
    if side is None or weight <= 0:
        return None

    order_type = str(config.get("execution_order_type") or "market").lower()
    time_in_force = config.get("execution_time_in_force") or None

    if side == "sell":
        held = account.position_for(symbol)
        if held is None or held.quantity <= 0:
            return None
        quantity = _round_shares(held.quantity * weight, config)
        if quantity <= 0:
            return None
        return OrderIntent(
            symbol=str(symbol).upper(),
            side="sell",
            quantity=quantity,
            order_type=order_type,
            time_in_force=time_in_force,
        )

    budget = account.buying_power
    if budget is None or budget <= 0:
        return None

    fraction = float(config.get("execution_position_fraction", 0.05))
    notional = budget * fraction * weight

    cap = config.get("execution_max_order_notional")
    if cap:
        notional = min(notional, float(cap))

    minimum = float(config.get("execution_min_order_notional", 1.0))
    if notional < minimum:
        return None

    return OrderIntent(
        symbol=str(symbol).upper(),
        side="buy",
        notional=round(notional, 2),
        order_type=order_type,
        time_in_force=time_in_force,
    )


def execute_decision(
    rating: str,
    symbol: str,
    config: Mapping[str, Any],
    broker: RobinhoodBroker | None = None,
) -> ExecutionResult:
    """Act on a Portfolio Manager rating according to the configured mode.

    Pass an existing ``broker`` to reuse a connection opened earlier in the run;
    when omitted, one is created and closed here.
    """
    mode = resolve_execution_mode(config)
    if mode == MODE_OFF:
        return ExecutionResult(
            MODE_OFF, rating, "Execution is disabled (config execution_enabled=False)."
        )

    owns_broker = broker is None
    try:
        if broker is None:
            broker = RobinhoodBroker.from_config(config)

        account = broker.get_account(symbol)
        intent = build_order_intent(rating, symbol, account, config)
        if intent is None:
            return ExecutionResult(
                mode, rating, f"Rating {rating!r} implies no order for {symbol.upper()}."
            )

        if mode == MODE_DRY_RUN:
            return ExecutionResult(
                mode,
                rating,
                f"Dry run — would place: {intent.describe()}. "
                f"Set execution_mode='live' and {ARMED_ENV_VAR}=1 to send it.",
                intent=intent,
            )

        logger.info("Placing live order: %s", intent.describe())
        order = broker.place_order(
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            notional=intent.notional,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            time_in_force=intent.time_in_force,
        )
        return ExecutionResult(
            MODE_LIVE,
            rating,
            f"Order submitted: {intent.describe()}",
            intent=intent,
            placed=True,
            order=order,
        )
    except (BrokerError, ValueError) as exc:
        logger.warning("Execution did not complete for %s: %s", symbol, exc)
        return ExecutionResult(mode, rating, f"Execution failed: {exc}")
    finally:
        if owns_broker and broker is not None:
            broker.close()

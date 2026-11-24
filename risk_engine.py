
"""Pure helpers for closed-candle filtering, paper risk checks and performance metrics."""
from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Iterable
import math
import pandas as pd


def filter_completed_candles(df: pd.DataFrame, interval_minutes: int, now: datetime,
                             daily: bool = False) -> pd.DataFrame:
    """Keep only completed bars. Intraday index timestamps are bar start times.
    Daily bars are complete only after their local trading date has ended.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is None:
        idx = idx.tz_localize(now.tzinfo)
    else:
        idx = idx.tz_convert(now.tzinfo)
    out.index = idx
    if daily:
        return out.loc[out.index.date < now.date()]
    cutoff = now - timedelta(minutes=int(interval_minutes))
    return out.loc[out.index <= cutoff]


def risk_check(entry: float, stop: float, quantity: int, capital: float,
               risk_pct: float, realized_today: float, daily_loss_pct: float,
               open_positions: int, max_open_positions: int,
               trades_today: int, max_trades_per_day: int) -> dict:
    """Return a deterministic paper-trade risk decision."""
    vals = [entry, stop, capital, risk_pct, daily_loss_pct]
    if any(not math.isfinite(float(v)) for v in vals):
        return {"allowed": False, "reason": "Risk inputs are invalid.", "risk_amount": math.nan}
    if entry <= 0 or stop <= 0 or entry == stop or quantity < 1 or capital <= 0:
        return {"allowed": False, "reason": "Entry, stop, capital or quantity is invalid.", "risk_amount": math.nan}
    risk_amount = abs(entry - stop) * int(quantity)
    max_risk = capital * risk_pct / 100.0
    daily_loss_limit = capital * daily_loss_pct / 100.0
    if risk_amount > max_risk:
        return {"allowed": False, "reason": f"Trade risk ₹{risk_amount:,.2f} exceeds per-trade limit ₹{max_risk:,.2f}.", "risk_amount": risk_amount}
    if realized_today <= -daily_loss_limit:
        return {"allowed": False, "reason": f"Daily loss limit ₹{daily_loss_limit:,.2f} has been reached.", "risk_amount": risk_amount}
    if open_positions >= max_open_positions:
        return {"allowed": False, "reason": f"Maximum open paper positions ({max_open_positions}) reached.", "risk_amount": risk_amount}
    if trades_today >= max_trades_per_day:
        return {"allowed": False, "reason": f"Maximum paper trades per day ({max_trades_per_day}) reached.", "risk_amount": risk_amount}
    return {"allowed": True, "reason": "Risk checks passed.", "risk_amount": risk_amount}


def performance_metrics(trades: pd.DataFrame) -> dict:
    """Metrics from closed paper trades; PnL should already include chosen costs if modeled."""
    if trades is None or trades.empty or "pnl" not in trades:
        return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "expectancy": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "expectancy": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), abs(float(losses.sum()))
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean() * 100),
        "net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0),
        "max_drawdown": float(abs(drawdown.min())) if len(drawdown) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }

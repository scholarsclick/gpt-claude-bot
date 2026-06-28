"""
risk_manager.py
===============
Risk controls + trade ledger (SQLite).

Hard rules enforced here (the strategy can want to trade; the risk manager has
final veto):
  * fixed small stake per trade (no martingale, no position sizing on losses)
  * daily max loss -> halt for the day
  * max consecutive losses -> halt
  * max drawdown from peak equity -> halt
  * max 1 open trade per asset per window
  * paper vs live separation in the ledger
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("risk")


@dataclass
class RiskState:
    allowed: bool
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        r = cfg["risk"]
        self.stake = r["stake_per_trade"]
        self.daily_max_loss = r["daily_max_loss"]
        self.max_consec = r["max_consecutive_losses"]
        self.max_dd = r["max_drawdown"]
        self.max_open_per_asset = r["max_open_trades_per_asset"]
        self.starting_bankroll = r["starting_bankroll"]
        db_path = cfg["database"]["path"]
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, mode TEXT, asset TEXT, market_id TEXT,
                side TEXT, stake REAL, entry_price REAL,
                model_prob REAL, implied_prob REAL, expected_value REAL,
                target_price REAL, status TEXT,
                resolved_price REAL, outcome TEXT, pnl REAL, reasons TEXT
            );
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, mode TEXT, equity REAL
            );
            """
        )
        self.conn.commit()

    # ----------------------------------------------------------- gating
    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def realized_pnl_today(self, mode: str) -> float:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='resolved' "
            "AND mode=? AND substr(ts,1,10)=?", (mode, self._today()))
        return float(cur.fetchone()[0] or 0.0)

    def consecutive_losses(self, mode: str) -> int:
        cur = self.conn.execute(
            "SELECT outcome FROM trades WHERE status='resolved' AND mode=? "
            "ORDER BY id DESC LIMIT 20", (mode,))
        n = 0
        for (outcome,) in cur.fetchall():
            if outcome == "loss":
                n += 1
            else:
                break
        return n

    def equity(self, mode: str) -> float:
        pnl = self.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='resolved' AND mode=?",
            (mode,)).fetchone()[0] or 0.0
        return self.starting_bankroll + float(pnl)

    def peak_equity(self, mode: str) -> float:
        cur = self.conn.execute("SELECT MAX(equity) FROM equity WHERE mode=?", (mode,))
        peak = cur.fetchone()[0]
        return float(peak) if peak is not None else self.starting_bankroll

    def open_trades(self, mode: str, asset: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status='open' AND mode=? AND asset=?",
            (mode, asset))
        return int(cur.fetchone()[0])

    def can_trade(self, mode: str, asset: str) -> RiskState:
        if self.realized_pnl_today(mode) <= -abs(self.daily_max_loss):
            return RiskState(False, "daily max loss reached — halted for today")
        if self.consecutive_losses(mode) >= self.max_consec:
            return RiskState(False, f"{self.max_consec} consecutive losses — halted")
        eq = self.equity(mode)
        dd = self.peak_equity(mode) - eq
        if dd >= self.max_dd:
            return RiskState(False, f"max drawdown {dd:.0f} reached — halted")
        if self.open_trades(mode, asset) >= self.max_open_per_asset:
            return RiskState(False, f"already {self.max_open_per_asset} open trade on {asset}")
        return RiskState(True, "ok")

    # ----------------------------------------------------------- ledger
    def open_trade(self, mode, asset, market_id, signal, target_price) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO trades (ts,mode,asset,market_id,side,stake,entry_price,"
            "model_prob,implied_prob,expected_value,target_price,status,reasons) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?, 'open', ?)",
            (ts, mode, asset, str(market_id), signal.side, self.stake,
             signal.entry_price,
             signal.probability_up if signal.side == "UP" else signal.probability_down,
             signal.implied_prob_up, signal.expected_value, target_price,
             "; ".join(signal.reasons)))
        self.conn.commit()
        log.info("[%s] OPEN %s %s stake=%.2f entry=%.3f EV=%.1f%%",
                 mode, asset, signal.side, self.stake,
                 signal.entry_price or 0, signal.expected_value * 100)
        return int(cur.lastrowid)

    def resolve_trade(self, trade_id: int, resolved_price: float):
        row = self.conn.execute(
            "SELECT mode,side,stake,entry_price,target_price FROM trades WHERE id=?",
            (trade_id,)).fetchone()
        if not row:
            return
        mode, side, stake, entry_price, target_price = row
        went_up = resolved_price > target_price
        won = (side == "UP" and went_up) or (side == "DOWN" and not went_up)
        # Polymarket payout: buy `shares = stake/entry_price`, each pays $1 if won.
        if entry_price and entry_price > 0:
            shares = stake / entry_price
            pnl = (shares - stake) if won else (-stake)
        else:
            pnl = stake if won else -stake
        outcome = "win" if won else "loss"
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE trades SET status='resolved', resolved_price=?, outcome=?, pnl=? WHERE id=?",
            (resolved_price, outcome, pnl, trade_id))
        # snapshot equity for drawdown tracking
        eq = self.equity(mode)
        peak = max(self.peak_equity(mode), eq)
        self.conn.execute("INSERT INTO equity (ts,mode,equity) VALUES (?,?,?)",
                          (ts, mode, eq))
        self.conn.commit()
        log.info("[%s] RESOLVE #%d %s -> %s pnl=%.2f equity=%.2f (peak=%.2f)",
                 mode, trade_id, side, outcome, pnl, eq, peak)
        return outcome, pnl

    def recent_trades(self, mode: str, limit: int = 50):
        cur = self.conn.execute(
            "SELECT * FROM trades WHERE mode=? ORDER BY id DESC LIMIT ?", (mode, limit))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self):
        self.conn.close()

"""
data_fetcher.py
===============
Market-data access layer.

Responsibilities:
  * Pull OHLCV candles (1m / 5m / 15m / 1h) for BTC and ETH via ccxt.
  * Pull L2 order book + recent trades for microstructure features.
  * Load historical CSV OHLCV for the backtester.

Everything degrades gracefully: if the network / exchange is unavailable the
methods return None (live caller treats that as "no data -> SKIP"). Nothing
here ever fabricates prices.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger("data_fetcher")

OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class OrderBookSnapshot:
    """Lightweight order-book view used by the microstructure features."""
    bids: List[List[float]]  # [[price, size], ...] best-first
    asks: List[List[float]]
    ts: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_bps(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None or self.mid in (None, 0):
            return None
        return (self.best_ask - self.best_bid) / self.mid * 1e4

    def imbalance(self, depth: int = 20) -> Optional[float]:
        """Order-book imbalance in [-1, 1]; +1 = bids dominate (bullish)."""
        bid_vol = sum(s for _, s in self.bids[:depth])
        ask_vol = sum(s for _, s in self.asks[:depth])
        total = bid_vol + ask_vol
        if total <= 0:
            return None
        return (bid_vol - ask_vol) / total


class DataFetcher:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ex_cfg = cfg["exchange"]
        self.exchange_name = ex_cfg["name"]
        self.depth = ex_cfg.get("orderbook_depth", 50)
        self._exchange = None
        self._init_exchange(ex_cfg)

    def _init_exchange(self, ex_cfg: dict) -> None:
        try:
            import ccxt  # imported lazily so backtest-only users need not install network deps
            klass = getattr(ccxt, self.exchange_name)
            params = {
                "enableRateLimit": True,
                "rateLimit": ex_cfg.get("rate_limit_ms", 250),
                "options": {"defaultType": "spot"},
            }
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if api_key and api_secret:
                params["apiKey"] = api_key
                params["secret"] = api_secret
            self._exchange = klass(params)
            log.info("Initialised exchange %s", self.exchange_name)
        except Exception as exc:  # pragma: no cover - network/env dependent
            log.warning("Could not init exchange (%s); live data disabled: %s",
                        self.exchange_name, exc)
            self._exchange = None

    # ------------------------------------------------------------------ OHLCV
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200
                    ) -> Optional[pd.DataFrame]:
        if self._exchange is None:
            return None
        for attempt in range(3):
            try:
                raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                df = pd.DataFrame(raw, columns=OHLCV_COLS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df
            except Exception as exc:  # pragma: no cover
                wait = 2 ** attempt
                log.warning("fetch_ohlcv %s %s failed (attempt %d): %s; retry in %ds",
                            symbol, timeframe, attempt + 1, exc, wait)
                time.sleep(wait)
        return None

    def fetch_all_timeframes(self, symbol: str) -> Dict[str, Optional[pd.DataFrame]]:
        ex = self.cfg["exchange"]
        return {
            "1m": self.fetch_ohlcv(symbol, ex["ohlcv_timeframe_fast"], limit=300),
            "5m": self.fetch_ohlcv(symbol, ex["ohlcv_timeframe_base"], limit=300),
            "15m": self.fetch_ohlcv(symbol, ex["ohlcv_timeframe_htf1"], limit=200),
            "1h": self.fetch_ohlcv(symbol, ex["ohlcv_timeframe_htf2"], limit=200),
        }

    # -------------------------------------------------------------- Microstructure
    def fetch_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        if self._exchange is None:
            return None
        try:
            ob = self._exchange.fetch_order_book(symbol, limit=self.depth)
            return OrderBookSnapshot(bids=ob.get("bids", []), asks=ob.get("asks", []))
        except Exception as exc:  # pragma: no cover
            log.debug("fetch_orderbook %s failed: %s", symbol, exc)
            return None

    def fetch_trade_flow(self, symbol: str, limit: int = 200) -> Optional[Dict[str, float]]:
        """Aggressive buy/sell volume from recent public trades -> flow imbalance."""
        if self._exchange is None:
            return None
        try:
            trades = self._exchange.fetch_trades(symbol, limit=limit)
            buy = sum(t["amount"] for t in trades if t.get("side") == "buy")
            sell = sum(t["amount"] for t in trades if t.get("side") == "sell")
            total = buy + sell
            imb = (buy - sell) / total if total > 0 else 0.0
            return {"buy_vol": buy, "sell_vol": sell, "flow_imbalance": imb}
        except Exception as exc:  # pragma: no cover
            log.debug("fetch_trade_flow %s failed: %s", symbol, exc)
            return None

    def current_price(self, symbol: str) -> Optional[float]:
        if self._exchange is None:
            return None
        try:
            return float(self._exchange.fetch_ticker(symbol)["last"])
        except Exception as exc:  # pragma: no cover
            log.debug("current_price %s failed: %s", symbol, exc)
            return None

    # -------------------------------------------------------------- Historical
    @staticmethod
    def load_historical_csv(path: str) -> pd.DataFrame:
        """Load a 1m OHLCV CSV for backtesting.

        Accepts either a millisecond/second epoch or ISO timestamp in the first
        column. Required columns (case-insensitive): timestamp/open/high/low/
        close/volume.
        """
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        ts = df.columns[0]
        df = df.rename(columns={ts: "timestamp"})
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            unit = "ms" if df["timestamp"].iloc[0] > 1e12 else "s"
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

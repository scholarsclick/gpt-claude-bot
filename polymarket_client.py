"""
polymarket_client.py
====================
Polymarket market discovery + (optional) order placement.

Read path (no credentials needed):
  * Gamma API  -> discover active BTC/ETH "Up or Down" 5-minute markets,
                  read target price, expiry, YES/NO outcome prices.
  * CLOB API   -> best bid/ask for a token id (tighter prices than Gamma).

Write path (live only, credentials required):
  * py-clob-client -> sign & submit an order. If the package or credentials are
                      missing, live order placement is refused (never silently
                      pretends to trade).

Polymarket's exact schema for these short-dated crypto markets evolves; the
parsing here is defensive and falls back gracefully. In paper/backtest modes
nothing in this file needs network access.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import requests

log = logging.getLogger("polymarket")


@dataclass
class PolyMarket:
    market_id: str
    question: str
    asset: Optional[str]            # "BTC" | "ETH" | None
    target_price: Optional[float]
    end_time: Optional[datetime]
    yes_token_id: Optional[str] = None
    no_token_id: Optional[str] = None
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    raw: dict = field(default_factory=dict)

    @property
    def seconds_remaining(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - datetime.now(timezone.utc)).total_seconds()


def _parse_asset(text: str) -> Optional[str]:
    t = text.lower()
    if "bitcoin" in t or "btc" in t:
        return "BTC"
    if "ethereum" in t or "eth" in t:
        return "ETH"
    return None


def _parse_target_price(text: str) -> Optional[float]:
    """Extract a target/strike like '$67,250' or '67250' from the question."""
    m = re.search(r"\$?\s*([\d]{2,3}(?:,\d{3})+(?:\.\d+)?|\d{3,7}(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_dt(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        if isinstance(s, (int, float)):
            return datetime.fromtimestamp(s, tz=timezone.utc)
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class PolymarketClient:
    def __init__(self, cfg: dict):
        pm = cfg["polymarket"]
        self.gamma = pm["gamma_base_url"].rstrip("/")
        self.clob = pm["clob_base_url"].rstrip("/")
        self.keywords = [k.lower() for k in pm.get("market_keywords", [])]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "polymarket-5m-bot/1.0"})
        self._clob_client = None  # lazy, live only

    # -------------------------------------------------------- discovery (read)
    def _get(self, url, params=None, retries=3):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # pragma: no cover - network dependent
                wait = 2 ** attempt
                log.debug("GET %s failed (%d): %s; retry in %ds", url, attempt + 1, exc, wait)
                time.sleep(wait)
        return None

    # Polymarket's recurring crypto up/down markets use a *deterministic* slug:
    #   {asset}-updown-{interval}-{window_ts}
    # where window_ts = unix_now - (unix_now % interval_seconds). We build the
    # slug for the current (and next) window directly instead of scanning every
    # active market — keyword scanning does not surface these short-dated markets.
    SLUG_PREFIX = {"BTC": "btc", "ETH": "eth"}

    @staticmethod
    def window_ts(interval_seconds: int = 300, ahead: int = 0) -> int:
        now = int(time.time())
        return now - (now % interval_seconds) + ahead * interval_seconds

    def find_active_markets(self, interval_seconds: int = 300) -> List[PolyMarket]:
        """Discover the current & next BTC/ETH 5-minute up/down markets by slug."""
        interval = "5m" if interval_seconds == 300 else f"{interval_seconds // 60}m"
        out: List[PolyMarket] = []
        for asset, prefix in self.SLUG_PREFIX.items():
            for ahead in (0, 1):  # current window + the one about to open
                ts = self.window_ts(interval_seconds, ahead)
                slug = f"{prefix}-updown-{interval}-{ts}"
                pm = self._fetch_by_slug(slug, asset, ts, interval_seconds)
                if pm is not None:
                    out.append(pm)
        if not out:
            out = self._keyword_fallback()
        log.info("Discovered %d active BTC/ETH up-down markets", len(out))
        return out

    def _fetch_by_slug(self, slug: str, asset: str, window_ts: int,
                       interval_seconds: int) -> Optional[PolyMarket]:
        # The event endpoint nests the tradable market(s) under "markets".
        data = self._get(f"{self.gamma}/events", params={"slug": slug})
        events = data if isinstance(data, list) else (data.get("data") if data else None)
        if not events:
            # some deployments expose the slug directly on /markets
            mdata = self._get(f"{self.gamma}/markets", params={"slug": slug})
            mkts = mdata if isinstance(mdata, list) else (mdata.get("data") if mdata else None)
            if not mkts:
                return None
            return self._build_market(mkts[0], asset, slug, window_ts, interval_seconds,
                                      title=mkts[0].get("question"))
        ev = events[0]
        markets = ev.get("markets") or []
        if not markets:
            return None
        return self._build_market(markets[0], asset, slug, window_ts, interval_seconds,
                                   title=ev.get("title") or markets[0].get("question"))

    def _build_market(self, m: dict, asset: str, slug: str, window_ts: int,
                      interval_seconds: int, title: Optional[str]) -> PolyMarket:
        yes_id, no_id = self._token_ids(m)
        yes_price, no_price = self._outcome_prices(m)
        end_time = (_parse_dt(m.get("endDate") or m.get("end_date_iso"))
                    or datetime.fromtimestamp(window_ts + interval_seconds, tz=timezone.utc))
        return PolyMarket(
            market_id=slug,
            question=(title or f"{asset} Up or Down").strip(),
            asset=asset,
            target_price=None,            # resolves vs window-open oracle price; bot uses candle open
            end_time=end_time,
            yes_token_id=yes_id, no_token_id=no_id,
            yes_price=yes_price, no_price=no_price, raw=m,
        )

    @staticmethod
    def _token_ids(m: dict):
        token_ids = m.get("clobTokenIds") or m.get("clob_token_ids")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = None
        if isinstance(token_ids, list) and len(token_ids) >= 2:
            return str(token_ids[0]), str(token_ids[1])
        return None, None

    @staticmethod
    def _outcome_prices(m: dict):
        prices = m.get("outcomePrices") or m.get("outcome_prices")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = None
        if isinstance(prices, list) and len(prices) >= 2:
            try:
                return float(prices[0]), float(prices[1])
            except (TypeError, ValueError):
                return None, None
        return None, None

    def _keyword_fallback(self) -> List[PolyMarket]:
        """Last-resort scan of active markets by keyword (older market styles)."""
        data = self._get(f"{self.gamma}/markets",
                         params={"active": "true", "closed": "false", "limit": 500})
        markets = data if isinstance(data, list) else (data.get("data", []) if data else [])
        out: List[PolyMarket] = []
        for m in markets or []:
            q = (m.get("question") or m.get("title") or "").strip()
            ql = q.lower()
            if not q or not any(k in ql for k in self.keywords):
                continue
            asset = _parse_asset(q)
            if asset is None:
                continue
            pm = self._parse_market(m, q, asset)
            if pm is not None:
                out.append(pm)
        return out

    def _parse_market(self, m: dict, q: str, asset: str) -> Optional[PolyMarket]:
        end_time = _parse_dt(m.get("endDate") or m.get("end_date_iso") or m.get("endDateIso"))
        target = _parse_target_price(q)

        yes_id = no_id = None
        token_ids = m.get("clobTokenIds") or m.get("clob_token_ids")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = None
        if isinstance(token_ids, list) and len(token_ids) >= 2:
            yes_id, no_id = str(token_ids[0]), str(token_ids[1])

        yes_price = no_price = None
        prices = m.get("outcomePrices") or m.get("outcome_prices")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = None
        if isinstance(prices, list) and len(prices) >= 2:
            try:
                yes_price, no_price = float(prices[0]), float(prices[1])
            except (TypeError, ValueError):
                pass

        return PolyMarket(
            market_id=str(m.get("id") or m.get("conditionId") or q),
            question=q, asset=asset, target_price=target, end_time=end_time,
            yes_token_id=yes_id, no_token_id=no_id,
            yes_price=yes_price, no_price=no_price, raw=m,
        )

    def refresh_prices(self, market: PolyMarket) -> PolyMarket:
        """Update YES/NO prices from the CLOB order book (best bid/ask midpoint)."""
        if market.yes_token_id:
            mid = self._clob_midpoint(market.yes_token_id)
            if mid is not None:
                market.yes_price = mid
                if market.no_price is None:
                    market.no_price = round(1 - mid, 4)
        if market.no_token_id and market.no_price is None:
            mid = self._clob_midpoint(market.no_token_id)
            if mid is not None:
                market.no_price = mid
        return market

    def _clob_midpoint(self, token_id: str) -> Optional[float]:
        data = self._get(f"{self.clob}/midpoint", params={"token_id": token_id})
        if data and "mid" in data:
            try:
                return float(data["mid"])
            except (TypeError, ValueError):
                return None
        return None

    # -------------------------------------------------------- order (write)
    def _ensure_clob_client(self):
        if self._clob_client is not None:
            return self._clob_client
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError:
            log.error("py-clob-client not installed; cannot place live orders.")
            return None
        pk = os.getenv("POLYMARKET_PRIVATE_KEY")
        if not pk:
            log.error("POLYMARKET_PRIVATE_KEY missing; cannot place live orders.")
            return None
        try:
            creds = None
            if os.getenv("POLYMARKET_API_KEY"):
                creds = ApiCreds(
                    api_key=os.getenv("POLYMARKET_API_KEY"),
                    api_secret=os.getenv("POLYMARKET_API_SECRET"),
                    api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE"),
                )
            client = ClobClient(
                self.clob, key=pk, chain_id=137,
                creds=creds,
                funder=os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            )
            if creds is None:
                client.set_api_creds(client.create_or_derive_api_creds())
            self._clob_client = client
            return client
        except Exception as exc:  # pragma: no cover
            log.error("Failed to init CLOB client: %s", exc)
            return None

    def place_order(self, token_id: str, price: float, size: float, side: str = "BUY"):
        """Place a limit order on the live CLOB. Returns the API response or None.

        This is only called from live mode in main.py after the LIVE_TRADING
        env guard has passed.
        """
        client = self._ensure_clob_client()
        if client is None:
            log.error("Live order refused: CLOB client unavailable.")
            return None
        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL
            order_args = OrderArgs(
                price=round(float(price), 3),
                size=round(float(size), 2),
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )
            signed = client.create_order(order_args)
            resp = client.post_order(signed)
            log.info("Placed live order token=%s price=%.3f size=%.2f -> %s",
                     token_id, price, size, resp)
            return resp
        except Exception as exc:  # pragma: no cover
            log.error("Live order placement failed: %s", exc)
            return None

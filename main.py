#!/usr/bin/env python3
"""
main.py
=======
Entry point + orchestration for the BTC/ETH 5-minute Polymarket Up/Down bot.

Modes (CLI):
    python main.py backtest                 # train + evaluate on historical data
    python main.py paper                    # paper trade (default, no real orders)
    python main.py live                      # live trade (requires LIVE_TRADING=true)
    python main.py fetch-data --days 30      # download 1m OHLCV to data/ for backtest
    python main.py train                      # (re)train model from data/ and save artifacts

Safety:
    * Default mode is paper. Live mode is refused unless config mode == "live"
      AND env LIVE_TRADING == "true".
    * Every trade goes through RiskManager.can_trade() before submission.
    * The bot is built to SKIP most markets; that is the intended behaviour.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from data_fetcher import DataFetcher
from features import MarketContext, build_features
from model import DirectionModel
from polymarket_client import PolymarketClient
from risk_manager import RiskManager
from strategy import SignalEngine


def load_config(path: Optional[str] = None) -> dict:
    path = path or os.getenv("BOT_CONFIG", "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def setup_logging(cfg: dict):
    lg = cfg.get("logging", {})
    os.makedirs(os.path.dirname(lg.get("file", "artifacts/bot.log")) or ".", exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(lg.get("file", "artifacts/bot.log")))
    except Exception:
        pass
    logging.basicConfig(
        level=getattr(logging, lg.get("level", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


log = logging.getLogger("main")


def exchange_symbol_for(cfg: dict, asset: str) -> Optional[str]:
    for a in cfg["assets"]:
        if a["symbol"] == asset:
            return a["exchange_symbol"]
    return None


def load_model(cfg: dict) -> DirectionModel:
    model = DirectionModel(cfg)
    if cfg["model"].get("enabled", True):
        model.load(cfg["model"]["model_path"], cfg["model"]["calibrator_path"])
    return model


# --------------------------------------------------------------------------- #
# fetch-data / train / backtest
# --------------------------------------------------------------------------- #
def cmd_fetch_data(cfg: dict, days: int):
    import pandas as pd
    fetcher = DataFetcher(cfg)
    if fetcher._exchange is None:
        log.error("No exchange connection; cannot fetch data.")
        return
    data_dir = cfg["backtest"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    limit_total = days * 24 * 60
    for a in cfg["assets"]:
        sym, asset = a["exchange_symbol"], a["symbol"]
        log.info("Downloading %d days of 1m %s ...", days, sym)
        all_rows = []
        since = fetcher._exchange.milliseconds() - limit_total * 60_000
        while True:
            try:
                batch = fetcher._exchange.fetch_ohlcv(sym, "1m", since=since, limit=1000)
            except Exception as exc:
                log.warning("fetch batch failed: %s", exc)
                time.sleep(2)
                continue
            if not batch:
                break
            all_rows += batch
            since = batch[-1][0] + 60_000
            if len(all_rows) >= limit_total or batch[-1][0] >= fetcher._exchange.milliseconds():
                break
            time.sleep(fetcher._exchange.rateLimit / 1000)
        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp")
        out = os.path.join(data_dir, f"{asset}_1m.csv")
        df.to_csv(out, index=False)
        log.info("Saved %d rows -> %s", len(df), out)


def _load_backtest_data(cfg: dict) -> Dict[str, "pd.DataFrame"]:
    import pandas as pd  # noqa
    data_dir = cfg["backtest"]["data_dir"]
    data = {}
    for a in cfg["assets"]:
        path = os.path.join(data_dir, f"{a['symbol']}_1m.csv")
        if os.path.exists(path):
            data[a["symbol"]] = DataFetcher.load_historical_csv(path)
            log.info("Loaded %d rows for %s", len(data[a["symbol"]]), a["symbol"])
        else:
            log.warning("Missing %s — run `python main.py fetch-data` first.", path)
    return data


def cmd_backtest(cfg: dict):
    from backtest import Backtester
    data = _load_backtest_data(cfg)
    if not data:
        log.error("No historical data found. Run: python main.py fetch-data --days 30")
        return
    model = DirectionModel(cfg)
    engine = SignalEngine(cfg, model=None)
    bt = Backtester(cfg)
    metrics = bt.run(data, engine, model)
    # persist trained model so paper/live can reuse it
    if model.art.kind != "rules_only":
        model.save(cfg["model"]["model_path"], cfg["model"]["calibrator_path"])
    print("\n===== BACKTEST RESULTS =====")
    print(json.dumps(metrics, indent=2, default=str))
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/backtest_results.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    log.info("Backtest results written to artifacts/backtest_results.json")
    _verdict(cfg, metrics)


def _verdict(cfg: dict, metrics: dict):
    if metrics.get("trades", 0) == 0:
        print("\nVerdict: strategy took no trades (skipped everything). "
              "This is acceptable — it means no positive-EV edge was found in this data.")
        return
    g = cfg["gates"]
    ok = (metrics["win_rate"] >= 0.5 and metrics["total_pnl"] > 0 and
          metrics["avg_ev"] >= g["min_ev_after_fees"])
    print(f"\nVerdict: {'PASS' if ok else 'FAIL'} — "
          f"win_rate={metrics['win_rate']:.1%} pnl={metrics['total_pnl']:.2f} "
          f"avg_ev={metrics['avg_ev']:.1%} sharpe={metrics['sharpe_annualized']}")
    if not ok:
        print("Do NOT enable live trading until the backtest and paper test are profitable.")


def cmd_train(cfg: dict):
    from backtest import Backtester
    data = _load_backtest_data(cfg)
    if not data:
        log.error("No data to train on.")
        return
    model = DirectionModel(cfg)
    bt = Backtester(cfg)
    samples = []
    for asset, df1 in data.items():
        samples.extend(bt.build_samples(asset, df1))
    samples.sort(key=lambda s: s.win_open_ts)
    bt.train_model(samples, model)
    if model.art.kind != "rules_only":
        model.save(cfg["model"]["model_path"], cfg["model"]["calibrator_path"])
        log.info("Saved trained model.")
    else:
        log.warning("Not enough data for ML model; staying rules_only.")


# --------------------------------------------------------------------------- #
# Live / paper trading loop
# --------------------------------------------------------------------------- #
class TradingLoop:
    def __init__(self, cfg: dict, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.fetcher = DataFetcher(cfg)
        self.poly = PolymarketClient(cfg)
        self.model = load_model(cfg)
        self.engine = SignalEngine(cfg, model=self.model)
        self.risk = RiskManager(cfg)
        self.open_trades: Dict[str, dict] = {}   # market_id -> {trade_id, market, side}
        self.poll = cfg["polymarket"]["poll_interval_seconds"]

    def _evaluate_market(self, market):
        sym = exchange_symbol_for(self.cfg, market.asset)
        if sym is None:
            return None
        tf = self.fetcher.fetch_all_timeframes(sym)
        if tf["1m"] is None or tf["5m"] is None:
            return None
        ob = self.fetcher.fetch_orderbook(sym)
        flow = self.fetcher.fetch_trade_flow(sym)
        price = self.fetcher.current_price(sym) or float(tf["1m"]["close"].iloc[-1])
        # 5m up/down markets resolve vs the price at the window open. Use the
        # open of the current (forming) 5m candle as the target when the market
        # carries no explicit strike (the deterministic-slug markets don't).
        target = market.target_price if market.target_price else float(tf["5m"]["open"].iloc[-1])
        ctx = MarketContext(
            target_price=target,
            seconds_remaining=market.seconds_remaining,
            yes_price=market.yes_price, no_price=market.no_price,
            candle_open=target,
        )
        feats = build_features(tf, ctx, orderbook=ob, trade_flow=flow)
        sig = self.engine.evaluate(market.asset, feats, ctx)
        return sig, price, target

    def _resolve_expired(self):
        for mid in list(self.open_trades.keys()):
            rec = self.open_trades[mid]
            market = rec["market"]
            if market.seconds_remaining > 0:
                continue
            sym = exchange_symbol_for(self.cfg, market.asset)
            final = self.fetcher.current_price(sym)
            if final is None:
                continue
            self.risk.resolve_trade(rec["trade_id"], final)
            del self.open_trades[mid]

    def step(self):
        self._resolve_expired()
        markets = self.poly.find_active_markets()
        for m in markets:
            if m.asset not in [a["symbol"] for a in self.cfg["assets"]]:
                continue
            m = self.poly.refresh_prices(m)
            if m.market_id in self.open_trades:
                continue
            result = self._evaluate_market(m)
            if result is None:
                continue
            sig, price, target = result
            tag = (f"{m.asset} {m.seconds_remaining:.0f}s left | {sig.side} "
                   f"p_up={sig.probability_up:.2f} EV={sig.expected_value:+.1%}")
            if sig.side == "SKIP":
                log.info("SKIP  %s | %s", tag, sig.skip_reason)
                continue
            rs = self.risk.can_trade(self.mode, m.asset)
            if not rs.allowed:
                log.info("BLOCK %s | risk: %s", tag, rs.reason)
                continue
            log.info("TRADE %s | %s", tag, "; ".join(sig.reasons))
            tid = self.risk.open_trade(self.mode, m.asset, m.market_id, sig, target)
            if self.mode == "live":
                token = m.yes_token_id if sig.side == "UP" else m.no_token_id
                size = self.risk.stake / (sig.entry_price or 0.5)
                self.poly.place_order(token, sig.entry_price, size, side="BUY")
            self.open_trades[m.market_id] = {"trade_id": tid, "market": m, "side": sig.side}

    def run(self):
        log.info("Starting %s trading loop (poll=%ss). Ctrl-C to stop.", self.mode, self.poll)
        try:
            while True:
                try:
                    self.step()
                except Exception as exc:
                    log.exception("loop error: %s", exc)
                time.sleep(self.poll)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            self.risk.close()


def cmd_trade(cfg: dict, mode: str):
    if mode == "live":
        if cfg.get("mode") != "live":
            log.error("Refusing live: set `mode: live` in config.yaml.")
            return
        if os.getenv("LIVE_TRADING", "false").lower() != "true":
            log.error("Refusing live: set LIVE_TRADING=true in .env.")
            return
        log.warning("LIVE TRADING ENABLED — real orders will be placed.")
    TradingLoop(cfg, mode).run()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="BTC/ETH 5m Polymarket Up/Down bot")
    parser.add_argument("command", nargs="?", default=None,
                        choices=["backtest", "paper", "live", "fetch-data", "train"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    command = args.command or cfg.get("mode", "paper")
    if command == "live" and cfg.get("mode") != "live":
        # honour the safer of the two; explicit CLI 'live' still needs config+env
        pass

    if command == "fetch-data":
        cmd_fetch_data(cfg, args.days)
    elif command == "train":
        cmd_train(cfg)
    elif command == "backtest":
        cmd_backtest(cfg)
    elif command in ("paper", "live"):
        cmd_trade(cfg, command)
    else:
        log.error("Unknown command: %s", command)


if __name__ == "__main__":
    main()

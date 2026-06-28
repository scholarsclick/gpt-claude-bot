"""
dashboard.py
============
Streamlit dashboard for the BTC/ETH 5-minute Polymarket bot.

Run:
    streamlit run dashboard.py

Shows:
  * Live BTC/ETH 5m markets with price vs target, model probability, Polymarket
    odds, EV, and the UP/DOWN/SKIP signal + reasons.
  * Recent trades, PnL, win rate, drawdown (from the SQLite ledger).
  * Latest backtest results (metrics, calibration curve, confusion matrix).

The dashboard is read-only on the trading side: it computes live signals for
display but never places orders. Run `python main.py paper|live` for execution.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pandas as pd
import streamlit as st

from main import load_config, load_model, exchange_symbol_for
from data_fetcher import DataFetcher
from features import MarketContext, build_features
from polymarket_client import PolymarketClient
from strategy import SignalEngine

st.set_page_config(page_title="BTC/ETH 5m Polymarket Bot", layout="wide")


@st.cache_resource
def _bootstrap():
    cfg = load_config()
    fetcher = DataFetcher(cfg)
    poly = PolymarketClient(cfg)
    model = load_model(cfg)
    engine = SignalEngine(cfg, model=model)
    return cfg, fetcher, poly, engine


cfg, fetcher, poly, engine = _bootstrap()
mode = cfg.get("mode", "paper")

st.title("📈 BTC/ETH 5-minute Polymarket Up/Down Bot")
st.caption(f"Mode: **{mode}** · The bot is designed to SKIP most markets and "
           "only trade on positive-EV mispricing.")

tab_live, tab_trades, tab_backtest = st.tabs(["🔴 Live Signals", "📒 Trades & PnL", "🧪 Backtest"])

# --------------------------------------------------------------------------- #
with tab_live:
    import time as _time
    c_a, c_b = st.columns([1, 3])
    if c_a.button("🔄 Refresh now"):
        st.cache_data.clear()
    auto = c_b.checkbox("Auto-refresh every 10s", value=False)

    @st.cache_data(ttl=8)
    def live_signals():
        """Always compute a per-asset Binance + model signal for the current 5m
        window. Attach Polymarket odds/EV when a matching market is found."""
        rows, diag = [], []
        # discover Polymarket markets (best-effort; signals still render without them)
        markets_by_asset = {}
        try:
            for m in poly.find_active_markets():
                markets_by_asset.setdefault(m.asset, []).append(m)
            diag.append(f"Polymarket markets found: "
                        f"{ {k: len(v) for k, v in markets_by_asset.items()} }")
        except Exception as exc:
            diag.append(f"Polymarket discovery error: {exc}")

        now = _time.time()
        sec_left = 300 - (now % 300)   # time remaining in the current 5m window

        for a in cfg["assets"]:
            asset, sym = a["symbol"], a["exchange_symbol"]
            try:
                tf = fetcher.fetch_all_timeframes(sym)
                if tf["1m"] is None or tf["5m"] is None:
                    rows.append({"Asset": asset, "Signal": "NO DATA",
                                 "Reason": f"Binance unreachable for {sym} "
                                           "(geo-block/VPN/network)"})
                    continue
                price = fetcher.current_price(sym) or float(tf["1m"]["close"].iloc[-1])
                # current 5m window open = open of the last (forming) 5m candle
                target = float(tf["5m"]["open"].iloc[-1])
                ob = fetcher.fetch_orderbook(sym)
                flow = fetcher.fetch_trade_flow(sym)

                # attach a Polymarket market for this asset, if any
                mkt = None
                for cand in markets_by_asset.get(asset, []):
                    if cand.seconds_remaining > 0:
                        mkt = poly.refresh_prices(cand)
                        break
                yes = mkt.yes_price if mkt else None
                no = mkt.no_price if mkt else None
                srem = mkt.seconds_remaining if mkt else sec_left

                ctx = MarketContext(target_price=target, seconds_remaining=srem,
                                    yes_price=yes, no_price=no, candle_open=target)
                feats = build_features(tf, ctx, orderbook=ob, trade_flow=flow)
                sig = engine.evaluate(asset, feats, ctx)
                lean = "UP" if sig.probability_up >= 0.5 else "DOWN"
                rows.append({
                    "Asset": asset,
                    "Price": round(price, 2),
                    "Window open": round(target, 2),
                    "Dist(bps)": round(feats.get("dist_from_target_bps", 0), 1),
                    "Sec left": round(srem),
                    "Model lean": lean,
                    "P(up)": round(sig.probability_up, 3),
                    "Conf": round(sig.confidence, 2),
                    "PM odds(up)": round(yes, 3) if yes is not None else "—",
                    "EV": round(sig.expected_value, 3) if yes is not None else "—",
                    "Signal": sig.side,
                    "Reason": sig.skip_reason or "; ".join(sig.reasons),
                })
            except Exception as exc:
                rows.append({"Asset": asset, "Signal": "ERR", "Reason": str(exc)})
        return rows, diag

    rows, diag = live_signals()
    for d in diag:
        st.caption(d)

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No data. Check network access to Binance and Polymarket.")
    else:
        def color_signal(v):
            return {"UP": "background-color:#103d10",
                    "DOWN": "background-color:#3d1010"}.get(v, "")
        sty = df.style.applymap(color_signal, subset=["Signal"]) if "Signal" in df else df.style
        st.dataframe(sty, use_container_width=True)
        if "Signal" in df:
            traded = df[df["Signal"].isin(["UP", "DOWN"])]
            st.metric("Actionable signals (UP/DOWN)", f"{len(traded)} / {len(df)}")
        st.caption("Model lean / P(up) always reflect live Binance price action. "
                   "Signal stays SKIP until Polymarket odds exist AND the EV edge "
                   "clears fees + slippage — by design, most windows are SKIP.")
    if auto:
        _time.sleep(10)
        st.rerun()

# --------------------------------------------------------------------------- #
with tab_trades:
    db_path = cfg["database"]["path"]
    if not os.path.exists(db_path):
        st.info("No trade ledger yet. Run `python main.py paper` to start logging.")
    else:
        conn = sqlite3.connect(db_path)
        trades = pd.read_sql_query(
            "SELECT * FROM trades WHERE mode=? ORDER BY id DESC LIMIT 200", conn, params=(mode,))
        resolved = trades[trades["status"] == "resolved"]
        c1, c2, c3, c4 = st.columns(4)
        total_pnl = resolved["pnl"].sum() if not resolved.empty else 0.0
        win_rate = (resolved["outcome"] == "win").mean() if not resolved.empty else 0.0
        c1.metric("Total PnL", f"{total_pnl:,.2f}")
        c2.metric("Win rate", f"{win_rate:.1%}")
        c3.metric("Resolved trades", len(resolved))
        c4.metric("Open trades", int((trades["status"] == "open").sum()))

        if not resolved.empty:
            start = cfg["risk"]["starting_bankroll"]
            curve = resolved.sort_values("id")
            curve["equity"] = start + curve["pnl"].cumsum()
            curve["peak"] = curve["equity"].cummax()
            curve["drawdown"] = curve["peak"] - curve["equity"]
            st.line_chart(curve.set_index("id")[["equity"]])
            st.caption(f"Max drawdown: {curve['drawdown'].max():.2f}")
        st.subheader("Recent trades")
        cols = ["ts", "asset", "side", "stake", "entry_price", "expected_value",
                "status", "outcome", "pnl", "reasons"]
        st.dataframe(trades[[c for c in cols if c in trades.columns]],
                     use_container_width=True, height=360)
        conn.close()

# --------------------------------------------------------------------------- #
with tab_backtest:
    path = "artifacts/backtest_results.json"
    if not os.path.exists(path):
        st.info("No backtest results yet. Run `python main.py backtest`.")
    else:
        with open(path) as fh:
            m = json.load(fh)
        if m.get("trades", 0) == 0:
            st.warning(m.get("note", "No trades taken in backtest."))
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Trades", m["trades"])
            c2.metric("Win rate", f"{m['win_rate']:.1%}")
            c3.metric("PnL", f"{m['total_pnl']:,.2f}")
            c4.metric("Avg EV", f"{m['avg_ev']:.1%}")
            c5.metric("Sharpe", m["sharpe_annualized"])
            st.metric("Max drawdown", f"{m['max_drawdown']:,.2f}")

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Calibration curve")
                if m.get("calibration"):
                    cal = pd.DataFrame(m["calibration"]).dropna(subset=["pred", "actual"])
                    if not cal.empty:
                        st.line_chart(cal.set_index("pred")[["actual"]])
                        st.caption("Closer to diagonal = better calibrated.")
            with colB:
                st.subheader("Confusion matrix")
                st.json(m.get("confusion_matrix", {}))

            st.subheader("Breakdowns")
            b1, b2, b3 = st.columns(3)
            b1.write("By time remaining"); b1.json(m.get("by_time_remaining", {}))
            b2.write("By volatility regime"); b2.json(m.get("by_vol_regime", {}))
            b3.write("By asset"); b3.json(m.get("by_asset", {}))

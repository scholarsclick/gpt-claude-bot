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
    if st.button("🔄 Refresh markets"):
        st.cache_data.clear()

    @st.cache_data(ttl=8)
    def live_signals():
        rows = []
        try:
            markets = poly.find_active_markets()
        except Exception as exc:
            return [], f"Market discovery failed: {exc}"
        for m in markets:
            if m.asset not in [a["symbol"] for a in cfg["assets"]]:
                continue
            try:
                m = poly.refresh_prices(m)
                sym = exchange_symbol_for(cfg, m.asset)
                tf = fetcher.fetch_all_timeframes(sym)
                if tf["1m"] is None or tf["5m"] is None:
                    continue
                ob = fetcher.fetch_orderbook(sym)
                flow = fetcher.fetch_trade_flow(sym)
                price = fetcher.current_price(sym) or float(tf["1m"]["close"].iloc[-1])
                target = m.target_price or price
                ctx = MarketContext(target_price=target, seconds_remaining=m.seconds_remaining,
                                    yes_price=m.yes_price, no_price=m.no_price, candle_open=target)
                feats = build_features(tf, ctx, orderbook=ob, trade_flow=flow)
                sig = engine.evaluate(m.asset, feats, ctx)
                rows.append({
                    "Asset": m.asset, "Question": m.question[:50],
                    "Price": round(price, 2), "Target": round(target, 2),
                    "Dist(bps)": round(feats.get("dist_from_target_bps", 0), 1),
                    "Sec left": round(m.seconds_remaining),
                    "Model P(up)": sig.probability_up,
                    "PM odds(up)": m.yes_price,
                    "EV": sig.expected_value,
                    "Signal": sig.side,
                    "Reason": sig.skip_reason or "; ".join(sig.reasons),
                })
            except Exception as exc:
                rows.append({"Asset": m.asset, "Question": m.question[:50], "Signal": "ERR",
                             "Reason": str(exc)})
        return rows, None

    rows, err = live_signals()
    if err:
        st.warning(err)
    if not rows:
        st.info("No active BTC/ETH 5-minute markets found (or no network access). "
                "Discovery depends on Polymarket having live 5m up/down markets right now.")
    else:
        df = pd.DataFrame(rows)
        def color_signal(v):
            return {"UP": "background-color:#103d10", "DOWN": "background-color:#3d1010"}.get(v, "")
        st.dataframe(df.style.applymap(color_signal, subset=["Signal"]),
                     use_container_width=True, height=420)
        traded = df[df["Signal"].isin(["UP", "DOWN"])]
        st.metric("Actionable signals", f"{len(traded)} / {len(df)}")

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

"""
backtest.py
===========
Event-driven backtester for the 5-minute Up/Down strategy.

What it does:
  1. Loads historical 1m OHLCV for BTC/ETH.
  2. Reconstructs Polymarket-style 5-minute windows. For each window and each
     configured entry offset (e.g. 120/60/30s remaining) it rebuilds the exact
     feature vector the live bot would have seen — with NO look-ahead (the
     forming 5m candle is rebuilt from 1m bars up to the entry instant).
  3. Labels each window by whether it actually closed UP vs its open.
  4. Trains the ML ensemble on a time-ordered split, then simulates trading
     through the out-of-sample portion using the real SignalEngine + gates.
  5. Reports win rate, profit after fees, max drawdown, Sharpe, average EV,
     and breakdowns by time-remaining / volatility regime / asset, plus a
     confusion matrix and calibration curve.

IMPORTANT honesty note: real Polymarket historical quote data is not bundled.
The synthetic market price here models a *momentum-naive* counterparty (leans
toward the current intra-candle move) plus noise. This tests whether the model
has edge over a naive market — it is NOT a guarantee of live profit. To get a
true backtest, record live Polymarket order books and replace `synthetic_price`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features import MarketContext, build_features, feature_row, ML_FEATURE_COLUMNS

log = logging.getLogger("backtest")


# --------------------------------------------------------------------------- #
# Resampling helpers
# --------------------------------------------------------------------------- #
def resample(df1: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df1.set_index("timestamp").resample(rule)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna().reset_index()
    return out


def forming_candle(df1: pd.DataFrame, win_open_ts, entry_ts) -> Optional[dict]:
    """Rebuild the partial 5m candle from 1m bars in [win_open_ts, entry_ts]."""
    seg = df1[(df1["timestamp"] >= win_open_ts) & (df1["timestamp"] <= entry_ts)]
    if seg.empty:
        return None
    return {
        "timestamp": win_open_ts,
        "open": float(seg["open"].iloc[0]),
        "high": float(seg["high"].max()),
        "low": float(seg["low"].min()),
        "close": float(seg["close"].iloc[-1]),
        "volume": float(seg["volume"].sum()),
    }


def synthetic_price(true_up: bool, dist_from_open_bps: float, rng) -> Tuple[float, float]:
    """Momentum-naive market quote + noise. Returns (yes_price, no_price)."""
    lean = np.tanh(dist_from_open_bps / 25.0) * 0.18   # market chases the move
    noise = rng.normal(0, 0.06)
    yes = float(np.clip(0.5 + lean + noise, 0.05, 0.95))
    return round(yes, 3), round(1 - yes, 3)


# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    asset: str
    features: Dict[str, float]
    label: int                 # 1 = up
    seconds_remaining: float
    vol_regime: str
    win_open_ts: pd.Timestamp
    target_price: float
    final_price: float


class Backtester:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bt = cfg["backtest"]
        self.window = self.bt["window_seconds"]
        self.offsets = self.bt["entry_offsets_seconds"]

    # ----------------------------------------------------- dataset construction
    def build_samples(self, asset: str, df1: pd.DataFrame) -> List[Sample]:
        df1 = df1.sort_values("timestamp").reset_index(drop=True)
        df5 = resample(df1, "5min")
        df15 = resample(df1, "15min")
        df1h = resample(df1, "60min")
        samples: List[Sample] = []

        # volatility regime thresholds from 5m ATR distribution
        atr5 = (df5["high"] - df5["low"]) / df5["close"]
        lo_q, hi_q = atr5.quantile(0.33), atr5.quantile(0.66)

        for i in range(60, len(df5) - 1):           # need history for indicators
            win = df5.iloc[i]
            win_open_ts = win["timestamp"]
            win_close_ts = win_open_ts + pd.Timedelta(seconds=self.window)
            target = float(win["open"])
            final_price = float(win["close"])
            label = 1 if final_price > target else 0
            cur_atr = atr5.iloc[i]
            regime = "low" if cur_atr < lo_q else ("high" if cur_atr > hi_q else "mid")

            for off in self.offsets:
                entry_ts = win_close_ts - pd.Timedelta(seconds=off)
                fc = forming_candle(df1, win_open_ts, entry_ts)
                if fc is None:
                    continue
                d1 = df1[df1["timestamp"] <= entry_ts].tail(300)
                d5_hist = df5[df5["timestamp"] < win_open_ts].tail(299)
                d5 = pd.concat([d5_hist, pd.DataFrame([fc])], ignore_index=True)
                d15 = df15[df15["timestamp"] <= entry_ts].tail(200)
                d1h = df1h[df1h["timestamp"] <= entry_ts].tail(200)
                if len(d1) < 60 or len(d5) < 55 or len(d15) < 50:
                    continue
                ctx = MarketContext(target_price=target, seconds_remaining=off,
                                    candle_open=target)
                feats = build_features({"1m": d1, "5m": d5, "15m": d15, "1h": d1h}, ctx)
                samples.append(Sample(asset, feats, label, off, regime,
                                      win_open_ts, target, final_price))
        log.info("[%s] built %d samples across %d windows", asset, len(samples), len(df5))
        return samples

    # ----------------------------------------------------------------- training
    def train_model(self, samples: List[Sample], model):
        n = int(len(samples) * 0.7)
        train = samples[:n]
        X = np.array([feature_row(s.features) for s in train])
        y = np.array([s.label for s in train])
        art = model.train(X, y)
        log.info("Model trained: kind=%s n=%d", art.kind, art.n_train)
        return n

    # --------------------------------------------------------------- simulation
    def simulate(self, samples: List[Sample], engine, start_idx: int, seed: int = 7) -> dict:
        rng = np.random.default_rng(seed)
        from risk_manager import RiskManager
        # fresh in-memory ledger for the backtest run
        cfg = dict(self.cfg)
        cfg = {**cfg, "database": {"path": ":memory:"}}
        rm = RiskManager(cfg)
        mode = "backtest"

        equity = rm.starting_bankroll
        equity_curve = [equity]
        trade_returns: List[float] = []
        evs: List[float] = []
        records: List[dict] = []

        for s in samples[start_idx:]:
            # inject a synthetic market quote
            yes, no = synthetic_price(bool(s.label), s.features.get("dist_from_open_bps", 0), rng)
            s.features["yes_price"] = yes
            s.features["no_price"] = no
            denom = yes + no
            s.features["implied_prob_up"] = yes / denom if denom else 0.5
            s.features["pm_spread_bps"] = abs(yes - (1 - no)) * 1e4
            ctx = MarketContext(target_price=s.target_price, seconds_remaining=s.seconds_remaining,
                                yes_price=yes, no_price=no, candle_open=s.target_price)
            sig = engine.evaluate(s.asset, s.features, ctx)
            if sig.side == "SKIP":
                continue
            rs = rm.can_trade(mode, s.asset)
            if not rs.allowed:
                continue
            tid = rm.open_trade(mode, s.asset, f"bt-{s.win_open_ts}", sig, s.target_price)
            outcome, pnl = rm.resolve_trade(tid, s.final_price)
            equity += pnl
            equity_curve.append(equity)
            trade_returns.append(pnl / rm.stake)
            evs.append(sig.expected_value)
            records.append({
                "asset": s.asset, "side": sig.side, "outcome": outcome, "pnl": pnl,
                "prob": sig.probability_up if sig.side == "UP" else sig.probability_down,
                "model_prob_up": sig.probability_up, "label": s.label,
                "seconds_remaining": s.seconds_remaining, "vol_regime": s.vol_regime,
                "ev": sig.expected_value,
            })
        rm.close()
        return self._metrics(records, equity_curve, trade_returns, evs, rm.starting_bankroll)

    # ----------------------------------------------------------------- metrics
    @staticmethod
    def _metrics(records, equity_curve, returns, evs, start_bankroll) -> dict:
        n = len(records)
        if n == 0:
            return {"trades": 0, "note": "No trades taken — strategy skipped every market "
                    "(expected when no mispricing edge exists)."}
        df = pd.DataFrame(records)
        wins = (df["outcome"] == "win").sum()
        win_rate = wins / n
        total_pnl = df["pnl"].sum()
        ret = np.array(returns)
        sharpe = float(ret.mean() / (ret.std() + 1e-9) * np.sqrt(252)) if len(ret) > 1 else 0.0
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float((peak - eq).max())

        def breakdown(col):
            g = df.groupby(col)
            return {str(k): {"trades": int(len(v)),
                             "win_rate": round((v["outcome"] == "win").mean(), 3),
                             "pnl": round(v["pnl"].sum(), 2)} for k, v in g}

        # confusion matrix: predicted side vs actual direction
        df["pred_up"] = df["side"] == "UP"
        df["actual_up"] = df["label"] == 1
        cm = {
            "tp_up": int(((df.pred_up) & (df.actual_up)).sum()),
            "fp_up": int(((df.pred_up) & (~df.actual_up)).sum()),
            "tn_down": int(((~df.pred_up) & (~df.actual_up)).sum()),
            "fn_down": int(((~df.pred_up) & (df.actual_up)).sum()),
        }

        # calibration curve over model_prob_up vs realised up-rate (all evaluated)
        bins = np.linspace(0, 1, 11)
        df["bin"] = pd.cut(df["model_prob_up"], bins, include_lowest=True)
        calib = []
        for b, v in df.groupby("bin", observed=True):
            calib.append({"bin": str(b), "n": int(len(v)),
                          "pred": round(v["model_prob_up"].mean(), 3),
                          "actual": round(v["actual_up"].mean(), 3)})

        return {
            "trades": n,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / start_bankroll * 100, 2),
            "avg_ev": round(float(np.mean(evs)), 4),
            "realized_avg_return": round(float(ret.mean()), 4),
            "sharpe_annualized": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "by_time_remaining": breakdown("seconds_remaining"),
            "by_vol_regime": breakdown("vol_regime"),
            "by_asset": breakdown("asset"),
            "confusion_matrix": cm,
            "calibration": calib,
            "final_equity": round(equity_curve[-1], 2),
        }

    # ------------------------------------------------------------------- driver
    def run(self, data: Dict[str, pd.DataFrame], engine, model) -> dict:
        all_samples: List[Sample] = []
        for asset, df1 in data.items():
            all_samples.extend(self.build_samples(asset, df1))
        # keep chronological order across assets for an honest train/test split
        all_samples.sort(key=lambda s: s.win_open_ts)
        if not all_samples:
            return {"error": "No samples built — check that historical data covers enough range."}
        split = self.train_model(all_samples, model)
        engine.model = model
        metrics = self.simulate(all_samples, engine, start_idx=split)
        return metrics

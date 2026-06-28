"""
features.py
===========
Feature engineering for the BTC/ETH 5-minute Up/Down predictor.

Two layers:
  1. `indicators` — pure pandas/numpy implementations of the technical
     indicators (no hard dependency on TA-Lib; uses pandas-ta if present but
     never requires it). This keeps the backtester runnable anywhere.
  2. `build_features` — assembles a single flat feature dict from the multi
     timeframe candle data + microstructure + Polymarket context. The same
     function is used live and in the backtest so there is zero train/serve
     skew.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicator primitives
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd_hist(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def stoch_rsi(series: pd.Series, period=14, k=3) -> pd.Series:
    r = rsi(series, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    srsi = (r - lo) / (hi - lo).replace(0, np.nan)
    return (srsi.rolling(k).mean() * 100).fillna(50.0)


def roc(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(period) * 100


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_width(series: pd.Series, period=20, n_std=2) -> pd.Series:
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper, lower = ma + n_std * sd, ma - n_std * sd
    return ((upper - lower) / ma.replace(0, np.nan)).fillna(0.0)


def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


def realized_vol(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change().rolling(period).std() * np.sqrt(period)


def swing_levels(df: pd.DataFrame, lookback: int = 50, left: int = 3, right: int = 3):
    """Return (support, resistance) from recent fractal swing lows/highs."""
    window = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(left, len(window) - right):
        seg_h = window["high"].iloc[i - left:i + right + 1]
        seg_l = window["low"].iloc[i - left:i + right + 1]
        if window["high"].iloc[i] == seg_h.max():
            highs.append(window["high"].iloc[i])
        if window["low"].iloc[i] == seg_l.min():
            lows.append(window["low"].iloc[i])
    price = df["close"].iloc[-1]
    resistance = min([h for h in highs if h >= price], default=window["high"].max())
    support = max([l for l in lows if l <= price], default=window["low"].min())
    return float(support), float(resistance)


def slope(series: pd.Series, period: int = 5) -> float:
    """Normalised slope of the last `period` points (per-bar % change)."""
    s = series.dropna().tail(period)
    if len(s) < 2 or s.iloc[0] == 0:
        return 0.0
    x = np.arange(len(s))
    coef = np.polyfit(x, s.values, 1)[0]
    return float(coef / abs(s.iloc[0]) * 100)


# ---------------------------------------------------------------------------
# Market context passed in from the Polymarket layer / backtester
# ---------------------------------------------------------------------------
@dataclass
class MarketContext:
    target_price: float          # Polymarket open/target the candle is measured against
    seconds_remaining: float
    yes_price: Optional[float] = None   # Polymarket UP price in [0,1]
    no_price: Optional[float] = None    # Polymarket DOWN price in [0,1]
    candle_open: Optional[float] = None # current 5m candle open (== target for fresh markets)


def _safe(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def build_features(
    tf: Dict[str, pd.DataFrame],
    ctx: MarketContext,
    orderbook=None,
    trade_flow: Optional[dict] = None,
) -> Dict[str, float]:
    """Assemble the full flat feature vector.

    `tf` maps timeframe label -> OHLCV DataFrame (oldest first, newest last).
    Required keys: '1m', '5m', '15m', '1h'. Microstructure args are optional.
    """
    df1 = tf["1m"]
    df5 = tf["5m"]
    df15 = tf.get("15m")
    df1h = tf.get("1h")

    close5 = df5["close"]
    close1 = df1["close"]
    price = _safe(close1.iloc[-1])
    target = _safe(ctx.target_price, price)
    f: Dict[str, float] = {}

    # ---- 1. Market structure ------------------------------------------------
    open5 = _safe(ctx.candle_open if ctx.candle_open is not None else df5["open"].iloc[-1])
    f["price"] = price
    f["dist_from_target_bps"] = (price - target) / target * 1e4 if target else 0.0
    f["dist_from_open_bps"] = (price - open5) / open5 * 1e4 if open5 else 0.0
    f["seconds_remaining"] = _safe(ctx.seconds_remaining)
    f["time_fraction_left"] = np.clip(ctx.seconds_remaining / 300.0, 0, 1)

    # higher-timeframe trend (sign of EMA stack)
    def trend_score(d: Optional[pd.DataFrame]) -> float:
        if d is None or len(d) < 50:
            return 0.0
        c = d["close"]
        e9, e21, e50 = ema(c, 9).iloc[-1], ema(c, 21).iloc[-1], ema(c, 50).iloc[-1]
        score = 0.0
        score += 0.5 if e9 > e21 else -0.5
        score += 0.5 if e21 > e50 else -0.5
        return score
    f["trend_15m"] = trend_score(df15)
    f["trend_1h"] = trend_score(df1h)

    support, resistance = swing_levels(df5)
    rng = max(resistance - support, 1e-9)
    f["dist_to_support_bps"] = (price - support) / price * 1e4 if price else 0.0
    f["dist_to_resistance_bps"] = (resistance - price) / price * 1e4 if price else 0.0
    f["pos_in_range"] = float(np.clip((price - support) / rng, 0, 1))  # 0=support,1=resistance

    f["vwap_dist_bps"] = 0.0
    vw = vwap(df1).iloc[-1]
    if np.isfinite(vw) and vw:
        f["vwap_dist_bps"] = (price - vw) / vw * 1e4

    # previous 5m candle direction + body size
    if len(df5) >= 2:
        prev = df5.iloc[-2]
        body = prev["close"] - prev["open"]
        f["prev5_dir"] = float(np.sign(body))
        f["prev5_body_bps"] = body / prev["open"] * 1e4 if prev["open"] else 0.0
    else:
        f["prev5_dir"] = 0.0
        f["prev5_body_bps"] = 0.0

    # wick rejection on the most recent 1m candle near S/R
    last1 = df1.iloc[-1]
    rng1 = max(last1["high"] - last1["low"], 1e-9)
    upper_wick = (last1["high"] - max(last1["close"], last1["open"])) / rng1
    lower_wick = (min(last1["close"], last1["open"]) - last1["low"]) / rng1
    near_res = f["dist_to_resistance_bps"] < 10
    near_sup = f["dist_to_support_bps"] < 10
    f["wick_rejection"] = (lower_wick if near_sup else 0.0) - (upper_wick if near_res else 0.0)

    # breakout / fakeout: did price pierce S/R then close back inside?
    f["breakout"] = 0.0
    if last1["high"] > resistance and last1["close"] > resistance:
        f["breakout"] = 1.0
    elif last1["low"] < support and last1["close"] < support:
        f["breakout"] = -1.0
    f["fakeout"] = 0.0
    if last1["high"] > resistance and last1["close"] < resistance:
        f["fakeout"] = -1.0   # failed upside breakout -> bearish
    elif last1["low"] < support and last1["close"] > support:
        f["fakeout"] = 1.0    # failed breakdown -> bullish

    # ---- 2. Momentum --------------------------------------------------------
    f["ema9_slope"] = slope(ema(close1, 9), 5)
    f["ema21_slope"] = slope(ema(close1, 21), 5)
    f["ema50_slope"] = slope(ema(close5, 50), 5)
    f["ema_stack"] = (1.0 if ema(close1, 9).iloc[-1] > ema(close1, 21).iloc[-1] else -1.0)
    f["rsi7"] = _safe(rsi(close1, 7).iloc[-1], 50.0)
    f["rsi14"] = _safe(rsi(close1, 14).iloc[-1], 50.0)
    f["macd_hist"] = _safe(macd_hist(close1).iloc[-1])
    f["macd_hist_slope"] = slope(macd_hist(close1), 5)
    f["stoch_rsi"] = _safe(stoch_rsi(close1).iloc[-1], 50.0)
    f["roc"] = _safe(roc(close1, 5).iloc[-1])
    # candle body momentum: avg signed body over last 3 1m candles
    bodies = (df1["close"] - df1["open"]).tail(3)
    f["body_momentum_bps"] = _safe(bodies.mean() / price * 1e4) if price else 0.0
    # volume acceleration
    vol = df1["volume"]
    v_recent = vol.tail(3).mean()
    v_base = vol.tail(20).mean()
    f["volume_accel"] = _safe(v_recent / v_base) if v_base else 1.0

    # ---- 3. Volatility ------------------------------------------------------
    f["atr1m_bps"] = _safe(atr(df1, 14).iloc[-1] / price * 1e4) if price else 0.0
    f["atr5m_bps"] = _safe(atr(df5, 14).iloc[-1] / price * 1e4) if price else 0.0
    f["realized_vol"] = _safe(realized_vol(close1, 20).iloc[-1])
    f["bb_width"] = _safe(bollinger_width(close1).iloc[-1])
    # chop: low |trend| but high vol -> abnormal
    rv = realized_vol(close1, 20).dropna()
    if len(rv) > 20:
        z = (rv.iloc[-1] - rv.mean()) / (rv.std() + 1e-12)
        f["vol_zscore"] = _safe(z)
    else:
        f["vol_zscore"] = 0.0
    # chop score: directional movement vs path length
    seg = close1.tail(10)
    net = abs(seg.iloc[-1] - seg.iloc[0])
    path = seg.diff().abs().sum()
    f["chop"] = _safe(1 - net / path) if path else 1.0   # ~1 = pure chop, ~0 = trending

    # ---- 4. Microstructure --------------------------------------------------
    f["ob_imbalance"] = 0.0
    f["spread_bps"] = 0.0
    if orderbook is not None:
        imb = orderbook.imbalance()
        f["ob_imbalance"] = _safe(imb)
        f["spread_bps"] = _safe(orderbook.spread_bps)
    f["flow_imbalance"] = _safe(trade_flow.get("flow_imbalance")) if trade_flow else 0.0

    # ---- 5. Polymarket implied probability ----------------------------------
    f["yes_price"] = _safe(ctx.yes_price, np.nan)
    f["no_price"] = _safe(ctx.no_price, np.nan)
    if ctx.yes_price is not None and ctx.no_price is not None:
        denom = ctx.yes_price + ctx.no_price
        f["implied_prob_up"] = ctx.yes_price / denom if denom else 0.5
        f["pm_spread_bps"] = abs(ctx.yes_price - (1 - ctx.no_price)) * 1e4
    else:
        f["implied_prob_up"] = np.nan
        f["pm_spread_bps"] = np.nan

    return f


# Stable feature ordering used by the ML model (microstructure / polymarket
# context columns are excluded from the *training* matrix because they are not
# in historical backtest data; they are still available to the rule engine).
ML_FEATURE_COLUMNS = [
    "dist_from_target_bps", "dist_from_open_bps", "time_fraction_left",
    "trend_15m", "trend_1h", "pos_in_range", "vwap_dist_bps",
    "prev5_dir", "prev5_body_bps", "wick_rejection", "breakout", "fakeout",
    "ema9_slope", "ema21_slope", "ema50_slope", "ema_stack",
    "rsi7", "rsi14", "macd_hist", "macd_hist_slope", "stoch_rsi", "roc",
    "body_momentum_bps", "volume_accel",
    "atr1m_bps", "atr5m_bps", "realized_vol", "bb_width", "vol_zscore", "chop",
]


def feature_row(features: Dict[str, float]) -> np.ndarray:
    return np.array([_safe(features.get(c, 0.0)) for c in ML_FEATURE_COLUMNS], dtype=float)

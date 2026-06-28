"""
strategy.py
===========
Hybrid signal engine + decision logic.

Flow:
  features -> rule_score (weighted) -> blend with calibrated ML prob
           -> directional probability
           -> compare against Polymarket implied prob -> expected value
           -> apply confidence / edge / volatility / spread / timing gates
           -> Signal(side=UP|DOWN|SKIP, reasons, EV, ...)

The cardinal rule of this bot: predicting a direction is NOT enough. We only
emit UP/DOWN when (a) the model is confident, (b) the market is mispriced
enough that EV after costs clears the buffer, and (c) no risk filter trips.
Otherwise we SKIP — and by design we expect to SKIP most markets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from features import MarketContext, feature_row


@dataclass
class Signal:
    asset: str
    side: str                       # "UP" | "DOWN" | "SKIP"
    probability_up: float
    probability_down: float
    confidence: float
    expected_value: float           # fraction of stake, after costs, for the chosen side
    implied_prob_up: Optional[float]
    entry_price: Optional[float]    # Polymarket price paid for the chosen side
    reasons: List[str] = field(default_factory=list)
    skip_reason: Optional[str] = None
    rule_score: float = 0.0
    ml_prob_up: Optional[float] = None
    seconds_remaining: float = 0.0

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["reasons"] = "; ".join(self.reasons)
        return d


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class SignalEngine:
    def __init__(self, cfg: dict, model=None):
        self.cfg = cfg
        self.model = model
        self.weights = cfg["weights"]
        self.gates = cfg["gates"]
        self.costs = cfg["costs"]
        mcfg = cfg.get("model", {})
        self.blend_rules = mcfg.get("blend_rules", 0.4)
        self.blend_ml = mcfg.get("blend_ml", 0.6)

    # -------------------------------------------------- rule-based components
    def _market_structure_score(self, f: Dict[str, float]) -> float:
        s = 0.0
        s += 0.4 * np.tanh((f["trend_15m"] + f["trend_1h"]))
        # position in range: near support -> bullish bias, near resistance -> bearish
        s += 0.3 * (0.5 - f["pos_in_range"]) * 2
        s += 0.2 * np.tanh(f["wick_rejection"] * 3)
        s += 0.3 * f["breakout"] + 0.2 * f["fakeout"]
        s += 0.15 * f["prev5_dir"] * np.tanh(abs(f["prev5_body_bps"]) / 20)
        s += -0.15 * np.tanh(f["vwap_dist_bps"] / 30)  # mean-reversion to VWAP
        return float(np.clip(s, -1, 1))

    def _momentum_score(self, f: Dict[str, float]) -> float:
        s = 0.0
        s += 0.25 * np.tanh(f["ema9_slope"] * 5)
        s += 0.15 * np.tanh(f["ema21_slope"] * 5)
        s += 0.15 * f["ema_stack"]
        s += 0.15 * np.tanh((f["rsi7"] - 50) / 20)
        s += 0.10 * np.tanh((f["rsi14"] - 50) / 20)
        s += 0.15 * np.tanh(f["macd_hist_slope"] * 3)
        s += 0.10 * np.tanh(f["roc"] / 2)
        s += 0.10 * np.tanh(f["body_momentum_bps"] / 15)
        # stoch RSI extremes -> fade
        if f["stoch_rsi"] > 85:
            s -= 0.1
        elif f["stoch_rsi"] < 15:
            s += 0.1
        return float(np.clip(s, -1, 1))

    def _volatility_score(self, f: Dict[str, float]) -> float:
        """Volatility score gates conviction down, it does not pick direction.
        Returns a [0,1] *quality* multiplier-style value mapped to [-? ] via chop.
        Here we return a signed-neutral score that penalises chop (push toward 0).
        """
        # high chop or abnormal vol -> negative magnitude (handled as confidence cut)
        penalty = 0.0
        penalty -= 0.6 * f["chop"]
        if abs(f["vol_zscore"]) > self.gates["abnormal_vol_zscore"]:
            penalty -= 0.4
        return float(np.clip(penalty, -1, 0))  # always <=0, acts as damper

    def _microstructure_score(self, f: Dict[str, float]) -> float:
        s = 0.0
        s += 0.6 * np.tanh(f.get("ob_imbalance", 0.0) * 2)
        s += 0.4 * np.tanh(f.get("flow_imbalance", 0.0) * 2)
        return float(np.clip(s, -1, 1))

    _DEFAULT_KEYS = (
        "trend_15m", "trend_1h", "pos_in_range", "wick_rejection", "breakout",
        "fakeout", "prev5_dir", "prev5_body_bps", "vwap_dist_bps", "ema9_slope",
        "ema21_slope", "ema_stack", "rsi7", "rsi14", "macd_hist_slope", "roc",
        "body_momentum_bps", "stoch_rsi", "chop", "vol_zscore", "ob_imbalance",
        "flow_imbalance",
    )

    def rule_probability_up(self, features: Dict[str, float]) -> (float, float):
        """Return (prob_up, raw_directional_score)."""
        f = {k: features.get(k, 50.0 if k in ("rsi7", "rsi14", "stoch_rsi") else 0.0)
             for k in self._DEFAULT_KEYS}
        ms = self._market_structure_score(f)
        mo = self._momentum_score(f)
        mi = self._microstructure_score(f)
        vol_damper = self._volatility_score(f)   # <= 0

        w = self.weights
        directional = (w["market_structure"] * ms +
                       w["momentum"] * mo +
                       w["microstructure"] * mi)
        norm = (w["market_structure"] + w["momentum"] + w["microstructure"]) or 1.0
        directional /= norm
        # apply volatility damper: shrink magnitude toward 0 in chop
        damp = 1.0 + w["volatility"] * vol_damper  # vol_damper<=0 -> damp<=1
        directional *= max(damp, 0.2)
        prob_up = _sigmoid(directional * 3.0)     # scale into a usable probability
        return float(prob_up), float(directional)

    # -------------------------------------------------------- ensemble + EV
    def evaluate(self, asset: str, features: Dict[str, float], ctx: MarketContext) -> Signal:
        rule_prob_up, rule_score = self.rule_probability_up(features)

        ml_prob_up = None
        if self.model is not None:
            ml_prob_up = self.model.predict_proba_up(feature_row(features))

        if ml_prob_up is not None:
            prob_up = self.blend_rules * rule_prob_up + self.blend_ml * ml_prob_up
        else:
            prob_up = rule_prob_up
        prob_up = float(np.clip(prob_up, 0.001, 0.999))
        prob_down = 1.0 - prob_up

        # confidence = distance from 50/50, damped by volatility quality
        base_conf = abs(prob_up - 0.5) * 2
        vol_quality = 1.0 + features.get("chop", 0) * -0.5
        confidence = float(np.clip(base_conf * vol_quality, 0, 1))

        sig = Signal(
            asset=asset,
            side="SKIP",
            probability_up=round(prob_up, 4),
            probability_down=round(prob_down, 4),
            confidence=round(confidence, 4),
            expected_value=0.0,
            implied_prob_up=features.get("implied_prob_up"),
            entry_price=None,
            rule_score=round(rule_score, 4),
            ml_prob_up=None if ml_prob_up is None else round(ml_prob_up, 4),
            seconds_remaining=ctx.seconds_remaining,
        )

        # choose candidate side from model
        if prob_up >= 0.5:
            side, model_prob, market_price = "UP", prob_up, ctx.yes_price
        else:
            side, model_prob, market_price = "DOWN", prob_down, ctx.no_price

        ev, entry_price = self._expected_value(model_prob, market_price)
        sig.expected_value = round(ev, 4)
        sig.entry_price = entry_price

        # ---- gating ---------------------------------------------------------
        skip = self._apply_gates(features, ctx, side, model_prob, ev, confidence)
        if skip:
            sig.skip_reason = skip
            sig.reasons = self._explain(features, side, model_prob)
            return sig

        sig.side = side
        sig.reasons = self._explain(features, side, model_prob)
        sig.reasons.append(f"EV after costs {ev:.1%} clears buffer")
        return sig

    def _expected_value(self, model_prob: float, market_price: Optional[float]):
        """EV per unit stake after slippage/fees for buying the chosen outcome.

        Polymarket share pays $1 if correct, costs `price` to buy.
        EV_$ per share = model_prob*1 - effective_price ; stake == effective_price.
        EV_fraction = (model_prob - effective_price - fee) / effective_price.
        """
        if market_price is None or market_price <= 0 or market_price >= 1:
            return -1.0, None
        slip = self.costs.get("slippage_bps", 0) / 1e4
        fee = self.costs.get("taker_fee", 0)
        eff = min(market_price * (1 + slip), 0.999)
        ev_dollar = model_prob - eff - fee
        ev_fraction = ev_dollar / eff
        return float(ev_fraction), float(eff)

    def _apply_gates(self, f, ctx, side, model_prob, ev, confidence) -> Optional[str]:
        g = self.gates
        strong = model_prob >= g["strong_edge_probability"]

        if model_prob < g["min_probability"] and not strong:
            return f"prob {model_prob:.2f} < min {g['min_probability']:.2f}"

        if confidence < (g["min_probability"] - 0.5) * 2 and not strong:
            return f"confidence {confidence:.2f} too low"

        # need a real market price to compute EV
        if ctx.yes_price is None or ctx.no_price is None:
            return "no Polymarket price available"

        required_ev = g["min_ev_after_fees"] + self.costs.get("safety_buffer_ev", 0)
        if ev < required_ev:
            return f"EV {ev:.1%} < required {required_ev:.1%} (no mispricing edge)"

        # spread gate
        pm_spread = f.get("pm_spread_bps")
        if pm_spread is not None and np.isfinite(pm_spread) and pm_spread > g["max_spread_bps"]:
            return f"PM spread {pm_spread:.0f}bps > {g['max_spread_bps']}"

        # timing gate
        if ctx.seconds_remaining < g["min_seconds_remaining"] and not strong:
            return f"{ctx.seconds_remaining:.0f}s left < {g['min_seconds_remaining']}s (weak edge)"

        # coin-flip near expiry: price hugging target with no mispricing
        if ctx.seconds_remaining <= g["near_expiry_seconds"]:
            if abs(f.get("dist_from_target_bps", 0)) < g["max_distance_to_target_bps"] and not strong:
                return "price hugging target near expiry (coin-flip)"

        # abnormal volatility / news
        if g.get("block_news_volatility", True):
            if abs(f.get("vol_zscore", 0)) > g["abnormal_vol_zscore"]:
                return f"abnormal volatility (z={f.get('vol_zscore', 0):.1f})"

        # chop filter
        if f.get("chop", 0) > 0.85 and not strong:
            return "market chopping (no clean direction)"

        return None

    @staticmethod
    def _explain(f, side, model_prob) -> List[str]:
        r = [f"model {side} p={model_prob:.2f}"]
        if f.get("trend_1h", 0) and np.sign(f["trend_1h"]) == (1 if side == "UP" else -1):
            r.append("1h trend aligned")
        if f.get("ob_imbalance", 0):
            d = "bid" if f["ob_imbalance"] > 0 else "ask"
            r.append(f"orderbook {d}-heavy ({f['ob_imbalance']:+.2f})")
        if abs(f.get("macd_hist_slope", 0)) > 0.01:
            r.append(f"MACD hist slope {f['macd_hist_slope']:+.3f}")
        if f.get("breakout", 0):
            r.append("breakout" if f["breakout"] > 0 else "breakdown")
        if f.get("fakeout", 0):
            r.append("fakeout reversal")
        if abs(f.get("wick_rejection", 0)) > 0.2:
            r.append("wick rejection at S/R")
        return r

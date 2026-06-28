# BTC/ETH 5-Minute Polymarket Up/Down Bot

A production-shaped prediction bot that decides whether the current **5-minute
BTC or ETH candle** will close **UP or DOWN** versus the Polymarket
target/open price — and **only trades when the edge clears fees, slippage, and a
safety buffer.**

> ⚠️ **This is not a guaranteed-win bot. No such thing exists.** The edge here
> comes from *selectivity*: the bot is built to **skip the large majority of
> markets** and trade only when price action, market structure, momentum,
> volatility, order flow, **and** Polymarket mispricing all line up. Backtest
> and paper-trade before risking real money. You can lose money.

---

## How it works

```
Binance OHLCV (1m/5m/15m/1h)  ┐
Order book + trade flow        ├─► features.py ─► strategy.py ─► Signal(UP/DOWN/SKIP)
Polymarket market (target/odds)┘        │             │
                                        ▼             ▼
                                    model.py     risk_manager.py ─► SQLite ledger
                                  (LightGBM +        (gates: stake, daily loss,
                                   calibration)       drawdown, consec losses)
```

The directional probability is an **ensemble**:

1. **Rule-based weighted score** across four pillars
   (market structure, momentum, volatility damper, microstructure).
2. **LightGBM classifier** trained on engineered features, with an
   **isotonic calibration** layer so the probability is honest (critical for EV).
3. Logistic-regression / rules-only **fallbacks** when data or libraries are
   missing.

A signal is only actionable when **all** of these hold:

| Gate | Default |
|------|---------|
| model probability for the side | ≥ 58% (≥ 66% allows late/borderline entries) |
| EV after fees + slippage + buffer | ≥ 3% |
| Polymarket YES/NO spread | ≤ 150 bps |
| time remaining | ≥ 45s (unless strong edge) |
| near expiry & price hugging target | skip (coin-flip) |
| abnormal volatility (z-score) | skip |
| chop filter | skip |
| open trades per asset | ≤ 1 |

Everything is configurable in **`config.yaml`**.

---

## Features computed (`features.py`)

- **Market structure:** price vs candle open / target, distance in bps, time
  remaining, 15m & 1h trend, swing-high/low support & resistance, VWAP distance,
  previous 5m candle direction & body size, wick rejection at S/R, breakout /
  fakeout detection.
- **Momentum:** EMA 9/21/50 slope & stack, RSI 7 & 14, MACD histogram slope,
  Stochastic RSI, rate of change, candle body momentum, volume acceleration.
- **Volatility:** ATR (1m & 5m), realized vol, Bollinger band width, chop score,
  volatility z-score (abnormal-move filter).
- **Microstructure (live only):** Binance order-book imbalance, bid/ask spread,
  aggressive buy/sell trade-flow imbalance.
- **Polymarket:** target price, expiry, YES/NO prices, implied probability,
  model-vs-market edge → expected value.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI + orchestration; backtest / paper / live / fetch-data / train |
| `data_fetcher.py` | ccxt OHLCV, order book, trade flow, historical CSV loader |
| `features.py` | indicator primitives + unified feature builder |
| `strategy.py` | rule engine, ensemble blend, EV, confidence gates → `Signal` |
| `model.py` | LightGBM + calibration ensemble with safe fallbacks |
| `polymarket_client.py` | Gamma/CLOB market discovery + live order placement |
| `risk_manager.py` | risk controls + SQLite trade ledger |
| `backtest.py` | event-driven backtester + metrics |
| `dashboard.py` | Streamlit dashboard |
| `config.yaml` | all thresholds, weights, risk limits |
| `requirements.txt`, `.env.example` | setup |

---

## Setup

```bash
git clone <repo> && cd gpt-claude-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys only if you intend to live-trade
```

Public Binance market data and Polymarket discovery need **no API keys**.
Keys are only required to place **live** orders.

---

## Usage

### 1. Download historical data
```bash
python main.py fetch-data --days 30
```
Writes `data/BTC_1m.csv` and `data/ETH_1m.csv`.

### 2. Backtest (required before anything live)
```bash
python main.py backtest
```
Trains the model on a time-ordered split and simulates the out-of-sample period.
Prints and saves (`artifacts/backtest_results.json`):
win rate, profit after fees, max drawdown, Sharpe, average EV, breakdowns by
time-remaining / volatility regime / asset, a confusion matrix, and a
calibration curve. It ends with a **PASS/FAIL verdict** and refuses to bless
live trading on a losing backtest.

> **Backtest honesty:** real historical Polymarket quotes are not bundled, so
> the backtester models a *momentum-naive* counterparty + noise as the market
> price. This tests whether the model has edge over a naive market — it is
> **not** a promise of live profit. For a true backtest, record live Polymarket
> order books and replace `synthetic_price()` in `backtest.py`.

### 3. Paper trade (default)
```bash
python main.py paper
```
Polls live Polymarket 5m markets, computes signals against live Binance data,
and logs simulated fills to SQLite — **no real orders**.

### 4. Live trade (only after profitable backtest **and** paper test)
Set in `config.yaml`:
```yaml
mode: live
```
and in `.env`:
```
LIVE_TRADING=true
POLYMARKET_PRIVATE_KEY=...
```
Then:
```bash
pip install py-clob-client
python main.py live
```
Live mode is **refused** unless both the config mode is `live` and
`LIVE_TRADING=true`.

### 5. Dashboard
```bash
streamlit run dashboard.py
```

---

## Risk management

- Fixed small stake per trade — **no martingale, no revenge trading, no
  averaging into losers.**
- Daily max loss, max consecutive losses, and max drawdown each **halt** trading.
- Max 1 open trade per asset per window.
- Paper mode first; live gated behind config + env flags.

Tune every limit in `config.yaml` under `risk:`.

---

## Responsible use

This software is for research and education. Prediction markets are speculative
and may be restricted in your jurisdiction — check local law. Trade only money
you can afford to lose. Past backtest performance does not guarantee future
results.

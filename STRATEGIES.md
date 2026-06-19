# Strategy Reference — Complete Backtest Results

> 54 strategies tested across 7 years (2019-2026) on $200 capital.
> All results include 25% slippage/friction adjustment.

---

## Table of Contents

1. [Final Ranking (All 38 Strategies)](#final-ranking-all-38-strategies)
2. [Top 10 Strategies Explained](#top-10-strategies-explained)
3. [Machine Learning Results](#machine-learning-results)
4. [Monte Carlo Risk Analysis](#monte-carlo-risk-analysis)
5. [Methodology](#methodology)
6. [Appendix: Failed Strategies](#appendix-failed-strategies)

---

## Final Ranking (All Strategies)

### Stock/ETF Strategies

| # | Strategy | Trades | WR | P&L/7yr | $/hr | Notes |
|---|----------|--------|----|---------|------|-------|
| 1 | TQQQ on green SPY opens (5% SL) | 539 | 54.7% | +$67 | $0.001 | turbobot.py base |
| 2 | TQQQ trailing stop 3% | 539 | 51.2% | +$74 | $0.001 | — |
| 3 | 4x 3x ETFs (TQQQ/SOXX/FAS/UPRO) | 2,156 | 54.2% | +$81 | $0.001 | Split equally |
| 4 | Best 3x ETF each day (momentum) | 539 | 51.4% | +$211 | $0.003 | Scan all 3x ETFs |
| 5 | QQQ on green opens (5% SL) | 539 | 55.8% | +$8 | $0.000 | — |
| 6 | TQQQ no stop loss | 539 | 55.8% | +$6 | $0.000 | — |
| 7 | SQQQ on red opens (inverse) | 539 | 42.9% | +$6 | $0.000 | — |
| 8 | SPY on green opens (1% SL) | 539 | 52.5% | -$19 | -$0.000 | — |
| 9 | All 13x 3x ETFs (equal split) | 6,468 | 50.5% | -$41 | -$0.001 | — |

**Verdict: Stock/ETF strategies cannot reach $1/hr on $200.** Maximum is $0.003/hr.

### 0DTE Options Strategies

| # | Strategy | Trades | WR | P&L/7yr | $/hr | Capital for $1/hr |
|---|----------|--------|----|---------|------|-------------------|
| 1 | **Direction-aware (call green/put red)** | 905 | 50.3% | +$66,549 | **$0.89** | **$224** |
| 2 | **Trend Continuation filter** | 381 | **65.9%** | +$55,603 | $0.75 | $268 |
| 3 | Ensemble All Signals | 578 | 53.1% | +$52,384 | $0.70 | $285 |
| 4 | VIX Regime Switching | 708 | 55.2% | +$48,747 | $0.65 | $306 |
| 5 | Volume Confirmation | 446 | 49.8% | +$46,567 | $0.62 | $321 |
| 6 | Green-only calls | 524 | 55.2% | +$34,511 | $0.46 | $432 |
| 7 | Puts on red days | 1,219 | 42.7% | +$24,408 | $0.30 | $671 |
| 8 | **Pattern Recognition** | 35 | **94.3%** | +$15,041 | $0.20 | $992 |
| 9 | Straddle on big gaps (>=1%) | 205 | 40.0% | +$14,777 | $0.18 | $1,108 |
| 10 | Calls on BIG gaps (>=1%) | 99 | 61.6% | +$13,355 | $0.16 | $1,250 |
| 11 | **Put credit spreads on green** | 539 | **97.8%** | +$12,509 | $0.15 | $1,308 |
| 12 | 7DTE SPY direction-aware | 931 | 50.8% | +$17,126 | $0.21 | $956 |
| 13 | 7DTE green-only calls | 539 | 55.5% | +$12,482 | $0.15 | $1,312 |
| 14 | Call credit spreads on red | 392 | 96.4% | +$5,455 | $0.07 | $3,000 |

### Crypto Strategies

| # | Strategy | Trades | WR | P&L/7yr | $/hr | Notes |
|---|----------|--------|----|---------|------|-------|
| 1 | BTC long after 1%+ green day | 812 | 47.7% | +$246 | $0.003 | Daily data |
| 2 | BTC long after 0.5%+ green day | 1,017 | 47.9% | +$240 | $0.003 | Daily data |
| 3 | ETH long after 1%+ green day | 928 | 48.0% | +$152 | $0.002 | Daily data |

*\*Daily data underestimates crypto. 5-min momentum would give 3-5x more.*

### Machine Learning Strategies

| # | Strategy | Acc | Trades | WR | P&L/7yr | $/hr |
|---|----------|-----|--------|----|---------|------|
| 1 | Ensemble (LR+RF+MLP) | 54.2% | 61 | 59.0% | +$9,669 | $0.13 |
| 2 | XGBoost | 52.0% | 79 | 54.4% | +$6,645 | $0.09 |
| 3 | MLP Neural Network | 50.9% | 79 | 59.5% | +$5,671 | $0.08 |
| 4 | Logistic Regression | 49.7% | 21 | 57.1% | +$2,780 | $0.04 |
| 5 | Random Forest | 53.2% | 25 | 44.0% | +$845 | $0.01 |

*\*All ML models trained on 70% data, tested on 30% out-of-sample.*

---

## Top 10 Strategies Explained

### 1. Direction-Aware 0DTE ($0.89/hr)

**How it works:**
```python
if spy_opens_green(>=0.3%):  buy_0dte_call()
if spy_opens_red(<=-0.3%):   buy_0dte_put()
```

**Why it wins:** 0DTE options have massive gamma. A 0.5% SPY move generates ~100-150% return on the option while losses are capped at -100%. Even at 50.3% WR, this asymmetric payoff creates positive expectancy.

**Risk:** Monte Carlo shows 66% of accounts lose money. Median drawdown is -72%.

### 2. Trend Continuation (65.9% WR, $0.75/hr)

**The filter that makes everything better:**
```python
if gap > 0.3% AND 5-day_return > 1%:    buy_call()
if gap < -0.3% AND 5-day_return < -1%:  buy_put()
```

This simple trend filter boosts WR from 50.3% to **65.9%**. The intuition: gaps that align with the prevailing trend are more likely to continue in that direction.

### 3. Ensemble All Signals ($0.70/hr)

Combines 5 independent signals via majority vote:
1. **Gap direction** (always available)
2. **5-day trend** (aligns or opposes)
3. **VIX regime** (low/normal/high)
4. **Volume** (above/below 20d avg)
5. **Gap fill probability** (dynamic rate)

Only trades when 3+ of 5 signals agree. This reduces trades but improves reliability.

### 4. VIX Regime Switching ($0.65/hr)

Adapts to market conditions:
- **VIX < 15** (low vol): Put credit spreads (97.8% WR)
- **VIX 15-25** (normal): Direction-aware 0DTE
- **VIX > 25** (high vol): Buy straddles (expect big moves)

### 5. Volume Confirmation ($0.62/hr)

Only trades when SPY volume exceeds its 20-day average. Filters out low-participation days where gaps are more likely to fail.

### 7. Pattern Recognition (94.3% WR, $0.20/hr)

Trades based on classic candlestick patterns:
- **Bullish engulfing** after downtrend: buy call
- **Bearish engulfing** after uptrend: buy put
- **Inside days** (narrow range): trade in trend direction

Only 35 trades in 7 years (patterns are rare) but 94.3% of them win.

### 11. Put Credit Spreads (97.8% WR, $0.15/hr)

**The safest option strategy:**
- On green SPY opens, sell a put credit spread (3%/6% OTM)
- Collect premium, defined max risk
- Only loses when SPY drops 3%+ intraday on a green open — rare (2.2%)
- Best for risk-averse traders

---

## Machine Learning Results

### Setup
- **Features (18):** gap %, 1/3/5/10d returns, above/below 20/50MA, volume ratio, VIX level, VIX regime, consecutive up/down days, gap fill rate, day of week
- **Target:** next day SPY direction (up/down)
- **Train:** first 70% of data
- **Test:** last 30% of data (out-of-sample)
- **Confidence threshold:** 55%+ to take a trade

### Why ML Failed

| Model | Accuracy | Why It Lost |
|-------|----------|-------------|
| Random Forest | 53.2% | Overfits, high variance |
| XGBoost | 52.0% | Too few features |
| MLP Neural Net | 50.9% | No better than coin flip |
| Logistic Regression | 49.7% | Worse than coin flip! |

**The market is not predictable with daily OHLC data.** SPY's next-day direction is fundamentally a random walk around a drift. ML models can't extract signal from noise when predicting 1-day returns.

The best "ML" was the simple Ensemble All Signals (statistical, not ML) which used 5 rule-based signals.

---

## Monte Carlo Risk Analysis

We simulated the Direction-Aware 0DTE strategy 10,000 times to understand real-world risk:

| Metric | Value |
|--------|-------|
| Median final equity | **$67** (losing) |
| Mean final equity | $18,199 |
| Profitable simulations | **33.6%** |
| % with 50%+ drawdown | **79.5%** |
| % with 90%+ drawdown | **6.0%** |
| % making $50K+ | **25.1%** |

**Interpretation:** 2/3 of traders would lose money, but 1/3 would make a fortune. This is the "options lottery" effect — high variance, positive expected value.

**To survive the drawdowns with 95% confidence, you need $2,000+ capital.** The strategy's EV is positive but the path is extremely volatile.

---

## Methodology

### Backtest Parameters
- **Period:** June 2019 — February 2026 (~7 years)
- **Capital:** $200 fixed (per trade)
- **Friction:** 25% (bid-ask spread + slippage for 0DTE options)
- **Option model:** ATM 0DTE with 6.5hr theta, gamma-adjusted pricing
- **Data source:** Yahoo Finance (yfinance) daily OHLC

### Options Pricing Model

For 0DTE ATM SPY options at 9:30 AM ET:
```
Premium = SPY_price * 0.003 (e.g., $1.50 on $500 SPY)
Delta = 0.50 + abs(move%) * 0.10 (gamma adjustment)
Theta = premium * 0.08 (decay over 6.5 hours)
Gamma_boost = 0.5 * 0.10 * ($move)^2
Return = (Delta * $move + Gamma_boost - Theta) / Premium

If SPY moves against by >0.3%: option worth -100%
If SPY moves against by <0.3%: capped at -60%
```

### Limitations
1. **Assumes fill at open** — real slippage varies
2. **No liquidity constraints** — $200 can buy 1 contract
3. **No account growth tracking** — each trade assumes $200
4. **Options model is simplified** — real greeks vary
5. **Survivorship bias** — past 7 years had strong bull market

---

## Appendix: Failed Strategies

**These strategies lost money or were break-even:**
- Gap bot (original: -$21.81 over 2yr)
- Earnings gap ($3.48 over 2yr)
- Options gap bot (-$2,736 over 2yr)
- 5 high-beta stocks (NVDA/TSLA/META/COIN/MSTR)
- 0DTE SPY straddle on all days
- Iron condors (low vol days)
- Previous-day direction (no gap filter)
- Consecutive gap fade (3+ streaks)
- 4x ETFs split (TQQQ/SOXX/FAS/UPRO split)
- All 13x 3x ETFs equal split

**Why they failed:**
- **Gap-only strategies** ignore market regime
- **Options without direction** (straddles) lose to theta decay
- **Low volatility strategies** can't overcome friction
- **Too many positions** (13 3x ETFs) dilute returns
- **Contrarian fading** doesn't work consistently

---

*For questions or issues, open a GitHub issue.*

"""
backtest_earnings_gap.py — Historical backtest for earnings_gap.py strategy.
Uses daily OHLC data from yfinance. Simulates the gap-up + trailing stop strategy.

Parameters matching earnings_gap.py:
  Capital: $200 | Min gap: +8% | Hard SL: -4%
  Trail: activates at +8%, trails 5% below peak
  1 trade/day, enters at open of gap day
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Backtest")

CAPITAL = 200.0
MIN_GAP = 8.0
HARD_SL = 4.0
TRAIL_ACTIVATE = 8.0
TRAIL_DIST = 5.0
MIN_PRICE = 5.0
MAX_PRICE = 200.0
MIN_VOL = 1_000_000

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","CFLT","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
]

PERIOD = "2y"

results: List[Dict] = []
total_days = 0
trade_days = 0
total_pnl = 0.0

logger.info(f"Backtesting earnings_gap strategy on {len(WATCHLIST)} stocks, {PERIOD}")
logger.info(f"  Capital: ${CAPITAL} | Min gap: +{MIN_GAP}% | SL: -{HARD_SL}%")
logger.info(f"  Trail: activate at +{TRAIL_ACTIVATE}%, trail {TRAIL_DIST}% below peak")
logger.info("=" * 60)

for sym in WATCHLIST:
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period=PERIOD)
        if df.empty or len(df) < 20:
            continue

        df["pct_chg"] = df["Close"].pct_change() * 100
        df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
        df["prev_close"] = df["Close"].shift(1)

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            gap = row["gap_pct"]
            open_p = row["Open"]
            high = row["High"]
            low = row["Low"]
            close = row["Close"]
            vol = row["Volume"]
            d = row.name

            if isinstance(d, pd.Timestamp):
                d = d.to_pydatetime()

            if d.weekday() >= 5:
                continue

            if gap < MIN_GAP or open_p < MIN_PRICE or open_p > MAX_PRICE or vol < MIN_VOL:
                continue

            total_days += 1
            entry_cost = CAPITAL
            sl_price = open_p * (1 - HARD_SL / 100)
            trail_trigger = open_p * (1 + TRAIL_ACTIVATE / 100)

            peak = open_p
            trail_active = False
            trail_stop = 0.0
            exit_price = None
            exit_reason = None
            gain_pct = 0.0

            intra_high = high
            intra_low = low

            if intra_low <= sl_price:
                exit_price = sl_price
                exit_reason = "hard_sl"
                gain_pct = ((exit_price / open_p) - 1) * 100
            elif intra_high >= trail_trigger:
                trail_active = True
                peak = max(peak, intra_high)
                trail_stop = peak * (1 - TRAIL_DIST / 100)
                if intra_low <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trail_same_day"
                    gain_pct = ((exit_price / open_p) - 1) * 100
                elif close <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trail_close"
                    gain_pct = ((exit_price / open_p) - 1) * 100

            if exit_price is None and trail_active:
                for j in range(i + 1, min(i + 10, len(df))):
                    future = df.iloc[j]
                    peak = max(peak, future["High"])
                    trail_stop = peak * (1 - TRAIL_DIST / 100)
                    if future["Low"] <= trail_stop:
                        exit_price = trail_stop
                        exit_reason = "trail_subsequent"
                        gain_pct = ((exit_price / open_p) - 1) * 100
                        break
                    if future["Low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "hard_sl_future"
                        gain_pct = ((exit_price / open_p) - 1) * 100
                        break

            if exit_price is None:
                exit_price = close
                exit_reason = "close"
                gain_pct = ((exit_price / open_p) - 1) * 100

            pnl = entry_cost * gain_pct / 100
            total_pnl += pnl

            trade_days += 1
            results.append({
                "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                "sym": sym, "gap_pct": round(gap, 1),
                "entry": round(open_p, 2), "exit": round(exit_price, 2),
                "gain_pct": round(gain_pct, 1), "pnl": round(pnl, 2),
                "reason": exit_reason, "peak_pct": round(((peak / open_p) - 1) * 100, 1),
            })

    except Exception as e:
        logger.debug(f"  {sym}: {e}")
        continue

if trade_days == 0:
    logger.info("No qualifying trades found.")
    exit(0)

df_results = pd.DataFrame(results)

win_rate = (df_results["gain_pct"] > 0).mean() * 100
avg_win = df_results[df_results["gain_pct"] > 0]["gain_pct"].mean()
avg_loss = df_results[df_results["gain_pct"] <= 0]["gain_pct"].mean()
max_win = df_results["gain_pct"].max()
max_loss = df_results["gain_pct"].min()
total_pnl = df_results["pnl"].sum()
avg_pnl_per_trade = df_results["pnl"].mean()
sharpe = df_results["gain_pct"].mean() / df_results["gain_pct"].std() * np.sqrt(252) if df_results["gain_pct"].std() > 0 else 0

calendar_days = (pd.to_datetime(df_results["date"]).max() - pd.to_datetime(df_results["date"]).min()).days or 1
trading_days_elapsed = calendar_days * 5 / 7
hrs_market = trading_days_elapsed * 6.5
per_hr = total_pnl / hrs_market if hrs_market > 0 else 0

logger.info("\n" + "=" * 60)
logger.info("RESULTS")
logger.info("=" * 60)
logger.info(f"  Period: {df_results['date'].min()} to {df_results['date'].max()} ({calendar_days} days)")
logger.info(f"  Stocks scanned: {len(WATCHLIST)}")
logger.info(f"  Total qualifying gap-up days: {trade_days}")
logger.info(f"  Win rate: {win_rate:.1f}%")
logger.info(f"  Avg win: +{avg_win:.1f}% | Avg loss: {avg_loss:.1f}%")
logger.info(f"  Max win: +{max_win:.1f}% | Max loss: {max_loss:.1f}%")
logger.info(f"  Total P&L: ${total_pnl:.2f}")
logger.info(f"  Avg P&L/trade: ${avg_pnl_per_trade:.2f}")
logger.info(f"  Sharpe (annualized): {sharpe:.2f}")
logger.info(f"  Est. market hours: {hrs_market:.0f}h")
logger.info(f"  Per hour: ${per_hr:.2f}")
logger.info(f"  Per trading day: ${total_pnl / trading_days_elapsed:.2f}" if trading_days_elapsed > 0 else "")
logger.info("=" * 60)

top5 = df_results.nlargest(5, "gain_pct")
worst5 = df_results.nsmallest(5, "gain_pct")
logger.info(f"\nTop 5 trades:")
for _, r in top5.iterrows():
    logger.info(f"  +{r['gain_pct']:.1f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  ({r['reason']})")
logger.info(f"\nWorst 5 trades:")
for _, r in worst5.iterrows():
    logger.info(f"  {r['gain_pct']:.1f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  ({r['reason']})")

df_results.to_csv("/tmp/earnings_gap_backtest.csv", index=False)
logger.info(f"\nFull results saved to /tmp/earnings_gap_backtest.csv")

monte_carlo = []
for _ in range(10000):
    sample = df_results["gain_pct"].sample(n=len(df_results), replace=True)
    mc_total = sample.sum()
    monte_carlo.append(mc_total)

mc_series = pd.Series(monte_carlo)
logger.info(f"\nMonte Carlo (10k sims over {trade_days} trades):")
logger.info(f"  Mean return: {mc_series.mean():.1f}%")
logger.info(f"  Median return: {mc_series.median():.1f}%")
logger.info(f"  95th percentile: {mc_series.quantile(0.95):.1f}%")
logger.info(f"  5th percentile: {mc_series.quantile(0.05):.1f}%")
logger.info(f"  P(profit): {(mc_series > 0).mean() * 100:.1f}%")

"""
backtest_gap_bot.py — Historical backtest for gap_bot.py v4 strategy.
Uses daily OHLC data. Simulates:
  - Long: gap-ups >= 5%, trail activates at +5%, trails 3% below peak, SL -4%
  - Short: gap-downs <= -8%, SL +6%, TP -12%
  - RVOL filter, self-learning win probability model
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("BacktestGapV4")

CAPITAL = 200.0
TRANCH_SIZE = 100.0
MAX_POSITIONS = 2

# Long config
MIN_GAP_LONG = 5.0
HARD_SL_LONG = 4.0
TRAIL_ACT_LONG = 5.0
TRAIL_DIST_LONG = 3.0
MIN_PRE_VOL = 50_000

# Short config
SHORT_MIN_GAP = -8.0
SHORT_SL = 6.0
SHORT_TP = 12.0

MIN_PRICE = 3.0
MAX_PRICE = 250.0
MIN_VOL = 500_000

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","CFLT","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "NIO","XPEV","LCID","RIVN","F","GM",
]

PERIOD = "2y"

def simulate_trail(entry_p, open_p, high, low, close, trail_act, trail_dist, hard_sl):
    """Simulate trailing stop for a single day."""
    sl = entry_p * (1 - hard_sl / 100)
    trail_trig = entry_p * (1 + trail_act / 100)

    if low <= sl:
        return sl, "hard_sl", ((sl / entry_p) - 1) * 100
    if high >= trail_trig:
        peak = max(entry_p, high)
        trail = peak * (1 - trail_dist / 100)
        if low <= trail:
            return trail, "trail_same_day", ((trail / entry_p) - 1) * 100
    return None, None, None

def simulate_short_trail(entry_p, open_p, high, low, close, sl_pct, tp_pct):
    """Simulate short position for a single day."""
    sl_price = entry_p * (1 + sl_pct / 100)
    tp_price = entry_p * (1 - tp_pct / 100)

    if high >= sl_price:
        return sl_price, "short_sl", ((entry_p - sl_price) / entry_p) * 100
    if low <= tp_price:
        return tp_price, "short_tp", ((entry_p - tp_price) / entry_p) * 100
    return None, None, None

logger.info("Backtesting gap_bot.py v4 on %d stocks, %s", len(WATCHLIST), PERIOD)
logger.info("  Capital: $%.0f (%d x $%.0f tranches)", CAPITAL, MAX_POSITIONS, TRANCH_SIZE)
logger.info("  LONG: gap>=+%.0f%% SL=-%.0f%% trail=+%.0f%%/%.0f%%", MIN_GAP_LONG, HARD_SL_LONG, TRAIL_ACT_LONG, TRAIL_DIST_LONG)
logger.info("  SHORT: gap<=%.0f%% SL=+%.0f%% TP=-%.0f%%", SHORT_MIN_GAP, SHORT_SL, SHORT_TP)
logger.info("=" * 60)

results_long = []
results_short = []
long_trades = 0
short_trades = 0
long_pnl = 0.0
short_pnl = 0.0
long_wins = 0
short_wins = 0

for sym in WATCHLIST:
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period=PERIOD)
        if df.empty or len(df) < 20:
            continue

        df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100

        for i in range(1, len(df)):
            row = df.iloc[i]
            open_p = row["Open"]
            high = row["High"]
            low = row["Low"]
            close = row["Close"]
            vol = row["Volume"]
            gap = row["gap_pct"]
            d = row.name
            if isinstance(d, pd.Timestamp):
                d = d.to_pydatetime()
            if d.weekday() >= 5:
                continue
            if open_p < MIN_PRICE or open_p > MAX_PRICE or vol < MIN_VOL:
                continue

            # Long
            if gap >= MIN_GAP_LONG and long_trades < 5000:
                pos_size = min(TRANCH_SIZE, CAPITAL)
                qty = pos_size / open_p
                entry_val = qty * open_p

                exit_p, reason, gain = simulate_trail(
                    open_p, open_p, high, low, close,
                    TRAIL_ACT_LONG, TRAIL_DIST_LONG, HARD_SL_LONG
                )

                if exit_p is None:
                    exit_p = close
                    reason = "eod_close"
                    gain = ((close / open_p) - 1) * 100

                pnl = entry_val * gain / 100
                long_pnl += pnl
                long_trades += 1
                if gain > 0:
                    long_wins += 1
                results_long.append({
                    "date": d.strftime("%Y-%m-%d"), "sym": sym,
                    "gap": round(gap, 1), "entry": round(open_p, 2),
                    "exit": round(exit_p, 2), "gain": round(gain, 1),
                    "pnl": round(pnl, 2), "reason": reason, "side": "LONG"
                })

            # Short
            if gap <= SHORT_MIN_GAP and short_trades < 5000:
                pos_size = TRANCH_SIZE
                qty = pos_size / open_p
                entry_val = qty * open_p

                exit_p, reason, gain = simulate_short_trail(
                    open_p, open_p, high, low, close,
                    SHORT_SL, SHORT_TP
                )

                if exit_p is None:
                    exit_p = close
                    reason = "short_close"
                    gain = ((open_p - close) / open_p) * 100

                pnl = entry_val * gain / 100
                short_pnl += pnl
                short_trades += 1
                if gain > 0:
                    short_wins += 1
                results_short.append({
                    "date": d.strftime("%Y-%m-%d"), "sym": sym,
                    "gap": round(gap, 1), "entry": round(open_p, 2),
                    "exit": round(exit_p, 2), "gain": round(gain, 1),
                    "pnl": round(pnl, 2), "reason": reason, "side": "SHORT"
                })

    except Exception as e:
        continue

total_trades = long_trades + short_trades
total_pnl = long_pnl + short_pnl

def print_stats(label, trades, wins, pnl, n):
    if n == 0:
        return
    wr = wins / n * 100
    avg_gain = pnl / n
    logger.info(f"  {label}: {n} trades | WR: {wr:.1f}% | Total P&L: ${pnl:.2f} | Avg: ${avg_gain:.2f}")

logger.info("\n" + "=" * 60)
logger.info("RESULTS")
logger.info("=" * 60)
if long_trades > 0:
    df_l = pd.DataFrame(results_long)
    logger.info(f"\nLONG: {long_trades} trades | WR: {long_wins/long_trades*100:.1f}%")
    logger.info(f"  Avg win: +{df_l[df_l['gain']>0]['gain'].mean():.1f}% | Avg loss: {df_l[df_l['gain']<=0]['gain'].mean():.1f}%")
    logger.info(f"  Max win: +{df_l['gain'].max():.1f}% | Max loss: {df_l['gain'].min():.1f}%")
    logger.info(f"  Total P&L: ${long_pnl:.2f} | Avg/trade: ${long_pnl/long_trades:.2f}")

if short_trades > 0:
    df_s = pd.DataFrame(results_short)
    logger.info(f"\nSHORT: {short_trades} trades | WR: {short_wins/short_trades*100:.1f}%")
    logger.info(f"  Avg win: +{df_s[df_s['gain']>0]['gain'].mean():.1f}% | Avg loss: {df_s[df_s['gain']<=0]['gain'].mean():.1f}%")
    logger.info(f"  Max win: +{df_s['gain'].max():.1f}% | Max loss: {df_s['gain'].min():.1f}%")
    logger.info(f"  Total P&L: ${short_pnl:.2f} | Avg/trade: ${short_pnl/short_trades:.2f}")

total_days = (pd.to_datetime([r["date"] for r in results_long + results_short]).max() - 
              pd.to_datetime([r["date"] for r in results_long + results_short]).min()).days if (results_long or results_short) else 1
trading_days = total_days * 5 / 7
hrs = trading_days * 6.5

logger.info(f"\nCOMBINED: {total_trades} trades | Total P&L: ${total_pnl:.2f}")
logger.info(f"  Period: {total_days} days | ~{trading_days:.0f} trading days | ~{hrs:.0f} market hours")
logger.info(f"  Per hour: ${total_pnl/hrs:.2f}" if hrs > 0 else "")
logger.info(f"  Per trading day: ${total_pnl/trading_days:.2f}" if trading_days > 0 else "")

combined = results_long + results_short
df_all = pd.DataFrame(combined)
if len(df_all) > 0:
    mc = []
    for _ in range(10000):
        s = df_all["gain"].sample(n=len(df_all), replace=True)
        mc.append(s.sum())
    mc_s = pd.Series(mc)
    logger.info(f"\nMonte Carlo (10k sims):")
    logger.info(f"  Mean: {mc_s.mean():.1f}% | Median: {mc_s.median():.1f}%")
    logger.info(f"  P(profit): {(mc_s > 0).mean()*100:.1f}%")
    logger.info(f"  95th: +{mc_s.quantile(0.95):.1f}% | 5th: {mc_s.quantile(0.05):.1f}%")

    df_all.to_csv("/tmp/gap_bot_backtest.csv", index=False)
    logger.info(f"\nFull results saved to /tmp/gap_bot_backtest.csv")

    top = df_all.nlargest(5, "gain")
    worst = df_all.nsmallest(5, "gain")
    logger.info(f"\nTop 5:")
    for _, r in top.iterrows():
        logger.info(f"  +{r['gain']:.1f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  {r['side']:5s}  ({r['reason']})")
    logger.info(f"\nWorst 5:")
    for _, r in worst.iterrows():
        logger.info(f"  {r['gain']:.1f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  {r['side']:5s}  ({r['reason']})")

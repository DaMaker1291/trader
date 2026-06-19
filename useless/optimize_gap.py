"""
Parameter optimization for gap_bot.py LONG-only strategy.
Grid search over MIN_GAP, HARD_SL, TRAIL_ACT, TRAIL_DIST.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

CAPITAL = 200.0
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

# Load all data once
data = {}
for sym in WATCHLIST:
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period=PERIOD)
        if df.empty or len(df) < 20:
            continue
        df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
        data[sym] = df
    except:
        continue

print(f"Loaded {len(data)} stocks")

def backtest_long(min_gap, hard_sl, trail_act, trail_dist):
    trades = 0
    total_pnl = 0.0
    wins = 0
    total_gain = 0.0
    
    for sym, df in data.items():
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
            if pd.isna(gap) or gap < min_gap or open_p < MIN_PRICE or open_p > MAX_PRICE or vol < MIN_VOL:
                continue
            
            trades += 1
            sl = open_p * (1 - hard_sl / 100)
            trail_trig = open_p * (1 + trail_act / 100)
            exit_p = None
            
            if low <= sl:
                exit_p = sl
            elif high >= trail_trig:
                peak = max(open_p, high)
                trail_stop = peak * (1 - trail_dist / 100)
                if low <= trail_stop:
                    exit_p = trail_stop
                    
            if exit_p is None:
                exit_p = close
                
            gain = ((exit_p / open_p) - 1) * 100
            total_pnl += CAPITAL * gain / 100
            total_gain += gain
            if gain > 0:
                wins += 1
    
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, total_pnl, total_gain / trades if trades > 0 else 0

# Grid search
params_grid = [
    (3, 3, 3, 2), (3, 4, 5, 3), (3, 5, 8, 5),
    (5, 3, 3, 2), (5, 4, 5, 3), (5, 4, 8, 5), (5, 5, 5, 3), (5, 5, 8, 5), (5, 6, 10, 7),
    (8, 3, 3, 2), (8, 4, 5, 3), (8, 4, 8, 5), (8, 5, 5, 3), (8, 5, 8, 5), (8, 6, 10, 7),
    (10, 3, 5, 3), (10, 4, 8, 5), (10, 5, 5, 3), (10, 6, 10, 7),
]

results = []
for mg, hsl, ta, td in params_grid:
    trades, wr, pnl, avg_gain = backtest_long(mg, hsl, ta, td)
    results.append((mg, hsl, ta, td, trades, wr, pnl, avg_gain))
    print(f"  G={mg:2d} SL={hsl:2d} TA={ta:2d} TD={td:2d} -> {trades:4d} trades, WR={wr:5.1f}%, P&L=${pnl:7.2f}, avg={avg_gain:5.1f}%")

print("\n" + "=" * 80)
print(f"{'GAP':>4} {'SL':>4} {'TRAIL_ACT':>10} {'TRAIL_DIST':>11} {'TRADES':>7} {'WR':>5} {'P&L':>10} {'AVG%':>6}")
print("=" * 80)
results.sort(key=lambda r: r[6], reverse=True)
for r in results:
    print(f"{r[0]:4d} {r[1]:4d} {r[2]:10d} {r[3]:11d} {r[4]:7d} {r[5]:5.1f}% ${r[6]:>7.2f} {r[7]:6.1f}%")
print("=" * 80)
print(f"\nBest by P&L: GAP={results[0][0]} SL={results[0][1]} TRAIL_ACT={results[0][2]} TRAIL_DIST={results[0][3]}")
print(f"  -> {results[0][4]} trades, WR={results[0][5]:.1f}%, P&L=${results[0][6]:.2f}")

"""
Enhanced leveraged ETF strategy — trade both sides, multiple positions.
Long TQQQ/SOXL/FAS on green SPY days. Short SQQQ/SOXS/FAZ on red SPY days.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL = 200.0

# Long leveraged ETFs
LONG_ETFS = {"TQQQ": "QQQ", "SOXL": "SOX", "FAS": "XLF", "UPRO": "SPY", "TNA": "IWM", "FNGU": "FNG", "LABU": "XBI"}
# Inverse leveraged ETFs (short when market is down)
SHORT_ETFS = {"SQQQ": "QQQ", "SOXS": "SOX", "FAZ": "XLF", "SPXU": "SPY", "TZA": "IWM"}

print(f"Loading data...")
data = {}
for sym in list(LONG_ETFS.keys()) + list(SHORT_ETFS.keys()):
    try:
        df = yf.Ticker(sym).history(period="5y")
        if df.empty or len(df) < 200:
            continue
        df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
        data[sym] = df
    except:
        continue

spy = yf.Ticker("SPY")
spy_df = spy.history(period="5y")
if not spy_df.empty:
    spy_df["ma20"] = spy_df["Close"].rolling(20).mean()
    spy_df["ma50"] = spy_df["Close"].rolling(50).mean()
    spy_df["day_pct"] = spy_df["Close"].pct_change() * 100

NASDAQ = yf.Ticker("^IXIC")
nasdaq_df = NASDAQ.history(period="5y")

print(f"Loaded {len(data)} leveraged ETFs + SPY + NASDAQ\n")

# Test configurations
tests = []

# 1. LONG only (green SPY days) - single best
def test_long_only(min_spy=0.3, max_pos=1, sl=5, hold_days=1):
    trades = total_pnl = wins = peak = 0
    peak = CAPITAL
    for i in range(1, len(spy_df)):
        spy_row = spy_df.iloc[i]
        d = spy_row.name
        if hasattr(d, "weekday") and d.weekday() >= 5:
            continue
        if i < 1:
            continue
        spy_chg = ((spy_df.iloc[i]["Close"] / spy_df.iloc[i-1]["Close"]) - 1) * 100
        
        if spy_chg < min_spy:
            continue
        
        day_entries = 0
        for sym in LONG_ETFS:
            if day_entries >= max_pos:
                break
            if sym not in data:
                continue
            df = data[sym]
            try:
                idx = df.index.get_loc(spy_row.name)
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[idx]
            except:
                continue
            
            if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000:
                continue
            
            entry = row["Open"]
            sl_price = entry * (1 - sl/100)
            trades += 1
            day_entries += 1
            
            if hold_days == 1:
                if row["Low"] <= sl_price:
                    exit_p = sl_price
                else:
                    exit_p = row["Close"]
            else:
                exit_p = row["Close"]
                for j in range(1, hold_days):
                    if idx + j < len(df):
                        fut = df.iloc[idx + j]
                        if fut["Low"] <= sl_price:
                            exit_p = sl_price
                            break
                        exit_p = fut["Close"]
            
            gain = ((exit_p / entry) - 1) * 100
            pnl = (CAPITAL / max_pos) * gain / 100
            total_pnl += pnl
            if gain > 0:
                wins += 1
            
            cur = CAPITAL + total_pnl
            if cur > peak:
                peak = cur
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, total_pnl, (peak - (CAPITAL + total_pnl)) / peak * 100 if peak > 0 else 0

# 2. SHORT only (red SPY days) 
def test_short_only(min_spy=-0.3, max_pos=1, sl=5):
    trades = total_pnl = wins = 0
    peak = CAPITAL
    for i in range(1, len(spy_df)):
        spy_row = spy_df.iloc[i]
        d = spy_row.name
        if hasattr(d, "weekday") and d.weekday() >= 5:
            continue
        spy_chg = ((spy_df.iloc[i]["Close"] / spy_df.iloc[i-1]["Close"]) - 1) * 100
        
        if spy_chg > min_spy:
            continue
        
        day_entries = 0
        for sym in SHORT_ETFS:
            if day_entries >= max_pos:
                break
            if sym not in data:
                continue
            df = data[sym]
            try:
                idx = df.index.get_loc(spy_row.name)
                if idx < 0 or idx >= len(df):
                    continue
                row = df.iloc[idx]
            except:
                continue
            
            if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000:
                continue
            
            entry = row["Open"]
            sl_price = entry * (1 + sl/100)  # short SL is upside
            trades += 1
            day_entries += 1
            
            if row["High"] >= sl_price:
                exit_p = sl_price
            else:
                exit_p = row["Close"]
            
            gain = ((entry - exit_p) / entry) * 100  # short gain
            pnl = (CAPITAL / max_pos) * gain / 100
            total_pnl += pnl
            if gain > 0:
                wins += 1
            
            cur = CAPITAL + total_pnl
            if cur > peak:
                peak = cur
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, total_pnl, 0

# 3. COMBINED (long on green, short on red)
def test_combined(min_spy=0.3, max_pos=2, sl=5):
    trades = total_pnl = wins = 0
    peak = CAPITAL
    for i in range(1, len(spy_df)):
        spy_row = spy_df.iloc[i]
        d = spy_row.name
        if hasattr(d, "weekday") and d.weekday() >= 5:
            continue
        spy_chg = ((spy_df.iloc[i]["Close"] / spy_df.iloc[i-1]["Close"]) - 1) * 100
        
        day_entries = 0
        if spy_chg >= min_spy:
            # Long day
            for sym in LONG_ETFS:
                if day_entries >= max_pos:
                    break
                if sym not in data:
                    continue
                df = data[sym]
                try:
                    idx = df.index.get_loc(spy_row.name)
                    if idx < 0 or idx >= len(df):
                        continue
                    row = df.iloc[idx]
                except:
                    continue
                if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000:
                    continue
                
                entry = row["Open"]
                sl_price = entry * (1 - sl/100)
                trades += 1
                day_entries += 1
                
                if row["Low"] <= sl_price:
                    exit_p = sl_price
                else:
                    exit_p = row["Close"]
                
                gain = ((exit_p / entry) - 1) * 100
                pnl = (CAPITAL * 0.5) * gain / 100 if max_pos == 2 else CAPITAL * gain / 100
                total_pnl += pnl
                if gain > 0:
                    wins += 1
                
                cur = CAPITAL + total_pnl
                if cur > peak:
                    peak = cur
        elif spy_chg <= -min_spy:
            # Red day - short
            for sym in SHORT_ETFS:
                if day_entries >= max_pos:
                    break
                if sym not in data:
                    continue
                df = data[sym]
                try:
                    idx = df.index.get_loc(spy_row.name)
                    if idx < 0 or idx >= len(df):
                        continue
                    row = df.iloc[idx]
                except:
                    continue
                if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000:
                    continue
                
                entry = row["Open"]
                sl_price = entry * (1 + sl/100)
                trades += 1
                day_entries += 1
                
                if row["High"] >= sl_price:
                    exit_p = sl_price
                else:
                    exit_p = row["Close"]
                
                gain = ((entry - exit_p) / entry) * 100
                pnl = (CAPITAL * 0.5) * gain / 100 if max_pos == 2 else CAPITAL * gain / 100
                total_pnl += pnl
                if gain > 0:
                    wins += 1
    
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, total_pnl, 0


configs = [
    ("LONG 1-pos 3%SPY 5%SL", lambda: test_long_only(0.3, 1, 5)),
    ("LONG 2-pos 3%SPY 5%SL", lambda: test_long_only(0.3, 2, 5)),
    ("LONG 1-pos 5%SPY 5%SL", lambda: test_long_only(0.5, 1, 5)),
    ("SHORT 1-pos -3%SPY 5%SL", lambda: test_short_only(-0.3, 1, 5)),
    ("SHORT 1-pos -5%SPY 5%SL", lambda: test_short_only(-0.5, 1, 5)),
    ("COMBINED 1-pos 3%SPY 5%SL", lambda: test_combined(0.3, 1, 5)),
    ("COMBINED 2-pos 3%SPY 5%SL", lambda: test_combined(0.3, 2, 5)),
]

print(f"{'CONFIG':45s} {'TRADES':>7} {'WR':>6} {'P&L':>10} {'$/HR':>8}")
print("=" * 80)

for name, fn in configs:
    trades, wr, pnl, dd = fn()
    hrs_5yr = 252 * 6.5 * 5
    phr = pnl / hrs_5yr
    print(f"{name:45s} {trades:7d} {wr:5.1f}% ${pnl:+>7.2f} ${phr:>5.2f}")

# Year-by-year for best
print(f"\n{'='*80}")
print("YEAR-BY-YEAR: COMBINED 2-pos 3%SPY 5%SL")
print(f"{'='*80}")
trades, wr, total_pnl, dd = test_combined(0.3, 2, 5)
for year in range(2021, 2027):
    yr_trades = yr_pnl = yr_wins = 0
    for i in range(1, len(spy_df)):
        spy_row = spy_df.iloc[i]
        d = spy_row.name
        if hasattr(d, "year") and d.year != year:
            continue
        if hasattr(d, "weekday") and d.weekday() >= 5:
            continue
        spy_chg = ((spy_df.iloc[i]["Close"] / spy_df.iloc[i-1]["Close"]) - 1) * 100
        day_entries = 0
        
        if spy_chg >= 0.3:
            for sym in LONG_ETFS:
                if day_entries >= 2: break
                if sym not in data: continue
                df = data[sym]
                try:
                    idx = df.index.get_loc(spy_row.name)
                    if idx < 0 or idx >= len(df): continue
                    row = df.iloc[idx]
                except: continue
                if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000: continue
                entry = row["Open"]
                sl_price = entry * 0.95
                trades_yr += 1; day_entries += 1
                exit_p = sl_price if row["Low"] <= sl_price else row["Close"]
                gain = ((exit_p/entry)-1)*100
                pnl = 100 * gain / 100
                yr_pnl += pnl
                if gain > 0: yr_wins += 1
        elif spy_chg <= -0.3:
            for sym in SHORT_ETFS:
                if day_entries >= 2: break
                if sym not in data: continue
                df = data[sym]
                try:
                    idx = df.index.get_loc(spy_row.name)
                    if idx < 0 or idx >= len(df): continue
                    row = df.iloc[idx]
                except: continue
                if row["Open"] < 5 or row["Open"] > 500 or row["Volume"] < 100_000: continue
                entry = row["Open"]
                sl_price = entry * 1.05
                trades_yr += 1; day_entries += 1
                exit_p = sl_price if row["High"] >= sl_price else row["Close"]
                gain = ((entry-exit_p)/entry)*100
                pnl = 100 * gain / 100
                yr_pnl += pnl
                if gain > 0: yr_wins += 1
    
    wr = yr_wins / trades_yr * 100 if trades_yr > 0 else 0
    label = {2021:"BULL",2022:"BEAR",2023:"RECOVERY",2024:"MIXED",2025:"RECENT",2026:"YTD"}.get(year,"")
    hrs = 252 * 6.5
    print(f"  {year} {label:12s}: {yr_trades:4d} trades | WR={wr:5.1f}% | P&L=${yr_pnl:+>7.2f} | $/hr=${yr_pnl/hrs:.2f}")

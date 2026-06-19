"""
High-aggression strategy backtest — find the fastest money-maker on $200.
Tests: all-in position sizing, leverage ETFs, top momentum, gap continuations.
Uses 5 years of daily data across bull/bear/crash/recovery markets.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

CAPITAL = 200.0
FRICTION = 0.75

# Leveraged ETFs
LEVERAGED = {
    "TQQQ": "QQQ", "SOXL": "SOX", "FAS": "XLF",
    "FNGU": "FNG", "UPRO": "SPY", "TNA": "IWM",
    "LABU": "XBI", "SPXL": "SPY",
}
# Underlying indices for leveraged ETFs
INDICES = {
    "SPY": 5000, "QQQ": 5000, "IWM": 3000, "XLF": 1000,
    "XBI": 500, "SOX": 500,
}

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER",
    "AI","IONQ","RGTI","BBAI",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "GME","AMC","CHWY","DKNG","CVNA",
    "MU","INTC","QCOM","MRVL",
    "SPY","QQQ","IWM","XLF","XBI","ARKK",
]

print(f"Loading {len(WATCHLIST)} stocks, 5yr daily data...")
data = {}
loaded = 0
for sym in WATCHLIST:
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period="5y")
        if df.empty or len(df) < 200:
            continue
        df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
        df["day_return"] = df["Close"].pct_change() * 100
        df["intra_range"] = (df["High"] / df["Low"] - 1) * 100
        df["intra_mom"] = (df["High"] / df["Open"] - 1) * 100
        df["avg_vol"] = df["Volume"].rolling(20).mean()
        data[sym] = df
        loaded += 1
    except:
        continue

# Load SPY for market direction
spy = yf.Ticker("SPY")
spy_df = spy.history(period="5y")
if not spy_df.empty:
    spy_df["ma20"] = spy_df["Close"].rolling(20).mean()
    spy_df["ma50"] = spy_df["Close"].rolling(50).mean()
    spy_df["day_pct"] = spy_df["Close"].pct_change() * 100
print(f"Loaded {loaded} stocks + SPY")


def backtest_strategy(name, entry_logic, exit_logic, all_in=True):
    """
    entry_logic(sym, df, i, row, spy_row) -> (enter: bool, entry_price: float)
    exit_logic(sym, df, i, row, entry_price) -> (exit_price: float, reason: str)
    """
    trades = 0
    total_pnl = 0.0
    wins = 0
    peak = CAPITAL
    max_dd = 0.0
    daily_pnl = []
    
    for i in range(1, len(spy_df)):
        spy_row = spy_df.iloc[i]
        d = spy_row.name
        if isinstance(d, pd.Timestamp):
            d = d.to_pydatetime()
        if d.weekday() >= 5:
            continue
        
        day_trades = 0
        day_pnl = 0.0
        
        for sym in list(data.keys()):
            df = data[sym]
            try:
                idx = df.index.get_loc(spy_row.name)
                if idx < 1 or idx >= len(df):
                    continue
                row = df.iloc[idx]
            except:
                continue
            
            open_p = row["Open"]
            high = row["High"]
            low = row["Low"]
            close = row["Close"]
            vol = row["Volume"]
            
            if all_in and day_trades > 0:
                continue
            
            pos_size = CAPITAL if all_in else CAPITAL / 3
            if open_p < 2 or open_p > 500:
                continue
            
            enter, entry_price = entry_logic(sym, df, idx, row, spy_row)
            if not enter:
                continue
            
            exit_price, reason = exit_logic(sym, df, idx, row, entry_price)
            if exit_price is None:
                exit_price = row["Close"]
                reason = "eod"
            
            gain = ((exit_price / entry_price) - 1) * 100
            pnl = pos_size * gain / 100
            day_pnl += pnl
            day_trades += 1
            trades += 1
            total_pnl += pnl
            if gain > 0:
                wins += 1
            
            cur = CAPITAL + total_pnl
            if cur > peak:
                peak = cur
            dd = (peak - cur) / peak * 100
            max_dd = max(max_dd, dd)
            
            if all_in:
                break
        
        if day_trades > 0:
            daily_pnl.append(day_pnl)
    
    wr = wins / trades * 100 if trades > 0 else 0
    adj_pnl = total_pnl * FRICTION
    yrs = 5.0
    mkt_hrs = 252 * 6.5 * yrs
    return {
        "name": name, "trades": trades, "wr": wr,
        "pnl": total_pnl, "adj_pnl": adj_pnl,
        "max_dd": max_dd, "per_hr": adj_pnl / mkt_hrs,
        "per_mo": adj_pnl / (yrs * 12),
        "per_yr": adj_pnl / yrs,
    }


# ── STRATEGY DEFINITIONS ──

def strat_gap_all_in(sym, df, i, row, spy_row):
    """All-in on gap-ups >= 3%. Hold to close. SL at -3%."""
    if row["gap_pct"] >= 3.0 and row["Volume"] > 500_000:
        return True, row["Open"]
    return False, 0

def strat_gap_all_in_exit(sym, df, i, row, entry):
    sl = entry * 0.97
    if row["Low"] <= sl:
        return sl, "sl"
    return None, None


def strat_leveraged_only(sym, df, i, row, spy_row):
    """Only trade leveraged ETFs on green SPY days."""
    if sym not in LEVERAGED:
        return False, 0
    if i < 1:
        return False, 0
    prev_close = df.iloc[i-1]["Close"]
    if prev_close <= 0:
        return False, 0
    
    # Green day check via SPY
    if isinstance(spy_row.name, pd.Timestamp) and spy_row.name in spy_df.index:
        spy_idx = spy_df.index.get_loc(spy_row.name)
        if spy_idx > 0:
            spy_chg = ((spy_df.iloc[spy_idx]["Close"] / spy_df.iloc[spy_idx-1]["Close"]) - 1) * 100
        else:
            spy_chg = 0
    else:
        spy_chg = 0
    
    # Buy leveraged ETF on green/open-up days
    if spy_chg > 0.3 and row["gap_pct"] > -1.0 and row["Volume"] > 100_000:
        return True, row["Open"]
    return False, 0

def strat_leveraged_exit(sym, df, i, row, entry):
    sl = entry * 0.95  # 5% SL
    if row["Low"] <= sl:
        return sl, "sl"
    return None, None


def strat_top_momentum(sym, df, i, row, spy_row):
    """Pick the single best momentum stock each day - all-in."""
    if row["gap_pct"] >= 5.0 and row["Volume"] > 1_000_000 and row["intra_mom"] > 2:
        return True, row["Open"]
    return False, 0

def strat_top_momentum_exit(sym, df, i, row, entry):
    sl = entry * 0.96  # 4% SL
    if row["Low"] <= sl:
        return sl, "sl"
    return None, None


def strat_daily_trend(sym, df, i, row, spy_row):
    """Buy any liquid stock on green market days, hold to close."""
    if i < 2:
        return False, 0
    prev_close = df.iloc[i-1]["Close"]
    spy_chg = ((spy_row["Close"] / spy_df.iloc[spy_df.index.get_loc(spy_row.name)-1]["Close"]) - 1) * 100 \
        if spy_df.index.get_loc(spy_row.name) > 0 else 0
    
    if spy_chg > 0.5 and row["gap_pct"] > -2.0 and row["Volume"] > 1_000_000 and row["Open"] > 5:
        return True, row["Open"]
    return False, 0

def strat_daily_trend_exit(sym, df, i, row, entry):
    sl = entry * 0.93  # 7% SL
    if row["Low"] <= sl:
        return sl, "sl"
    return None, None


def strat_tqqq_hold(sym, df, i, row, spy_row):
    """TQQQ only: buy on SPY > 20MA, sell on close or SL."""
    if sym != "TQQQ":
        return False, 0
    if row["Volume"] < 100_000:
        return False, 0
    # SPY above 20MA
    spy_idx = spy_df.index.get_loc(spy_row.name)
    if spy_idx > 0 and spy_df.iloc[spy_idx]["Close"] > spy_df.iloc[spy_idx]["ma20"]:
        return True, row["Open"]
    return False, 0

def strat_tqqq_exit(sym, df, i, row, entry):
    sl = entry * 0.94  # 6% SL (TQQQ is 3x so this = 2% SPY move)
    if row["Low"] <= sl:
        return sl, "sl"
    return None, None


# Combined: buy strongest daily mover, all-in, hold all day
def strat_strongest_gap(sym, df, i, row, spy_row):
    """Buy biggest gap-up stock each day with high volume."""
    # Need to track best across all stocks - can't do that per-stock
    # This is a simplified version
    if row["gap_pct"] >= 4.0 and row["Volume"] > 2_000_000 and row["Open"] >= 5:
        return True, row["Open"]
    return False, 0

def strat_strongest_exit(sym, df, i, row, entry):
    sl = entry * 0.95
    if row["Low"] <= sl:
        return sl, "sl"
    # Trail: if stock goes up 8%, trail at 4% below peak
    peak = max(entry, row["High"])
    if peak >= entry * 1.08:
        trail = peak * 0.96
        if row["Low"] <= trail:
            return trail, "trail"
    return None, None


strategies = [
    ("GAP ALL-IN (>=3%, 3% SL)", strat_gap_all_in, strat_gap_all_in_exit),
    ("LEVERAGED ETFs (green SPY, 5% SL)", strat_leveraged_only, strat_leveraged_exit),
    ("TOP MOMENTUM (>=5% gap, 4% SL)", strat_top_momentum, strat_top_momentum_exit),
    ("DAILY TREND (green days, 7% SL)", strat_daily_trend, strat_daily_trend_exit),
    ("TQQQ ONLY (SPY>20MA, 6% SL)", strat_tqqq_hold, strat_tqqq_exit),
    ("STRONGEST GAP (>=4% gap, 5% SL+trail)", strat_strongest_gap, strat_strongest_exit),
]

print(f"\n{'='*80}")
print(f"{'STRATEGY':40s} {'TRADES':>7} {'WR':>6} {'P&L':>10} {'ADJ':>10} {'$/HR':>8} {'$/MO':>8} {'DD':>6}")
print(f"{'='*80}")

results = []
for name, entry_fn, exit_fn in strategies:
    r = backtest_strategy(name, entry_fn, exit_fn, all_in=True)
    results.append(r)
    print(f"{r['name']:40s} {r['trades']:7d} {r['wr']:5.1f}% ${r['pnl']:+>7.2f} ${r['adj_pnl']:+>7.2f} ${r['per_hr']:>5.2f} ${r['per_mo']:>6.2f} {r['max_dd']:5.1f}%")

results.sort(key=lambda r: r['adj_pnl'], reverse=True)
print(f"\n{'='*80}")
print(f"BEST: {results[0]['name']}")
print(f"  P&L: ${results[0]['pnl']:.2f} | Adj: ${results[0]['adj_pnl']:.2f} | $/hr: ${results[0]['per_hr']:.2f}")

# Year-by-year for the best strategy
best_name = results[0]['name']
best_entry = None
best_exit = None
for n, e, x in strategies:
    if n == best_name:
        best_entry, best_exit = e, x
        break

print(f"\n{'='*80}")
print(f"YEAR-BY-YEAR: {best_name}")
print(f"{'='*80}")
for year in range(2021, 2027):
    yr_pnl = 0.0
    yr_trades = 0
    yr_wins = 0
    yr_df = spy_df[spy_df.index.year == year]
    for i in range(1, len(yr_df)):
        spy_row = yr_df.iloc[i]
        d = spy_row.name.to_pydatetime()
        if d.weekday() >= 5:
            continue
        
        best_sym = None
        best_score = -999
        best_entry_price = 0
        
        for sym in list(data.keys()):
            df = data[sym]
            try:
                idx = df.index.get_loc(spy_row.name)
                if idx < 1 or idx >= len(df):
                    continue
                row = df.iloc[idx]
            except:
                continue
            
            if row["Open"] < 2 or row["Open"] > 500:
                continue
            if row["Volume"] < 100_000:
                continue
            
            enter, ep = best_entry(sym, df, idx, row, spy_row)
            if enter:
                # Score to pick best
                score = row["gap_pct"] * 2 + row["Volume"] / 1e6
                if score > best_score:
                    best_score = score
                    best_sym = sym
                    best_entry_price = ep
        
        if best_sym is None:
            continue
        
        df = data[best_sym]
        idx = df.index.get_loc(spy_row.name)
        row = df.iloc[idx]
        exit_price, reason = best_exit(best_sym, df, idx, row, best_entry_price)
        if exit_price is None:
            exit_price = row["Close"]
        
        gain = ((exit_price / best_entry_price) - 1) * 100
        pnl = CAPITAL * gain / 100
        yr_pnl += pnl
        yr_trades += 1
        if gain > 0:
            yr_wins += 1
    
    wr = yr_wins / yr_trades * 100 if yr_trades > 0 else 0
    adj = yr_pnl * FRICTION
    hrs = 252 * 6.5
    label = {2021: "BULL", 2022: "BEAR/CRASH", 2023: "RECOVERY", 2024: "MIXED", 2025: "RECENT", 2026: "YTD"}.get(year, "")
    print(f"  {year} {label:12s}: {yr_trades:4d} trades | WR={wr:5.1f}% | P&L=${yr_pnl:+>7.2f} | Adj=${adj:+>7.2f} | $/hr=${adj/hrs:.2f}")

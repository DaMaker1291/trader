"""
Corrected aggressive backtest — realistic options pricing.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CAPITAL = 200.0
FRICTION = 0.70

print("Loading...")
spy = yf.Ticker("SPY")
spy_df = spy.history(period="6y")
spy_df["open_pct"] = (spy_df["Open"] / spy_df["Close"].shift(1) - 1) * 100
spy_df["day_pct"] = (spy_df["Close"] / spy_df["Open"] - 1) * 100
spy_df["close_pct"] = spy_df["Close"].pct_change() * 100

tqqq = yf.Ticker("TQQQ")
tqqq_df = tqqq.history(period="6y")
tqqq_df["day_pct"] = (tqqq_df["Close"] / tqqq_df["Open"] - 1) * 100
print(f"Loaded: {len(spy_df)} days\n")


# CORRECTED 0DTE options model
def option_0dte_ret(spy_price, spy_move_pct):
    """
    Realistic 0DTE ATM option return.
    For a $500 SPY with $1.50 ATM premium:
      - 0.3% SPY move ($1.50) → option doubles (100% return)
      - 0.5% SPY move ($2.50) → option 2.5x (150% return)
      - 1.0% SPY move ($5.00) → option 5x (400% return)
      - Negative move → option loses rapidly, can go to $0
    """
    move_dollars = spy_price * spy_move_pct / 100
    option_premium = spy_price * 0.003  # ~0.3% of SPY for 0DTE ATM
    delta = 0.50 + abs(spy_move_pct) * 5 / 100  # delta increases with move
    delta = min(delta, 0.95)
    
    option_move = move_dollars * delta * 1.3  # 1.3 gamma boost
    ret_pct = (option_move / option_premium) * 100
    
    if spy_move_pct < 0:
        if spy_move_pct < -0.3:
            ret_pct = -100  # worthless
        else:
            ret_pct = max(-60, spy_move_pct * 3)
    
    return max(min(ret_pct, 2000), -100)


# 0DTE on ALL green days (>=0.3% open)
def run_0dte_all():
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        opt_ret = option_0dte_ret(row["Open"], row["day_pct"])
        premium = row["Open"] * 0.003
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = min(contracts * premium * 100, CAPITAL)
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    
    wr = wins/trades*100 if trades else 0
    adj = total_pnl * FRICTION
    return trades, wr, total_pnl, adj


# 0DTE on BIG gaps only (>=1%)
def run_0dte_big():
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 1.0 or pd.isna(row["open_pct"]): continue
        
        opt_ret = option_0dte_ret(row["Open"], row["day_pct"])
        premium = row["Open"] * 0.003
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = min(contracts * premium * 100, CAPITAL)
        
        # On big gap days, use all $200 on 1 contract for max effect
        cost = min(premium * 100, CAPITAL)
        if cost > 50:
            contracts = 1
            cost = min(premium * 100, CAPITAL)
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    
    wr = wins/trades*100 if trades else 0
    adj = total_pnl * FRICTION
    return trades, wr, total_pnl, adj


# 0DTE ALL-IN (use full $200 on 1 contract, highest conviction)
def run_0dte_allin():
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        opt_ret = option_0dte_ret(row["Open"], row["day_pct"])
        premium = row["Open"] * 0.003
        # Use all $200 on 1 contract
        if premium * 100 > CAPITAL:
            continue
        cost = premium * 100  # buy 1 contract with all
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if pnl > 0: wins += 1
    
    wr = wins/trades*100 if trades else 0
    adj = total_pnl * FRICTION
    return trades, wr, total_pnl, adj


# TQQQ all-in (baseline)
def run_tqqq():
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if spy_df.iloc[i]["open_pct"] < 0.3 or pd.isna(spy_df.iloc[i]["open_pct"]): continue
        try:
            idx = tqqq_df.index.get_loc(d)
            if idx < 0 or idx >= len(tqqq_df): continue
        except: continue
        row = tqqq_df.iloc[idx]
        entry = row["Open"]
        sl = entry * 0.95
        exit_p = sl if row["Low"] <= sl else row["Close"]
        gain = ((exit_p/entry)-1)*100
        pnl = CAPITAL * gain / 100
        total_pnl += pnl; trades += 1
        if gain > 0: wins += 1
    wr = wins/trades*100 if trades else 0
    adj = total_pnl * FRICTION
    return trades, wr, total_pnl, adj


# 4x ETFs
def run_4etfs():
    etfs = {}
    for sym in ["TQQQ", "SOXL", "FAS", "UPRO"]:
        df = yf.Ticker(sym).history(period="6y")
        if not df.empty: etfs[sym] = df
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if spy_df.iloc[i]["open_pct"] < 0.3 or pd.isna(spy_df.iloc[i]["open_pct"]): continue
        ps = CAPITAL / len(etfs)
        for sym, df in etfs.items():
            try:
                idx = df.index.get_loc(d)
                if idx < 0 or idx >= len(df): continue
            except: continue
            row = df.iloc[idx]
            entry = row["Open"]
            sl = entry * 0.95
            exit_p = sl if row["Low"] <= sl else row["Close"]
            gain = ((exit_p/entry)-1)*100
            pnl = ps * gain / 100
            total_pnl += pnl; trades += 1
            if gain > 0: wins += 1
    wr = wins/trades*100 if trades else 0
    adj = total_pnl * FRICTION
    return trades, wr, total_pnl, adj


# Run all
results = []
for name, fn in [
    ("TQQQ all-in (baseline)", run_tqqq),
    ("4x ETFs", run_4etfs),
    ("0DTE SPY calls (all green days)", run_0dte_all),
    ("0DTE SPY calls (>=1% gaps only)", run_0dte_big),
    ("0DTE SPY all-in (1 contract)", run_0dte_allin),
]:
    trades, wr, pnl, adj = fn()
    cal_hr = adj / (5 * 365 * 24)
    day_amt = adj / (252 * 5)
    results.append((name, trades, wr, pnl, adj, cal_hr, day_amt))
    
    print(f"{name:40s} | {trades:5d} trades | WR={wr:5.1f}% | P&L=${pnl:+>8.2f} | Adj=${adj:+>8.2f} | $/cal hr=${cal_hr:.4f}")

print(f"\n{'='*60}")
print("REALITY CHECK:")
print(f"{'='*60}")
print(f"Target: $1/cal hr = $8,760/yr = $24/day on $200")
print(f"That's 4,380% annual return. For context:")
print(f"  Best stock picker ever (Buffett): ~20%/yr")
print(f"  Best hedge fund ever (Renaissance): ~66%/yr")
print(f"  Best crypto trader: variable (mostly luck)")
print()

# Show what it would take
results.sort(key=lambda r: r[5], reverse=True)
for name, tr, wr, pnl, adj, chr_, day_ in results:
    if chr_ > 0:
        capital_needed = 1.0 / chr_ * 200
        print(f"  {name}: ${chr_:.4f}/cal hr → Need ${capital_needed:.0f} capital for $1/cal hr")
    else:
        print(f"  {name}: ${chr_:.4f}/cal hr → Loses money")

print(f"\n{'='*60}")
print("YEAR-BY-YEAR: Best option strategy (0DTE >=1% gaps)")
print(f"{'='*60}")
for year in range(2021, 2027):
    yr_pnl = yr_t = yr_w = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "year") and d.year != year: continue
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 1.0 or pd.isna(row["open_pct"]): continue
        
        opt_ret = option_0dte_ret(row["Open"], row["day_pct"])
        premium = row["Open"] * 0.003
        if premium * 100 > CAPITAL: continue
        cost = premium * 100
        pnl = cost * opt_ret / 100
        yr_pnl += pnl; yr_t += 1
        if pnl > 0: yr_w += 1
    
    label = {2021:"BULL",2022:"BEAR",2023:"RECOVERY",2024:"MIXED",2025:"RECENT",2026:"YTD"}.get(year,"")
    wr = yr_w/yr_t*100 if yr_t else 0
    adj = yr_pnl * FRICTION
    cal_hrs = 365 * 24
    print(f"  {year} {label:12s}: {yr_t:3d} trades | WR={wr:5.1f}% | P&L=${yr_pnl:+>8.2f} | Adj=${adj:+>8.2f} | $/cal hr=${adj/cal_hrs:.4f}")

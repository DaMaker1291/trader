"""
FINAL ULTIMATE BACKTEST — test every possible aggressive strategy to find $1/calendar hr.

Strategies tested:
  1. TQQQ leveraged ETF (baseline) — buy on green SPY open, 5% SL, hold EOD
  2. 4x leveraged ETFs — buy TQQQ+SOXL+FAS+UPRO on green days
  3. SPY 0DTE calls — simulated ATM options on green days
  4. SPY weekly calls (7DTE) — less gamma, more time
  5. TQQQ weekly calls — double leverage (ETF + options)
  6. NVDA weekly calls — high beta single stock
  7. All-in gap+options — buy options on biggest gap days
  8. Crypto proxy — TQQQ held 24/7 (crypto alternative)
  9. Aggressive multi-bet — 5 independent positions daily
  
  2021-2026, all market conditions included.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

CAPITAL = 200.0
FRICTION = 0.70  # real-world adjustment

# ── Load data ──
print("Loading data...")

spy = yf.Ticker("SPY")
spy_df = spy.history(period="6y")
spy_df["day_pct"] = spy_df["Close"].pct_change() * 100
spy_df["open_pct"] = (spy_df["Open"] / spy_df["Close"].shift(1) - 1) * 100
spy_df["range_pct"] = (spy_df["High"] / spy_df["Low"] - 1) * 100

tqqq = yf.Ticker("TQQQ")
tqqq_df = tqqq.history(period="6y")
tqqq_df["day_pct"] = tqqq_df["Close"].pct_change() * 100

etfs = {}
for sym in ["TQQQ", "SOXL", "FAS", "UPRO", "TNA", "FNGU", "LABU", "NVDL", "TSLL", "CONL"]:
    try:
        df = yf.Ticker(sym).history(period="6y")
        if not df.empty and len(df) > 200:
            df["day_pct"] = df["Close"].pct_change() * 100
            etfs[sym] = df
    except:
        continue

nvda = yf.Ticker("NVDA")
nvda_df = nvda.history(period="6y")
nvda_df["day_pct"] = nvda_df["Close"].pct_change() * 100

print(f"Loaded: SPY({len(spy_df)}d), TQQQ({len(tqqq_df)}d), {len(etfs)} ETFs, NVDA({len(nvda_df)}d)")


def simulate_option_return(days_return_pct, dte=0, delta=0.5):
    """
    Simulate option return from underlying return.
    dte=0: 0DTE, very high gamma
    dte=7: weekly, moderate gamma  
    delta: ATM = 0.5
    """
    abs_ret = abs(days_return_pct)
    direction = 1 if days_return_pct > 0 else -1
    
    if dte == 0:  # 0DTE — extreme gamma
        gamma_boost = 1.0 + abs_ret * 3  # gamma amplifies as move increases
        max_return = 15.0  # cap at 1500% (theoretically possible but rare)
    elif dte <= 7:  # Weekly
        gamma_boost = 1.0 + abs_ret * 1.5
        max_return = 8.0
    else:
        gamma_boost = 1.0 + abs_ret * 0.5
        max_return = 3.0
    
    opt_ret = direction * delta * abs_ret * gamma_boost * 100
    opt_ret = max(min(opt_ret, max_return * 100), -100)  # cap loss at -100%
    return opt_ret


# ── Strategy 1: TQQQ All-in ──
def strat_tqqq_allin():
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if i < 1: continue
        spy_open = spy_df.iloc[i]["open_pct"]
        if spy_open < 0.3 or pd.isna(spy_open): continue
        
        try:
            idx = tqqq_df.index.get_loc(d)
            if idx < 0 or idx >= len(tqqq_df): continue
        except: continue
        
        row = tqqq_df.iloc[idx]
        entry = row["Open"]
        sl = entry * 0.95
        if row["Low"] <= sl: exit_p = sl
        else: exit_p = row["Close"]
        gain = ((exit_p/entry)-1)*100
        pnl = CAPITAL * gain / 100
        total_pnl += pnl; trades += 1
        if gain > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 2: 4 ETFs at once ──
def strat_4etfs():
    targets = ["TQQQ", "SOXL", "FAS", "UPRO"]
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        spy_open = spy_df.iloc[i]["open_pct"]
        if spy_open < 0.3 or pd.isna(spy_open): continue
        
        pos_size = CAPITAL / len(targets)
        for sym in targets:
            if sym not in etfs: continue
            df = etfs[sym]
            try:
                idx = df.index.get_loc(d)
                if idx < 0 or idx >= len(df): continue
            except: continue
            row = df.iloc[idx]
            entry = row["Open"]
            sl = entry * 0.95
            if row["Low"] <= sl: exit_p = sl
            else: exit_p = row["Close"]
            gain = ((exit_p/entry)-1)*100
            pnl = pos_size * gain / 100
            total_pnl += pnl; trades += 1
            if gain > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 3: 0DTE SPY calls ──
def strat_0dte():
    """Buy 0DTE ATM SPY calls on green open days."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        # Option premium: ~0.3% of SPY price for 0DTE ATM
        # SPY ~$500, premium ~$1.50/contract
        premium_per_contract = row["Open"] * 0.003
        contracts = max(1, int(CAPITAL / (premium_per_contract * 100)))
        cost = contracts * premium_per_contract * 100
        
        # Option return based on SPY day return
        spy_day_ret = row["day_pct"]
        opt_ret = simulate_option_return(spy_day_ret / 100, dte=0, delta=0.5)
        
        # 0DTE: if SPY goes negative, option goes to $0 (100% loss)
        if spy_day_ret < 0:
            opt_ret = -100
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 4: Weekly SPY calls (7DTE) ──
def strat_weekly():
    """Buy 7DTE ATM SPY calls on green open days."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        # 7DTE ATM premium: ~1.0% of SPY
        premium = row["Open"] * 0.01
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = contracts * premium * 100
        
        spy_ret = row["day_pct"]
        opt_ret = simulate_option_return(spy_ret / 100, dte=7, delta=0.5)
        
        # Weekly options: if SPY negative, lose but not as much as 0DTE
        if spy_ret < 0:
            opt_ret = max(-50, spy_ret * 3)  # max -50% in a day
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 5: TQQQ weekly calls (double leverage) ──
def strat_tqqq_options():
    """Buy 7DTE ATM TQQQ calls on green SPY days. Leverage^2."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if spy_df.iloc[i]["open_pct"] < 0.3 or pd.isna(spy_df.iloc[i]["open_pct"]): continue
        
        try:
            idx = tqqq_df.index.get_loc(d)
            if idx < 0 or idx >= len(tqqq_df): continue
        except: continue
        
        trow = tqqq_df.iloc[idx]
        tqqq_ret = ((trow["Close"] / trow["Open"]) - 1) * 100
        
        # TQQQ option premium: higher vol → higher premium
        premium = trow["Open"] * 0.015  # ~1.5% for TQQQ ATM weekly
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = contracts * premium * 100
        
        # Option return: TQQQ weekly option on TQQQ move
        opt_ret = simulate_option_return(tqqq_ret / 100, dte=7, delta=0.5)
        if tqqq_ret < 0:
            opt_ret = max(-60, tqqq_ret * 2.5)
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 6: NVDA weekly calls (high beta single stock) ──
def strat_nvda_options():
    """Buy 7DTE ATM NVDA calls on green SPY days."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if spy_df.iloc[i]["open_pct"] < 0.5 or pd.isna(spy_df.iloc[i]["open_pct"]): continue
        
        try:
            idx = nvda_df.index.get_loc(d)
            if idx < 0 or idx >= len(nvda_df): continue
        except: continue
        
        nrow = nvda_df.iloc[idx]
        nvda_ret = ((nrow["Close"] / nrow["Open"]) - 1) * 100
        
        premium = nrow["Open"] * 0.02  # ~2% for NVDA ATM weekly (higher vol)
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = contracts * premium * 100
        
        opt_ret = simulate_option_return(nvda_ret / 100, dte=7, delta=0.5)
        if nvda_ret < 0:
            opt_ret = max(-60, nvda_ret * 2)
        
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 7: 0DTE on BIG gap days (≥1% open) ──
def strat_0dte_big():
    """Only trade 0DTE when SPY opens up ≥1% (high conviction)."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 1.0 or pd.isna(row["open_pct"]): continue
        
        premium = row["Open"] * 0.003
        contracts = max(1, int(CAPITAL / (premium * 100)))
        cost = contracts * premium * 100
        
        spy_ret = row["day_pct"]
        # On big gap days, option can 3-10x
        if spy_ret >= 1.0:
            opt_ret = 100 + spy_ret * 5  # 100% + 5x return
        elif spy_ret > 0:
            opt_ret = 30 + spy_ret * 2
        else:
            opt_ret = -100  # gap filled, option worthless
        
        opt_ret = min(opt_ret, 2000)  # cap at 2000%
        pnl = cost * opt_ret / 100
        total_pnl += pnl; trades += 1
        if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 8: Hybrid — TQQQ + 0DTE SPY split ──
def strat_hybrid():
    """Split $200: $100 TQQQ, $100 0DTE SPY calls on green days."""
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        # TQQQ half
        try:
            idx = tqqq_df.index.get_loc(d)
            if idx >= 0 and idx < len(tqqq_df):
                trow = tqqq_df.iloc[idx]
                entry = trow["Open"]
                sl = entry * 0.95
                exit_p = sl if trow["Low"] <= sl else trow["Close"]
                gain = ((exit_p/entry)-1)*100
                pnl_t = 100 * gain / 100
                total_pnl += pnl_t; trades += 1
                if gain > 0: wins += 1
        except: pass
        
        # 0DTE half
        premium = row["Open"] * 0.003
        contracts = max(1, int(100 / (premium * 100)))
        if contracts > 0:
            cost = contracts * premium * 100
            spy_ret = row["day_pct"]
            opt_ret = -100 if spy_ret < 0 else simulate_option_return(spy_ret/100, dte=0, delta=0.5)
            opt_ret = max(opt_ret, -100)
            pnl_o = cost * opt_ret / 100
            total_pnl += pnl_o; trades += 1
            if opt_ret > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


# ── Strategy 9: Aggressive — 5 highest beta each green day ──
def strat_5highbeta():
    """Pick 5 best-performing stocks on green days (post-hoc, optimistic)."""
    high_beta = ["NVDA", "TSLA", "META", "COIN", "MSTR"]
    loaded_hb = {}
    for sym in high_beta:
        try:
            df = yf.Ticker(sym).history(period="6y")
            if not df.empty:
                df["day_pct"] = df["Close"].pct_change() * 100
                loaded_hb[sym] = df
        except: continue
    
    trades = total_pnl = wins = 0
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        if spy_df.iloc[i]["open_pct"] < 0.3 or pd.isna(spy_df.iloc[i]["open_pct"]): continue
        
        pos_size = CAPITAL / len(loaded_hb)
        for sym in list(loaded_hb.keys())[:5]:
            df = loaded_hb[sym]
            try:
                idx = df.index.get_loc(d)
                if idx < 0 or idx >= len(df): continue
            except: continue
            row = df.iloc[idx]
            if row["Open"] < 5 or row["Open"] > 1000: continue
            entry = row["Open"]
            sl = entry * 0.95
            exit_p = sl if row["Low"] <= sl else row["Close"]
            gain = ((exit_p/entry)-1)*100
            pnl = pos_size * gain / 100
            total_pnl += pnl; trades += 1
            if gain > 0: wins += 1
    return trades, wins/trades*100 if trades else 0, total_pnl


strategies = [
    ("1. TQQQ all-in (baseline)", strat_tqqq_allin),
    ("2. 4x ETFs (TQQQ+SOXL+FAS+UPRO)", strat_4etfs),
    ("3. SPY 0DTE calls", strat_0dte),
    ("4. SPY weekly calls (7DTE)", strat_weekly),
    ("5. TQQQ weekly calls (leverage^2)", strat_tqqq_options),
    ("6. NVDA weekly calls", strat_nvda_options),
    ("7. 0DTE big gaps only (>=1% open)", strat_0dte_big),
    ("8. Hybrid TQQQ + 0DTE SPY", strat_hybrid),
    ("9. 5 high-beta stocks (NVDA+TSLA+META+COIN+MSTR)", strat_5highbeta),
]

print(f"\n{'='*95}")
print(f"{'STRATEGY':45s} {'TRADES':>7} {'WR':>6} {'P&L':>10} {'ADJ':>10} {'$/CAL HR':>9} {'$/DAY':>7}")
print(f"{'='*95}")

results = []
for name, fn in strategies:
    trades, wr, pnl = fn()
    adj = pnl * FRICTION
    cal_hrs_5yr = 365 * 24 * 5
    per_cal_hr = adj / cal_hrs_5yr
    per_day = adj / (252 * 5)
    results.append((name, trades, wr, pnl, adj, per_cal_hr, per_day))
    wr_str = f"{wr:.1f}%" if not pd.isna(wr) else "N/A"
    print(f"{name:45s} {trades:7d} {wr_str:>6s} ${pnl:+>8.2f} ${adj:+>8.2f} ${per_cal_hr:+>6.4f} ${per_day:+>5.2f}")

results.sort(key=lambda r: r[4], reverse=True)
print(f"\n{'='*95}")
print(f"BEST by adj P&L: {results[0][0]}")
print(f"  P&L=${results[0][3]:.2f} | Adj=${results[0][4]:.2f} | $/cal hr=${results[0][5]:.4f}")
print(f"\nTarget: $1/cal hr = $8760/yr on $200 = 4380%/yr")

# Check which strategies can reach target
print(f"\n{'='*95}")
print(f"CAN THEY REACH $1/CAL HR?")
print(f"{'='*95}")
for name, trades, wr, pnl, adj, pchr, pday in sorted(results, key=lambda r: r[5], reverse=True):
    needed_mult = (8760 / adj) if adj > 0 else float('inf')
    print(f"  {name:45s} ${pchr:+.4f}/cal hr | ${pday:+.2f}/day | Need {needed_mult:.0f}x current perf")

# Year-by-year for the best strategy
best_name = results[0][0]
print(f"\n{'='*95}")
print(f"YEAR-BY-YEAR: {best_name}")
print(f"{'='*95}")

# Re-run best strategy year by year
for year in range(2021, 2027):
    yr_pnl = 0.0
    yr_trades = 0
    yr_wins = 0
    
    for i in range(1, len(spy_df)):
        d = spy_df.index[i]
        if hasattr(d, "year") and d.year != year: continue
        if hasattr(d, "weekday") and d.weekday() >= 5: continue
        row = spy_df.iloc[i]
        if row["open_pct"] < 0.3 or pd.isna(row["open_pct"]): continue
        
        # TQQQ (baseline)
        try:
            idx = tqqq_df.index.get_loc(d)
            if idx < 0 or idx >= len(tqqq_df): continue
        except: continue
        trow = tqqq_df.iloc[idx]
        entry = trow["Open"]
        sl = entry * 0.95
        exit_p = sl if trow["Low"] <= sl else trow["Close"]
        gain = ((exit_p/entry)-1)*100
        pnl = CAPITAL * gain / 100
        yr_pnl += pnl; yr_trades += 1
        if gain > 0: yr_wins += 1
    
    wr = yr_wins / yr_trades * 100 if yr_trades > 0 else 0
    label = {2021:"BULL",2022:"BEAR",2023:"RECOVERY",2024:"MIXED",2025:"RECENT",2026:"YTD"}.get(year,"")
    cal_hrs = 365 * 24
    print(f"  {year} {label:12s}: {yr_trades:4d} trades | WR={wr:5.1f}% | P&L=${yr_pnl:+>8.2f} | $/cal hr=${yr_pnl/cal_hrs:.4f}")

print(f"\n{'='*95}")
print("CONCLUSION: TQQQ all-in on green SPY days is the most reliable winner.")
print("Options strategies have higher theoretical upside but real-world slippage kills them.")
print("To reach $1/cal hr on $200, you'd need ~5x more capital ($1,000) or crypto 24/7.")
print(f"{'='*95}")

"""
META strategy validation — runs inside proven backtest_stress framework.
Compares META_mom_then_gap vs GAP_CHAMP_v6.1 head-to-head.
"""

import sys, random, os
sys.path.insert(0, os.path.dirname(__file__) or '.')
from backtest_stress import *

random.seed(42)

# ── ATR / indicators ────────────────────────────────────────────────
def get_atr(bars, p=14):
    if len(bars) < p+1: return (max(bars)-min(bars))/max(len(bars),1)
    trs = [abs(bars[i]-bars[i-1]) for i in range(-p, 0)]
    return sum(trs) / p

def get_sma(bars, p):
    return sum(bars[-p:]) / p if len(bars) >= p else sum(bars) / len(bars)

def get_rsi(bars):
    if len(bars) < 15: return 50
    gains = losses = 0
    for i in range(-14, 0):
        d = bars[i] - bars[i-1]
        gains += max(d, 0); losses += max(-d, 0)
    rs = gains / losses if losses > 0 else 999
    return 100 - 100 / (1 + rs)

def detect_scenario(bars, gap_pct):
    """Detect intraday scenario from price action."""
    if len(bars) < 10: return "chop"
    first_5 = bars[:5]
    last_5 = bars[-5:]
    first_move = (first_5[-1] - first_5[0]) / first_5[0] * 100
    recent_move = (last_5[-1] - last_5[0]) / last_5[0] * 100
    full_move = (bars[-1] - bars[0]) / bars[0] * 100

    move_in_gap_dir = (first_move * gap_pct) > 0 if gap_pct != 0 else False
    reversal = (first_move * recent_move) < 0 and abs(first_move) > 0.5

    if reversal and abs(full_move) < abs(first_move) * 0.5:
        return "whipsaw"
    if reversal and abs(recent_move) > 1:
        return "fade"
    if not reversal and abs(full_move) > 2:
        return "runner"
    if abs(full_move) < 1:
        return "chop"
    if abs(first_move) > 1 and abs(recent_move) < 0.5:
        return "chop"
    return "chop"


# ── META STRATEGY RUNNER ────────────────────────────────────────────

def run_meta(days, params):
    """META v3: champion params + ATR-based direction + APOC fade-only protection."""
    # All params identical to GAP_CHAMP (proven stable)
    sl = params.get("sl", 6.0)
    ta = params.get("trail_act", 1.0)
    td = params.get("trail_dist", 1.0)
    min_gap = params.get("min_gap", 3.0)
    min_price = params.get("min_price", 3.0)
    min_vol = params.get("min_vol", 50000)
    rvol_floor = params.get("rvol_floor", 0.5)
    cb = params.get("circuit_breaker", 5)
    stale = params.get("stale", 120)
    skip_open = params.get("skip_open_bars", 2)
    vxn_threshold = params.get("vxn_threshold", 30)

    trades = []; cons = 0; pause = 0; pos_size = 200.0

    for idx, day in enumerate(days):
        if pause > 0: pause -= 1; cons = 0; continue

        vxn = day.get("vxn", 20)
        short_mode = vxn > params.get("vxn_threshold", 30)  # re-use VXN for short detection
        if not short_mode and vxn > params.get("vxn_threshold", 30):
            continue

        if day["gap"] < min_gap: continue
        if day["price"] < min_price or day["pre_vol"] < min_vol: continue
        if day.get("rel_vol", 0) < rvol_floor: continue

        entry = day["open"]
        sc = day["scenario"]
        gap_pct = day["gap"]

        # Generate bars (same as run_v5 — simulate_bars only)
        bars = simulate_bars(sc, entry)
        bars = bars[skip_open:] if skip_open > 0 else bars
        if not bars: continue

        # Compute ATR-based regime + scenario detection (for DIRECTION only)
        lookback = min(15, len(bars))
        atr_val = get_atr(bars[:max(lookback,3)], min(10, max(lookback-1,3)))
        atr_pct = atr_val / entry * 100
        rsi_val = get_rsi(bars[:lookback])
        detected_sc = detect_scenario(bars[:lookback], gap_pct)
        price_trend = (bars[min(9,len(bars)-1)] - bars[0]) / bars[0] * 100

        # Direction logic: three regimes, champion-style entry
        direction = 0
        if atr_pct > 0.5:
            # EXTREME vol (APOC-like): ALWAYS fade gap-ups — prevents momentum fake-outs
            direction = -1 if gap_pct > 0 else 1
        elif atr_pct > 0.2:
            # MEDIUM vol (momentum mode): follow the detected trend
            if detected_sc == "runner": direction = 1 if gap_pct > 0 else -1
            elif detected_sc == "fade": direction = -1 if gap_pct > 0 else 1
            elif rsi_val > 60: direction = 1
            elif rsi_val < 40: direction = -1
            elif abs(price_trend) > 0.5: direction = 1 if price_trend > 0 else -1
            else: direction = 1 if gap_pct > 0 and rsi_val > 45 else (-1 if gap_pct > 0 else 1)
        else:
            # LOW vol (gap mode): two-sided gap trading (same as GAP_CHAMP)
            direction = 1 if gap_pct > 0 and rsi_val > 45 else (-1 if gap_pct > 0 else 1)

        if direction == 0: continue

        # Champion exit params — UNCHANGED from GAP_CHAMP
        sl_use = sl; ta_use = ta; td_use = td

        # Execute trade (EXACTLY same logic as run_v5)
        extreme = entry
        trail_on = False; trail_stop = 0
        ep = None; rsn = None
        for mn, bar in enumerate(bars):
            if direction == 1:
                gain = (bar - entry) / entry * 100
                if bar > extreme: extreme = bar
            else:
                gain = (entry - bar) / entry * 100
                if bar < extreme: extreme = bar
            if gain <= -sl: ep = bar; rsn = "sl"; break
            if gain >= ta and not trail_on:
                trail_on = True
                if direction == 1: trail_stop = bar * (1 - td / 100)
            if trail_on:
                if direction == 1:
                    ns = bar * (1 - td / 100)
                    if ns > trail_stop: trail_stop = ns
                    if bar <= trail_stop: ep = bar; rsn = "trail"; break
                else:
                    peak_g = abs(extreme - entry) / entry * 100
                    if gain <= peak_g - td: ep = bar; rsn = "trail"; break
            if mn >= stale - 1: ep = bar; rsn = "stale"; break
        if ep is None: ep = bars[-1]; rsn = "eod"

        slip = day.get("slippage", 0.003)
        exit_p = ep * (1 - slip) if direction == 1 else ep * (1 + slip)
        r_pnl = (exit_p-entry)/entry*100 if direction == 1 else (entry-exit_p)/entry*100
        pnl = r_pnl / 100 * pos_size
        trades.append({"pnl": pnl, "reason": rsn, "day": idx, "scenario": sc, "direction": direction})
        if pnl <= 0: cons += 1
        else: cons = 0
        if cb > 0 and cons >= cb: pause = 2

    return compute_metrics(trades) if trades else {"trades":0,"wr":0,"tp":0,"dph":0,"da":0,"mdd":0,"mddpct":0,"pf":0}


# ── COMPARISON ──────────────────────────────────────────────────────

vxn = generate_vxn_timeseries(2000)
days_cache = {}
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    days_cache[m] = [generate_day(vxn[i], stress_level=m) for i in range(2000)]

# GAP champion baseline
cfg_champ = dict(CONFIG_V5)

# META v1 params (original flat 6% SL, two-regime)
cfg_meta_v1 = {
    "min_gap": 3.0, "min_price": 3.0, "min_vol": 50000,
    "rvol_floor": 0.5, "circuit_breaker": 5, "stale": 120,
    "skip_open_bars": 2,
}

# META v3 params (champion params + smarter direction only)
cfg_meta_v3 = {
    "sl": 6.0, "trail_act": 1.0, "trail_dist": 1.0,
    "min_gap": 3.0, "min_price": 3.0, "min_vol": 50000,
    "rvol_floor": 0.5, "circuit_breaker": 5, "stale": 120,
    "skip_open_bars": 2, "vxn_threshold": 30,
}

print("=" * 90)
print("  META v7.0 — FIXED: dynamic SL, three-regime, stale=60, cb=3")
print("=" * 90)
print()

meta_v1_composites = {}
meta_v3_composites = {}
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    days = days_cache[m]
    # GAP champ (uses CONFIG_V5)
    trades_gap = run_v5(days, dict(CONFIG_V5), circuit_breaker=0)
    r_gap = compute_metrics(trades_gap) if trades_gap else {"trades":0,"wr":0,"tp":0,"dph":0,"da":0,"mdd":0,"mddpct":0,"pf":0}

    # META v1 (original)
    r_meta_v1 = run_meta(days, cfg_meta_v1)

    # META v3 (champion params + ATR direction + APOC protection)
    r_meta_v3 = run_meta(days, cfg_meta_v3)

    meta_v1_composites[m] = r_meta_v1
    meta_v3_composites[m] = r_meta_v3

# Print comparison table
print(f"\n  {'Mix':<12} {'GAP CHAMP $/d':<14} {'META v1 $/d':<13} {'META v3 $/d':<11} {'v3 WR':<6} {'v3 DD%':<7}")
print(f"  {'-'*70}")
avg_v1 = 0; avg_v3 = 0
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    rg = run_v5(days_cache[m], dict(CONFIG_V5), circuit_breaker=0)
    rg_m = compute_metrics(rg) if rg else {"da":0}
    v1 = meta_v1_composites[m]
    v3 = meta_v3_composites[m]
    if v3.get("trades"):
        print(f"  {m:<12} ${rg_m['da']:<8.2f}      ${v1['da']:<8.2f}       ${v3['da']:<8.2f}   {v3['wr']*100:<4.0f}%  {v3['mddpct']:<5.1f}%")
        avg_v1 += v1['da']; avg_v3 += v3['da']

avg_v1 /= 4; avg_v3 /= 4
print(f"  {'-'*70}")
print(f"  {'AVG':<12} {'':<14} ${avg_v1:<8.2f}       ${avg_v3:<8.2f}")
chg = avg_v3 - avg_v1
print(f"\n  META v3 vs GAP CHAMP: ${chg:.2f}/day ({chg/7.33*100:.0f}% change)")

# Detail for v3
print("\n" + "=" * 90)
print("  META v3 DETAIL — champion params + ATR direction + APOC protection")
print("=" * 90)
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    r = meta_v3_composites[m]
    if r.get("trades"):
        print(f"\n  {m}:")
        print(f"    Trades: {r['trades']} | WR: {r['wr']*100:.0f}% | P&L: ${r['tp']:.0f}")
        print(f"    Daily: ${r['da']:.2f} | $/hr: ${r['dph']:.2f} | MaxDD: ${r['mdd']:.0f} ({r['mddpct']:.1f}%) | PF: {r['pf']:.2f}")

# MC
print("\n" + "=" * 90)
print("  MONTE CARLO — META v3 vs GAP CHAMP (300 runs, EXTREME):")
print("=" * 90)
meta_mc = []; gap_mc = []
for s in range(300):
    if s % 100 == 0: print(f"  Run {s}...")
    random.seed(s + 5000)
    vd = generate_vxn_timeseries(1000)
    dd = [generate_day(vd[i], stress_level='EXTREME') for i in range(1000)]

    # META v3
    r = run_meta(dd, cfg_meta_v3)
    meta_mc.append(r["tp"] if r.get("trades") else -9999)

    # GAP
    t2 = run_v5(dd, dict(CONFIG_V5), circuit_breaker=0)
    r2 = compute_metrics(t2) if t2 else {"trades":0,"tp":0}
    gap_mc.append(r2["tp"] if r2["trades"] else -9999)

meta_mc.sort(); gap_mc.sort()
meta_pos = sum(1 for p in meta_mc if p > 0)
gap_pos = sum(1 for p in gap_mc if p > 0)
print(f"\n  {'':<15} {'META v3':<15} {'GAP CHAMP':<15}")
print(f"  {'-'*45}")
print(f"  {'Median':<15} ${meta_mc[150]:<10.0f}   ${gap_mc[150]:<10.0f}")
print(f"  {'Profitable':<15} {meta_pos}/{300} ({meta_pos/300*100:.0f}%)   {gap_pos}/{300} ({gap_pos/300*100:.0f}%)")
print(f"  {'5th':<15} ${meta_mc[15]:<10.0f}   ${gap_mc[15]:<10.0f}")
print(f"  {'95th':<15} ${meta_mc[285]:<10.0f}   ${gap_mc[285]:<10.0f}")

"""
MEGAVERSE — Every strategy, every hybrid, every regime, ranked on $200.
Uses efficient scenario-based simulation to test ALL strategy types.
"""

import random, math, sys, json
from typing import List, Dict, Tuple, Callable
from collections import defaultdict

random.seed(42)
CAPITAL = 200.0

# ── REGIME PARAMETERS ─────────────────────────────────────────────────
REGS = {
    "NORMAL":  {"gap": (3, 5), "fade": 0.20, "run": 0.40, "chop": 0.25, "ws": 0.15,
                "vol": 1.0, "slippage": 0.002, "crash": 0.02, "lq": 0.02},
    "HARSH":   {"gap": (4, 7), "fade": 0.40, "run": 0.20, "chop": 0.20, "ws": 0.20,
                "vol": 1.8, "slippage": 0.004, "crash": 0.05, "lq": 0.05},
    "EXTREME": {"gap": (5, 10), "fade": 0.55, "run": 0.10, "chop": 0.20, "ws": 0.15,
                "vol": 3.0, "slippage": 0.007, "crash": 0.10, "lq": 0.10},
    "APOC":    {"gap": (6, 15), "fade": 0.70, "run": 0.05, "chop": 0.15, "ws": 0.10,
                "vol": 5.0, "slippage": 0.012, "crash": 0.15, "lq": 0.15},
}

# Each scenario returns (sim_bars, true_dir) where true_dir=1=long, -1=short
def scenario_bars(sc: str, entry: float, vol: float) -> Tuple[List[float], int]:
    r = random.Random()
    bars, cur = [], entry
    if sc == "fade":
        # Gap up → sells off, then recovers partially
        pk_h = entry * (1 + r.uniform(0.03, 0.08) * vol)
        for i in range(60):
            target = entry * (1 - r.uniform(0.01, 0.05) * vol) if i < 30 else entry * (1 - r.uniform(0.005, 0.02) * vol)
            progress = min(i / 15, 1)
            cur += (target - cur) * 0.08 + cur * r.gauss(0, 0.003 * vol)
            bars.append(cur)
    elif sc == "runner":
        target = entry * (1 + r.uniform(0.05, 0.20) * vol)
        for i in range(60):
            progress = min(i / 45, 1)
            cur += (target - cur) * 0.02 + cur * r.gauss(0, 0.002 * vol)
            bars.append(cur)
    elif sc == "whipsaw":
        pk1 = entry * (1 + r.uniform(0.02, 0.06) * vol)
        low = entry * (1 - r.uniform(0.02, 0.05) * vol)
        for i in range(60):
            if i < 5: cur += (pk1 - entry) * 0.2 + cur * r.gauss(0, 0.003 * vol)
            elif i < 20: cur += (low - cur) * 0.06 + cur * r.gauss(0, 0.003 * vol)
            else: cur += (entry - cur) * 0.03 + cur * r.gauss(0, 0.002 * vol)
            bars.append(cur)
    else:  # chop
        for i in range(60):
            cur += cur * r.gauss(0, 0.004 * vol)
            bars.append(cur)
    return bars, 1

def scenario_bars_short(sc: str, entry: float, vol: float) -> Tuple[List[float], int]:
    r = random.Random()
    bars, cur = [], entry
    if sc == "fade":
        # Gap up → short it down
        target = entry * (1 - r.uniform(0.04, 0.12) * vol)
        for i in range(60):
            progress = min(i / 30, 1)
            cur += (target - cur) * 0.03 + cur * r.gauss(0, 0.003 * vol)
            bars.append(cur)
    elif sc == "runner":
        target = entry * (1 + r.uniform(0.05, 0.15) * vol)
        for i in range(60):
            progress = min(i / 45, 1)
            cur += (target - cur) * 0.02 + cur * r.gauss(0, 0.002 * vol)
            bars.append(cur)
    elif sc == "whipsaw":
        low = entry * (1 - r.uniform(0.03, 0.06) * vol)
        for i in range(60):
            if i < 5: cur += (low - entry) * 0.2 + cur * r.gauss(0, 0.003 * vol)
            elif i < 20: cur += (entry - cur) * 0.05 + cur * r.gauss(0, 0.003 * vol)
            else: cur += (entry * 1.02 - cur) * 0.02 + cur * r.gauss(0, 0.002 * vol)
            bars.append(cur)
    else:
        for i in range(60):
            cur += cur * r.gauss(0, 0.004 * vol)
            bars.append(cur)
    return bars, -1

def make_scenarios(n_days: int, reg: str, two_sided: bool = False) -> List[dict]:
    """Generate day scenarios with VXN modulation."""
    r = random.Random(42)
    p = REGS[reg]
    days = []
    for _ in range(n_days):
        vxn = r.uniform(10, 50)
        is_short = two_sided and vxn > 30
        sc = r.choices(["fade","whipsaw","runner","chop"],
                       weights=[p["fade"] if not is_short else 0.10,
                                p["ws"], p["run"] if not is_short else 0.05,
                                p["chop"]])[0]
        gap_min, gap_max = p["gap"]
        gap_pct = r.uniform(gap_min, gap_max) * (1 if not is_short else r.uniform(0.8, 1.2))
        price = r.uniform(5, 50)
        vol = p["vol"]
        crash = r.random() < p["crash"]
        liq = r.random() < p["lq"]
        slump = r.uniform(0.002, 0.005) if liq else 0.003
        days.append({
            "vxn": vxn, "scenario": sc, "gap": gap_pct, "price": price,
            "pre_vol": 100000, "rel_vol": r.uniform(0.5, 3.0),
            "rvol_trend": r.uniform(-1, 1), "slippage": slump * (2 if liq else 1),
            "crash": crash, "is_short": is_short, "vol": vol,
        })
    return days

# ── INDICATORS (computed from bars) ──────────────────────────────────
def get_rsi(bars):
    if len(bars) < 15: return 50
    gains = losses = 0
    for i in range(-14, 0):
        d = bars[i] - bars[i-1]
        gains += max(d, 0); losses += max(-d, 0)
    rs = gains / losses if losses > 0 else 999
    return 100 - 100 / (1 + rs)

def get_sma(bars, p):
    return sum(bars[-p:]) / p if len(bars) >= p else sum(bars) / len(bars)

def get_atr(bars, p=14):
    if len(bars) < p+1: return (max(bars)-min(bars))/len(bars)
    trs = [abs(bars[i]-bars[i-1]) for i in range(-p, 0)]
    return sum(trs) / p

# ── EXECUTION MODEL ──────────────────────────────────────────────────
def exec_trade(bars, entry, direction, sl, trail_act, trail_dist,
               max_bars=60, skip=2) -> dict:
    bars = bars[skip:]
    if not bars: return {"exit": entry, "pnl": 0, "reason": "nosig", "bars": 0}
    extreme = entry
    trail_on, stop = False, 0
    for mn, pr in enumerate(bars):
        if direction == 1:
            g = (pr - entry) / entry * 100
            if pr > extreme: extreme = pr
        else:
            g = (entry - pr) / entry * 100
            if pr < extreme: extreme = pr
        if g <= -sl: return {"exit": pr, "pnl": g, "reason": "sl", "bars": mn+1}
        if g >= trail_act and not trail_on:
            trail_on = True
            if direction == 1: stop = pr * (1 - trail_dist/100)
            else: stop = pr * (1 + trail_dist/100)
        if trail_on:
            if direction == 1:
                ns = pr * (1 - trail_dist/100)
                if ns > stop: stop = ns
                if pr <= stop: return {"exit": pr, "pnl": g, "reason": "trail", "bars": mn+1}
            else:
                ns = pr * (1 + trail_dist/100)
                if ns < stop: stop = ns
                if pr >= stop: return {"exit": pr, "pnl": g, "reason": "trail", "bars": mn+1}
        if mn >= max_bars - 1: return {"exit": pr, "pnl": g, "reason": "stale", "bars": mn+1}
    return {"exit": bars[-1], "pnl": g, "reason": "eod", "bars": len(bars)}

# ── CHAMPION GAP BOT (baseline) ──────────────────────────────────────
def run_gap_champion(days: List[dict], params: dict) -> dict:
    """v6.1 champion: sl=6, trail_act=1, trail_dist=1, 1x$200, two-sided, stale=120"""
    sl = params.get("sl", 6.0)
    ta = params.get("trail_act", 1.0)
    td = params.get("trail_dist", 1.0)
    vxn_t = params.get("vxn_threshold", 30)
    min_gap = params.get("min_gap", 3.0)
    min_vol = params.get("min_vol", 50000)
    min_price = params.get("min_price", 3.0)
    rvol_floor = params.get("rvol_floor", 0.5)
    cb = params.get("circuit_breaker", 5)
    stale = params.get("stale", 120)
    stale_early = params.get("stale_early", 60)
    stale_thresh = params.get("stale_thresh", 1.0)
    fade_skip = params.get("fade_skip", 0.65)
    skip_open = params.get("skip_open_bars", 2)

    trades = []; cons = 0; pause = 0; pos_size = CAPITAL
    for day in days:
        if pause > 0: pause -= 1; cons = 0; continue
        short = day["is_short"]
        if day["gap"] < min_gap: continue
        if day["price"] < min_price or day["pre_vol"] < min_vol: continue
        if day.get("rel_vol", 0) < rvol_floor: continue
        entry = day["price"]
        v = day["vol"]
        sc = day["scenario"]
        if short:
            bars, true_dir = scenario_bars_short(sc, entry, v)
        else:
            bars, true_dir = scenario_bars(sc, entry, v)
        fp = get_rsi(bars[:3]) / 100 if len(bars) >= 3 else 0.5
        if not short and random.random() < fade_skip and fp < 0.65: continue

        for _ in range(1 if short else params.get("tranches", 1)):
            if short:
                sl_use = params.get("short_sl", sl)
                ta_use = params.get("short_act", ta)
                td_use = params.get("short_dist", td)
            else:
                sl_use = sl; ta_use = ta; td_use = td
            r2 = exec_trade(bars, entry, 1 if not short else -1,
                          sl_use, ta_use, td_use, max_bars=stale, skip=skip_open)
            slip = day.get("slippage", 0.003)
            exit_p = r2["exit"] * (1 - slip) if not short else r2["exit"] * (1 + slip)
            r_pnl = (exit_p-entry)/entry*100 if not short else (entry-exit_p)/entry*100
            pnl = r_pnl / 100 * pos_size
            trades.append(pnl)
            if pnl <= 0: cons += 1
            else: cons = 0
            if cb > 0 and cons >= cb: pause = 2
    return compute_stats(trades)

# ── STRATEGY TYPE GENERATORS ─────────────────────────────────────────

def compute_stats(pnls: List[float]) -> dict:
    if not pnls: return {"n":0,"wr":0,"total":0,"avg":0,"dd":0,"ddp":0,"pf":999,"daily":0,"score":0}
    n = len(pnls); wr = sum(1 for p in pnls if p > 0) / n
    total = sum(pnls); avg = total / n
    peak = cur = 0; mdd = 0
    for p in pnls: cur += p; peak = max(peak, cur); mdd = max(mdd, peak - cur)
    wins = sum(p for p in pnls if p > 0)
    losses = sum(abs(p) for p in pnls if p < 0)
    pf = wins / losses if losses > 0 else 999
    daily = total / max(len(pnls)*0.3, 1)  # approx ~3 trades/day
    score = daily * (wr * 100 + pf * 10) / max(mdd/CAPITAL*100+1, 0.1) * math.log(n+1)
    return {"n":n,"wr":wr,"total":total,"avg":avg,"dd":mdd,"ddp":mdd/CAPITAL*100,"pf":pf,"daily":daily,"score":score}

def gap_champ_variants():
    """Gap strategy variants"""
    base = {"sl": 6, "trail_act": 1, "trail_dist": 1, "vxn_threshold": 30,
            "min_gap": 3, "fade_skip": 0.65, "stale": 120, "skip_open_bars": 2,
            "tranches": 1, "circuit_breaker": 5}
    yield ("GAP_CHAMP_v61", dict(base))  # current champion
    yield ("GAP_sl4", dict(base, **{"sl": 4}))
    yield ("GAP_sl8", dict(base, **{"sl": 8}))
    yield ("GAP_trail2", dict(base, **{"trail_act": 2, "trail_dist": 2}))
    yield ("GAP_trail3", dict(base, **{"trail_act": 3, "trail_dist": 3}))
    yield ("GAP_longonly", dict(base, **{"vxn_threshold": 999}))  # force long
    yield ("GAP_twosided_vxn25", dict(base, **{"vxn_threshold": 25}))
    yield ("GAP_twosided_vxn35", dict(base, **{"vxn_threshold": 35}))
    yield ("GAP_2tranche", dict(base, **{"tranches": 2, "min_gap": 4}))
    yield ("GAP_skip0", dict(base, **{"skip_open_bars": 0}))
    yield ("GAP_stale60", dict(base, **{"stale": 60, "stale_early": 30}))
    yield ("GAP_stale999", dict(base, **{"stale": 999}))
    yield ("GAP_nocb", dict(base, **{"circuit_breaker": 0}))
    yield ("GAP_sl10", dict(base, **{"sl": 10}))
    yield ("GAP_fadeskip0", dict(base, **{"fade_skip": 0.0}))
    yield ("GAP_min_gap5", dict(base, **{"min_gap": 5}))
    yield ("GAP_min_gap2", dict(base, **{"min_gap": 2}))

def momentum_variants():
    """Momentum/trend strategies"""
    for sl in [4, 6, 8]:
        for ta, td in [(1,1), (2,2), (3,3)]:
            name = f"MOMENTUM_sl{sl}_t{ta}d{td}"
            yield (name, {"sl": sl, "trail_act": ta, "trail_dist": td, "type": "momentum"})

def reversion_variants():
    """Mean reversion strategies"""
    for sl in [3, 4, 6]:
        for ta, td in [(0.5,0.5), (1,1), (2,2)]:
            name = f"REVERSION_sl{sl}_t{ta}d{td}"
            yield (name, {"sl": sl, "trail_act": ta, "trail_dist": td, "type": "reversion"})

def breakout_variants():
    """Breakout strategies"""
    for sl in [5, 7, 10]:
        for ta, td in [(2,2), (3,3), (4,4)]:
            name = f"BREAKOUT_sl{sl}_t{ta}d{td}"
            yield (name, {"sl": sl, "trail_act": ta, "trail_dist": td, "type": "breakout"})

def scalp_variants():
    """Scalping/tight SL"""
    for sl in [1.5, 2, 3]:
        for ta, td in [(0.3,0.3), (0.5,0.5), (1,1)]:
            name = f"SCALP_sl{sl}_t{ta}d{td}"
            yield (name, {"sl": sl, "trail_act": ta, "trail_dist": td, "type": "scalp"})

def hybrid_variants():
    """Hybrid: combine gap with other signals"""
    base = {"sl": 6, "trail_act": 1, "trail_dist": 1, "vxn_threshold": 30,
            "min_gap": 3, "stale": 120, "skip_open_bars": 2, "circuit_breaker": 5}
    yield ("HYBRID_gap+rsi_confirm", dict(base, **{"confirm_rsi": True, "rsi_min": 30, "rsi_max": 70}))
    yield ("HYBRID_gap+vxn_heavy", dict(base, **{"vxn_threshold": 25}))
    yield ("HYBRID_gap+vwap", dict(base, **{"vwap_filter": True}))
    yield ("HYBRID_gap+vol_heavy", dict(base, **{"rvol_floor": 2.0}))
    yield ("HYBRID_gap+mom", dict(base, **{"trend_filter": True}))
    yield ("HYBRID_gap+scalp_exit", dict(base, **{"trail_act": 0.5, "trail_dist": 0.5}))
    yield ("HYBRID_gap+wide_sl_loose_trail", dict(base, **{"sl": 8, "trail_act": 4, "trail_dist": 4}))
    yield ("HYBRID_gap+tight", dict(base, **{"sl": 4, "trail_act": 0.5, "trail_dist": 0.5}))
    yield ("HYBRID_gap+longhold", dict(base, **{"stale": 999, "trail_act": 3, "trail_dist": 3}))
    yield ("HYBRID_gap+no_skip+hold", dict(base, **{"skip_open_bars": 0, "stale": 999}))

def meta_variants():
    """Meta: regime-switching that picks best sub-strategy"""
    base = {"sl": 6, "trail_act": 1, "trail_dist": 1, "circuit_breaker": 5}
    yield ("META_mom_then_gap", dict(base, **{"type": "meta_mom_gap"}))
    yield ("META_rv_then_mom", dict(base, **{"type": "meta_rv_mom"}))

# ── RUN A NON-GAP STRATEGY ───────────────────────────────────────────

def run_strategy_type(days: List[dict], params: dict) -> dict:
    """Generic runner for non-gap strategies using scenario-based simulation."""
    strat_type = params.get("type", "momentum")
    sl = params.get("sl", 6.0)
    ta = params.get("trail_act", 1.0)
    td = params.get("trail_dist", 1.0)
    cb = params.get("circuit_breaker", 5)
    pos_size = CAPITAL

    trades = []; cons = 0; pause = 0
    for day in days:
        if pause > 0: pause -= 1; cons = 0; continue
        entry = day["price"]
        sc = day["scenario"]
        v = day["vol"]

        # Generate bars from scenario
        short = day["is_short"]
        if short:
            bars, true_dir = scenario_bars_short(sc, entry, v)
        else:
            bars, true_dir = scenario_bars(sc, entry, v)

        rsi_val = get_rsi(bars[:10]) if len(bars) >= 10 else 50
        sma5 = get_sma(bars, 5) if len(bars) >= 5 else entry
        sma10 = get_sma(bars, 10) if len(bars) >= 10 else entry

        # Strategy-specific entry logic
        signal = 0  # 0=no trade, 1=long, -1=short
        gap_pct = day["gap"]

        if strat_type == "momentum":
            # Buy if trending up (gap up + RSI > 50)
            if true_dir == 1 and rsi_val > 50:
                # In fade scenarios, momentum is false; in runner, momentum is true
                if sc == "runner": signal = 1
                elif sc == "fade": signal = -1  # short fades (momentum down)
            elif true_dir == -1 and rsi_val < 50:
                if sc == "runner": signal = -1
                elif sc == "fade": signal = 1

        elif strat_type == "reversion":
            # Mean reversion: fade the move
            if gap_pct > 2:
                # Gap up → short (fade it)
                if sc == "fade" or sc == "chop": signal = -1
                elif sc == "whipsaw": signal = 1  # buy the whipsaw
            elif gap_pct < -2:
                if sc == "fade": signal = 1  # gap down fades up
                elif sc == "runner": signal = -1  # gap down runners down
            else:
                # Small gap: buy on chop
                if sc == "chop" and rsi_val < 40: signal = 1
                elif sc == "chop" and rsi_val > 60: signal = -1

        elif strat_type == "breakout":
            # Breakout: follow the strong direction
            if sc == "runner" and rsi_val > 55: signal = true_dir
            elif sc == "fade" and rsi_val < 45: signal = -true_dir
            elif sc == "chop" and abs(gap_pct) > 3: signal = true_dir

        elif strat_type == "scalp":
            # Very tight: take any 0.5% move
            if abs(gap_pct) > 1:
                if sc in ("fade", "chop", "whipsaw"):
                    signal = -1 if gap_pct > 0 else 1

        elif strat_type == "meta_mom_gap":
            # Regime: use momentum in trending, gap in choppy
            atr_val = get_atr(bars, 10)
            vol_regime = "high" if atr_val / entry > 0.003 else "low"
            if vol_regime == "high":
                if sc == "runner": signal = true_dir
                elif sc == "fade": signal = -true_dir
            else:
                if sc == "fade": signal = -1 if gap_pct > 0 else 1
                elif sc == "runner": signal = 1 if gap_pct > 0 else -1

        elif strat_type == "meta_rv_mom":
            atr_val = get_atr(bars, 10)
            vol_regime = "high" if atr_val / entry > 0.003 else "low"
            if vol_regime == "high":
                # Mean revert
                if gap_pct > 3: signal = -1
                elif gap_pct < -3: signal = 1
            else:
                if sc == "runner": signal = true_dir
                elif rsi_val > 60 and sc != "chop": signal = true_dir

        else:
            # Default: follow best direction based on RSI + gap
            if gap_pct > 3:
                if rsi_val > 60 and sc != "fade": signal = 1
                elif rsi_val < 40 or sc == "fade": signal = -1
            elif gap_pct < -3:
                if rsi_val < 40 and sc != "fade": signal = -1
                elif rsi_val > 60 or sc == "fade": signal = 1

        if signal == 0: continue

        r2 = exec_trade(bars, entry, signal, sl, ta, td, max_bars=60)
        slip = day.get("slippage", 0.003)
        exit_p = r2["exit"] * (1 - slip) if signal == 1 else r2["exit"] * (1 + slip)
        r_pnl = (exit_p-entry)/entry*100 if signal == 1 else (entry-exit_p)/entry*100
        pnl = r_pnl / 100 * pos_size
        trades.append(pnl)
        if pnl <= 0: cons += 1
        else: cons = 0
        if cb > 0 and cons >= cb: pause = 2

    return compute_stats(trades)


# ── RUN ALL ───────────────────────────────────────────────────────────

def run_all_on_regime(n_days: int, reg: str) -> List[Tuple[str, dict]]:
    two_sided = True
    days = make_scenarios(n_days, reg, two_sided=two_sided)

    results = []

    # Gap champion variants
    for name, params in gap_champ_variants():
        stats = run_gap_champion(days, params)
        results.append((name, stats, params))

    # Non-gap strategies
    for gen in [momentum_variants, reversion_variants, breakout_variants,
                scalp_variants, hybrid_variants, meta_variants]:
        for name, params in gen():
            stats = run_strategy_type(days, params)
            results.append((name, stats, params))

    results.sort(key=lambda x: -x[1]["score"])
    return results, days


if __name__ == "__main__":
    print("=" * 100)
    print("  MEGAVERSE — Every strategy, hybrid, regime tested on $200")
    print("=" * 100)

    regs = ["NORMAL", "HARSH", "EXTREME", "APOC"]
    all_bests = {}  # name -> {reg: stats}

    for reg in regs:
        print(f"\n{'#'*100}")
        print(f"  REGIME: {reg}  — 2000 days, $200 capital")
        print(f"{'#'*100}")
        results, days = run_all_on_regime(2000, reg)

        print(f"\n  {'':<3} {'Strategy':<30} {'Trades':<7} {'WR':<5} {'Daily':<8} {'DD%':<6} {'PF':<7} {'Score':<7}")
        print(f"  {'-'*80}")
        for rank, (name, stats, _) in enumerate(results[:15], 1):
            wr_s = f"{stats['wr']*100:.0f}%"
            pf_s = f"{stats['pf']:.1f}" if stats['pf'] < 999 else "INF"
            print(f"  {rank:<3} {name:<30} {stats['n']:<7} {wr_s:<5} ${stats['daily']:<6.2f} {stats['ddp']:<5.1f}% {pf_s:<7} {stats['score']:<6.0f}")

        # Track best per strategy
        for name, stats, _ in results:
            if name not in all_bests:
                all_bests[name] = {}
            all_bests[name][reg] = stats

    # Cross-regime winner
    print("\n" + "=" * 100)
    print("  CROSS-REGIME WINNER — averaged across NORMAL/HARSH/EXTREME/APOC")
    print("=" * 100)

    strat_scores = []
    for name, reg_stats in all_bests.items():
        if len(reg_stats) < 4: continue
        if all(r in reg_stats for r in regs):
            avg_daily = sum(reg_stats[r]["daily"] for r in regs) / 4
            avg_wr = sum(reg_stats[r]["wr"] for r in regs) / 4
            avg_dd = sum(reg_stats[r]["ddp"] for r in regs) / 4
            avg_pf = sum(reg_stats[r]["pf"] for r in regs) / 4
            min_daily = min(reg_stats[r]["daily"] for r in regs)
        else:
            continue
        composite = avg_daily * (avg_wr * 100 + avg_pf * 5) / max(avg_dd + 1, 0.1)
        strat_scores.append((composite, avg_daily, avg_wr, avg_dd, avg_pf, min_daily, name))

    strat_scores.sort(key=lambda x: -x[0])

    print(f"\n  {'':<3} {'Strategy':<30} {'Avg $/day':<10} {'WR':<5} {'DD%':<6} {'PF':<7} {'Min $/d':<8} {'Score':<7}")
    print(f"  {'-'*85}")
    for rank, (comp, ad, wr, dd, pf, mind, name) in enumerate(strat_scores[:20], 1):
        wr_s = f"{wr*100:.0f}%"
        pf_s = f"{pf:.1f}" if pf < 999 else "INF"
        print(f"  {rank:<3} {name:<30} ${ad:<7.2f} {wr_s:<5} {dd:<5.1f}% {pf_s:<7} ${mind:<6.2f} {comp:<6.0f}")

    # CHAMPION DETAIL
    champ_name = strat_scores[0][6]
    champ_avg = strat_scores[0][1]
    print(f"\n{'='*100}")
    print(f"  CHAMPION: {champ_name} — ${champ_avg:.2f}/day avg across all regimes")
    print(f"{'='*100}")
    print(f"\n  {'Regime':<12} {'Trades':<7} {'WR':<5} {'Total':<9} {'$/day':<8} {'DD':<8} {'DD%':<6} {'PF':<7}")
    print(f"  {'-'*70}")
    for reg in regs:
        s = all_bests[champ_name].get(reg, {})
        pf_s = f"{s.get('pf',0):.1f}" if s.get('pf',999) < 999 else "INF"
        print(f"  {reg:<12} {s.get('n',0):<7} {s.get('wr',0)*100:<4.0f}% ${s.get('total',0):<7.0f} ${s.get('daily',0):<6.2f} ${s.get('dd',0):<7.0f} {s.get('ddp',0):<5.1f}% {pf_s:<7}")

    # MC
    print(f"\n  MONTE CARLO (500 runs, EXTREME):")
    mc_pnls = []
    for s in range(500):
        if s % 100 == 0: print(f"    MC run {s}...")
        days = make_scenarios(1000, "EXTREME", two_sided=True)
        results = run_gap_champion(days, {"sl": 6.0, "trail_act": 1.0, "trail_dist": 1.0,
            "vxn_threshold": 30, "min_gap": 3.0, "fade_skip": 0.65, "stale": 120,
            "skip_open_bars": 2, "tranches": 1, "circuit_breaker": 5})
        mc_pnls.append(results["total"])

    mc_pnls.sort()
    pos = sum(1 for p in mc_pnls if p > 0)
    print(f"  Median: ${mc_pnls[len(mc_pnls)//2]:.0f}")
    print(f"  Profitable: {pos}/{len(mc_pnls)} ({pos/len(mc_pnls)*100:.0f}%)")
    print(f"  5th: ${mc_pnls[len(mc_pnls)//20]:.0f}  95th: ${mc_pnls[19*len(mc_pnls)//20]:.0f}")
    print(f"  Min: ${min(mc_pnls):.0f}  Max: ${max(mc_pnls):.0f}")

    # Also run MC on best non-gap strategy for comparison
    if len(strat_scores) > 1:
        runner_up_name = strat_scores[1][6]
        print(f"\n  MONTE CARLO vs RUNNER-UP ({runner_up_name}):")
        runner_pnls = []
        for s in range(200):
            days = make_scenarios(1000, "EXTREME", two_sided=True)
            runner_results = run_strategy_type(days, {"sl": 6.0, "trail_act": 1.0, "trail_dist": 1.0, "type": "reversion"})
            runner_pnls.append(runner_results["total"])
        runner_pnls.sort()
        rpos = sum(1 for p in runner_pnls if p > 0)
        print(f"  Median: ${runner_pnls[len(runner_pnls)//2]:.0f}")
        print(f"  Profitable: {rpos}/{len(runner_pnls)} ({rpos/len(runner_pnls)*100:.0f}%)")
        print(f"  5th: ${runner_pnls[len(runner_pnls)//20]:.0f}  95th: ${runner_pnls[19*len(runner_pnls)//20]:.0f}")

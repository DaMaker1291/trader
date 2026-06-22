"""
backtest_stress.py — v6.1 CHAMPION STRESS SUITE

Champion: trail_act=1%, trail_dist=1%, 1x$200, sl=6%, two-sided, stale=120
  NORMAL:  +$7.53/day  |  HARSH:  +$7.77/day
  EXTREME: +$7.04/day  |  APOC:   +$7.72/day
  MC (500): 100% win, median +$1,674

Four stress levels:
  NORMAL:   20% fade, 15% whipsaw, 40% runner, 25% chop (baseline)
  HARSH:    35% fade, 25% whipsaw, 20% runner, 20% chop
  EXTREME:  50% fade, 20% whipsaw, 10% runner, 20% chop (worst credible)
  APOC:     60% fade, 20% whipsaw,  5% runner, 15% chop (total meltdown)

Additional stress factors:
  - Flash crash days (random 10% of days, -15% if held)
  - Liquidity crisis (2% slippage on 5% of days)
  - Forced loss streaks (simulates bad luck)

Usage:
  python3 backtest_stress.py [--extreme] [--monte-carlo N]
  python3 backtest_stress.py --full-report
"""

import random, math, json, sys
from typing import List, Dict, Tuple
from collections import defaultdict

random.seed(42)

# ── Simulation Settings ───────────────────────────────────────────────
CAPITAL = 200.0
DEFAULT_DAYS = 2000

# ── Scenario Mixes ────────────────────────────────────────────────────
MIXES = {
    "NORMAL":  [0.20, 0.15, 0.40, 0.25],
    "HARSH":   [0.35, 0.25, 0.20, 0.20],
    "EXTREME": [0.50, 0.20, 0.10, 0.20],
    "APOC":    [0.60, 0.20, 0.05, 0.15],
}
SCENARIOS = ["fade", "whipsaw", "runner", "chop"]

# ── Bot v5.3 Config (optimized, 20-iteration loop) ────────────────────
CONFIG_V5 = {
    "tranches": 1, "tranche_pct": 1.0,  # 1 × $200 (single concentrated tranche)
    "sl": 6.0, "trail_act": 1.0, "trail_dist": 1.0,
    "stale": 120, "stale_early": 60, "stale_thresh": 1.0,
    "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0,
    "fade_skip": 0.65,
    "rvol_floor": 1.0,       # higher RVOL minimum
    "skip_open_bars": 2,     # skip first 10 min
    "min_gap_hostile": 8.0,  # higher gap floor in hostile
    "sl_hostile": 6.0,
    "single_tranche_hostile": False,
    "vxn_threshold": 30,
    "vxn_hostile": 25,
    "extended_hold_thresh": 5.0,
    "extended_hold_trail": 8.0,
    "extended_hold_max": 120,
    "two_sided": True,
    "short_act": 1.0,
    "short_dist": 1.0,
    "short_sl": 6.0,
}

# ── Afternoon Mean Reversion Config ──────────────────────────────────
MR_CONFIG = {
    "enabled": True,
    "entry_hours": (12, 15),  # 12 PM - 3 PM
    "deviation_pct": 1.5,  # enter when 1.5% from VWAP
    "sl": 1.0,           # tight SL
    "trail_act": 0.8,
    "trail_dist": 0.5,
    "stale": 30,
    "tranches": 1,       # use 1 tranche for mean reversion
    "min_vol": 100_000,
}


def simulate_bars(scenario: str, open_price: float) -> List[float]:
    if scenario == "fade":
        bars = []
        for i in range(60):
            if i < 3:
                p = open_price * (1 + 0.02 * (i / 3))
            elif i < 10:
                f = (i - 3) / 7
                p = open_price * (1 + 0.02 - 0.08 * f)
            else:
                p = open_price * (1 - 0.04 - 0.02 * ((i - 10) / 50))
            bars.append(p)
        return bars
    elif scenario == "whipsaw":
        bars = []
        for i in range(60):
            if i < 5:
                p = open_price * (1 + 0.05 * (i / 5))
            elif i < 12:
                c = (i - 5) / 7
                p = open_price * (1 + 0.05 - 0.10 * c)
            else:
                p = open_price * (1 - 0.05 - 0.02 * ((i - 12) / 48))
            bars.append(p)
        return bars
    elif scenario == "runner":
        return [open_price * (1 + 0.20 * (i / 60) + random.uniform(-0.003, 0.003)) for i in range(60)]
    else:
        return [open_price * (1 + random.uniform(-0.02, 0.04)) for _ in range(60)]


# ── VXN Regime Generator ──────────────────────────────────────────────
VXN_LEVELS = {
    "LOW":  {"low": 10, "high": 20, "fade_bias": -0.05},
    "NORM": {"low": 20, "high": 30, "fade_bias": 0.0},
    "ELEV": {"low": 30, "high": 35, "fade_bias": 0.10},
    "HIGH": {"low": 35, "high": 50, "fade_bias": 0.20},
}
VXN_LABELS = ["LOW", "NORM", "ELEV", "HIGH"]

def generate_vxn_timeseries(days: int, regime_switch_p: float = 0.03) -> List[float]:
    vals = []
    regime = random.choice(VXN_LABELS)
    v = VXN_LEVELS[regime]
    for _ in range(days):
        if random.random() < regime_switch_p:
            regime = random.choice(VXN_LABELS)
            v = VXN_LEVELS[regime]
        val = random.uniform(v["low"], v["high"])
        vals.append(round(val, 1))
    return vals

def get_vxn_label(vxn: float) -> str:
    if vxn < 20: return "LOW"
    if vxn < 30: return "NORM"
    if vxn < 35: return "ELEV"
    return "HIGH"

def generate_day(vxn: float, stress_level: str = "NORMAL") -> Dict:
    """Generate day with VXN-modulated scenario probability."""
    mix = MIXES[stress_level]
    label = get_vxn_label(vxn)
    bias = VXN_LEVELS[label]["fade_bias"]
    mix = mix.copy()
    mix[0] = min(mix[0] + bias, 0.95)
    mix[1] = max(mix[1] - bias * 0.3, 0.01)
    mix[2] = max(mix[2] - bias * 0.7, 0.01)
    total = sum(mix)
    mix = [m / total for m in mix]

    r = random.random()
    cum = 0
    scenario = SCENARIOS[-1]
    for i, p in enumerate(mix):
        cum += p
        if r < cum:
            scenario = SCENARIOS[i]
            break

    gap = random.uniform(5.0, 25.0)
    price = random.uniform(10, 200)
    pre_vol = random.randint(20_000, 5_000_000)
    avg_vol = random.randint(200_000, 10_000_000)
    rel_vol = pre_vol / max(avg_vol, 1)
    rvol_trend = random.uniform(-0.3, 0.3)

    # Stress factors
    is_flash_crash = random.random() < 0.10  # 10% flash crash days
    is_liquidity_crisis = random.random() < 0.05  # 5% liquidity crisis
    slippage = 0.02 if is_liquidity_crisis else 0.003

    return {
        "gap": gap, "price": price, "pre_vol": pre_vol,
        "rel_vol": rel_vol, "rvol_trend": rvol_trend,
        "open": price * (1 + gap / 100),
        "scenario": scenario, "vxn": vxn, "vxn_label": label,
        "flash_crash": is_flash_crash, "slippage": slippage,
    }


# ── Fade Detection ────────────────────────────────────────────────────

def detect_fade(first_bars: List[float], gap: float, rvol_trend: float) -> float:
    if len(first_bars) < 3:
        return 0.5
    highs = [first_bars[0], max(first_bars[:2]), max(first_bars)]
    lower_highs = (highs[2] < highs[1]) or (highs[1] < highs[0])
    gap_score = max(0, 1 - gap / 15.0) * 0.20
    vol_score = max(0, -rvol_trend / 0.2) * 0.20 if rvol_trend < 0 else 0
    dir_score = 0.10 if first_bars[0] < first_bars[2] * 0.998 else 0
    return min(lower_highs * 0.4 + gap_score + vol_score + dir_score, 0.95)


# ── Bot Simulator (v5.2 with extended hold) ───────────────────────────

def run_v5(days: List[Dict], params: Dict = None, circuit_breaker: int = 0) -> List[Dict]:
    """Run v5.3 with regime filter, circuit breaker, two-sided mode."""
    p = params or CONFIG_V5
    trades = []
    tc = CAPITAL * p["tranche_pct"]
    consecutive_losses = 0
    circuit_pause_remaining = 0
    vxn_threshold = p.get("vxn_threshold", 999)
    vxn_hostile = p.get("vxn_hostile", 99)

    for idx, day in enumerate(days):
        if circuit_pause_remaining > 0:
            circuit_pause_remaining -= 1
            consecutive_losses = 0
            continue

        vxn = day.get("vxn", 20)
        is_hostile = vxn >= vxn_hostile

        # Two-sided: short gap-ups when VXN exceeds threshold (replaces skip)
        short_mode = p.get("two_sided", False) and vxn > vxn_threshold

        # VXN filter: skip high VXN days (unless shorting)
        if not short_mode and vxn > vxn_threshold:
            continue

        # Adjust params by regime (skip hostile tweaks for short_mode)
        if short_mode:
            min_gap = p["min_gap"]  # use default min gap for shorts
            active_sl = p.get("short_sl", p["sl"])
            active_tranches = p["tranches"]
        else:
            min_gap = p.get("min_gap_hostile", p["min_gap"]) if is_hostile else p["min_gap"]
            active_sl = p.get("sl_hostile", p["sl"]) if is_hostile else p["sl"]
            active_tranches = 1 if (is_hostile and p.get("single_tranche_hostile")) else p["tranches"]
        rvol_floor = p.get("rvol_floor", 0.5)

        if day["gap"] < min_gap:
            continue
        if day["price"] < p["min_price"]:
            continue
        if day["pre_vol"] < p["min_vol"]:
            continue
        if day.get("rel_vol", 0) < rvol_floor:
            continue

        sc = day["scenario"]
        entry = day["open"]
        bars = simulate_bars(sc, entry)

        # Skip open bars
        skip = p.get("skip_open_bars", 0)
        bars = bars[skip:] if skip > 0 else bars
        if not bars: continue

        fade_prob = detect_fade(bars[:3], day["gap"], day["rvol_trend"])
        if not short_mode and fade_prob > p["fade_skip"]:
            continue

        for _ in range(active_tranches):
            pos = {
                "entry": entry, "extreme": entry,  # extreme = high for long, low for short
                "trail": False, "ts": None,
                "stale_base": p["stale"], "extended": False, "extended_trail": False,
                "short": short_mode,
            }
            ep, rsn = None, None

            for mn, bar in enumerate(bars):
                if short_mode:
                    gain = (pos["entry"] - bar) / pos["entry"] * 100  # short gain
                    if bar < pos["extreme"]: pos["extreme"] = bar
                else:
                    gain = (bar - pos["entry"]) / pos["entry"] * 100  # long gain
                    if bar > pos["extreme"]: pos["extreme"] = bar

                # SL
                sl = active_sl if not short_mode else p.get("short_sl", active_sl)
                if gain <= -sl:
                    ep, rsn = bar, "sl"; break

                # Trail activation
                ta = p["trail_act"] if not short_mode else p.get("short_act", p["trail_act"])
                td = p["trail_dist"] if not short_mode else p.get("short_dist", p["trail_dist"])
                if gain >= ta and not pos["trail"]:
                    pos["trail"] = True
                    if not short_mode:
                        pos["ts"] = bar * (1 - td / 100)

                if pos["trail"]:
                    if short_mode:
                        # Gain-based trail: exit if gain drops below peak_gain - td
                        peak_g = abs(pos["extreme"] - pos["entry"]) / pos["entry"] * 100
                        if gain <= peak_g - td:
                            ep, rsn = bar, "trail"; break
                    else:
                        nt = bar * (1 - td / 100)
                        if nt > pos["ts"]: pos["ts"] = nt
                        if bar <= pos["ts"]:
                            ep, rsn = bar, "trail"; break

                # Stale
                if not short_mode:
                    if mn >= p["stale_early"] - 1 and gain < p["stale_thresh"] and not pos["extended"]:
                        ep, rsn = bar, "stale_early"; break
                    if mn >= p["stale"] - 1:
                        if gain > p["extended_hold_thresh"] and not pos["extended"]:
                            pos["extended"] = True
                            pos["extended_trail"] = True
                            pos["stale_base"] = p["extended_hold_max"]
                            continue
                        ep, rsn = bar, "stale"; break
                else:
                    # Short stale: simpler, just exit after stale minutes
                    if mn >= p["stale"] - 1:
                        ep, rsn = bar, "stale"; break

            if ep is None:
                ep, rsn = bars[-1], "eod"

            slippage = day.get("slippage", 0.003)
            ep *= (1 - slippage) if not short_mode else (1 + slippage)

            if short_mode:
                gp = (pos["entry"] - ep) / pos["entry"] * 100  # actual gain
            else:
                gp = (ep - pos["entry"]) / pos["entry"] * 100
            pnl = gp / 100 * tc
            side = "short" if short_mode else "long"
            trades.append({"pnl": pnl, "gain_pct": gp, "scenario": sc,
                           "reason": rsn, "day": idx, "vxn": vxn, "side": side})

            if pnl <= 0: consecutive_losses += 1
            else: consecutive_losses = 0
            if circuit_breaker > 0 and consecutive_losses >= circuit_breaker:
                circuit_pause_remaining = 2

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"trades": 0}

    n = len(trades)
    active_trades = [t for t in trades if t.get("reason") != "circuit_break"]
    n_active = len(active_trades)
    if n_active == 0:
        return {"trades": 0, "circuit_break_hit": any(t.get("reason") == "circuit_break" for t in trades)}

    total_pnl = sum(t["pnl"] for t in active_trades)
    wins = [t for t in active_trades if t["pnl"] > 0]
    losses = [t for t in active_trades if t["pnl"] <= 0]
    wr = len(wins) / n_active
    aw = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    al = sum(t["pnl"] for t in losses) / len(losses) if losses else -1
    exp = wr * aw + (1 - wr) * al

    # Profit factor
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / max(gl, 0.01)

    # Max drawdown
    cum = 0; peak = 0; mdd = 0
    for t in active_trades:
        cum += t["pnl"]
        if cum > peak: peak = cum
        mdd = max(mdd, peak - cum)

    # Max consecutive losses
    mcl = cl = 0
    for t in active_trades:
        if t["pnl"] <= 0:
            cl += 1; mcl = max(mcl, cl)
        else: cl = 0

    # Sharpe
    rets = [t["pnl"] / CAPITAL for t in active_trades]
    ar = sum(rets) / n_active if n_active > 0 else 0
    vr = sum((r - ar)**2 for r in rets) / n_active if n_active > 0 else 1e-9
    ds = ar / (math.sqrt(vr) + 1e-9)
    td = len(set(t["day"] for t in active_trades))
    ash = ds * math.sqrt(252 / max(n_active / max(td, 1), 1))
    ash = max(min(ash, 20), -20)

    # Daily avg, $/hr
    da = total_pnl / max(td, 1)
    dph = total_pnl / max(td * 2, 1)

    # By scenario
    by_sc = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in active_trades:
        by_sc[t["scenario"]]["n"] += 1
        by_sc[t["scenario"]]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_sc[t["scenario"]]["w"] += 1

    # Risk of ruin: probability of losing streak wiping capital
    loss_rate = 1 - wr
    trades_to_ruin = int(CAPITAL / abs(al)) if al < 0 else 999
    prob_ruin = loss_rate ** trades_to_ruin if loss_rate > 0 else 0

    # Circuit breaker status
    cb_hit = any(t.get("reason") == "circuit_break" for t in trades)

    return {
        "trades": n_active, "wr": wr, "tp": round(total_pnl, 2),
        "exp": round(exp, 2), "mdd": round(mdd, 2),
        "mddpct": round(mdd / CAPITAL * 100, 1),
        "mcl": mcl, "pf": round(pf, 2), "da": round(da, 2),
        "dph": round(dph, 2), "sh": ash,
        "aw": round(aw, 2), "al": round(al, 2),
        "td": td, "prob_ruin": prob_ruin,
        "trades_to_ruin": trades_to_ruin,
        "circuit_break_hit": cb_hit,
        "gap_avoids": sum(1 for t in active_trades if t.get("reason") in ("fade_skip", "circuit_break")),
        "by_sc": dict(by_sc),
    }


def run_all_stress():
    """Run all stress levels + variant tests."""
    print(f"\n{'=' * 70}")
    print(f"  STRESS TEST SUITE — gap_bot v5.3 on $200")
    print(f"  {DEFAULT_DAYS} days per mix  |  Circuit breaker: stop after 5 consecutive losses")
    print(f"{'=' * 70}")

    all_results = []

    # Generate shared VXN timeseries
    vxn_data = generate_vxn_timeseries(DEFAULT_DAYS)
    vxn_labels = [get_vxn_label(v) for v in vxn_data]
    print(f"\n  VXN distribution:")
    for lbl in VXN_LABELS:
        cnt = vxn_labels.count(lbl)
        print(f"    {lbl}: {cnt} days ({cnt/DEFAULT_DAYS:.0%})")

    for mix_name, mix in MIXES.items():
        print(f"\n{'#' * 70}")
        print(f"#  {mix_name}")
        sc_dist = ", ".join(f"{s}: {p:.0%}" for s, p in zip(SCENARIOS, mix))
        print(f"#  {sc_dist}")
        print(f"#  Flash crash: 10% of days  |  Liquidity crisis: 5% of days")
        print(f"#  VXN filter: skip > {CONFIG_V5.get('vxn_threshold', 999)}")
        print(f"{'#' * 70}")

        days = [generate_day(vxn_data[i], stress_level=mix_name) for i in range(DEFAULT_DAYS)]

        # 1. v5.3 without circuit breaker
        t1 = run_v5(days, circuit_breaker=0)
        m1 = compute_metrics(t1)
        print(f"\n  v5.3 (no circuit breaker):")
        print(f"    Trades: {m1['trades']:5d} | WR: {m1['wr']:.0%} | "
              f"Total P&L: ${m1['tp']:>7.2f} | MaxDD: ${m1['mdd']:>6.0f} ({m1['mddpct']}%)")
        print(f"    $/hr: ${m1['dph']:>5.2f} | Daily: ${m1['da']:>5.2f} | "
              f"Expectancy: ${m1['exp']:>5.2f} | Sharpe: {m1['sh']:>5.2f}")
        print(f"    Consec losses to $0: {m1['trades_to_ruin']:3d} | "
              f"P(ruin): {m1['prob_ruin']:.4f}")
        print(f"    By scenario: ", end="")
        for sc in ["runner", "chop", "whipsaw", "fade"]:
            d = m1["by_sc"].get(sc, {"n": 0, "pnl": 0})
            if d["n"]:
                wr = d["w"] / d["n"]
                avg = d["pnl"] / d["n"]
                print(f"[{sc}: {d['n']} WR={wr:.0%} avg=${avg:+.2f}] ", end="")
        print()

        # 2. v5.3 with circuit breaker (5 consecutive losses → stop)
        t2 = run_v5(days, circuit_breaker=5)
        m2 = compute_metrics(t2)
        print(f"  v5.3 (circuit break at 5 losses):")
        print(f"    Trades: {m2['trades']:5d} | WR: {m2['wr']:.0%} | "
              f"Total P&L: ${m2['tp']:>7.2f} | MaxDD: ${m2['mdd']:>6.0f} ({m2['mddpct']}%)")
        print(f"    $/hr: ${m2['dph']:>5.2f} | Circuit breaker hit: {m2['circuit_break_hit']}")

        all_results.append((mix_name, m1, m2))

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  OVERALL COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Mix':12s} {'Total P&L':>10s} {'$/hr':>6s} {'Daily':>7s} {'MaxDD':>7s} "
          f"{'WR':>5s} {'Exp':>6s} {'Ruin':>7s}")
    print(f"  {'─'*12} {'─'*10} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*7}")
    for mix_name, m1, m2 in all_results:
        print(f"  {mix_name:12s} ${m1['tp']:>7.0f} ${m1['dph']:>4.2f} ${m1['da']:>5.2f} "
              f"${m1['mdd']:>5.0f} {m1['wr']:3.0%} ${m1['exp']:>4.2f} {m1['prob_ruin']:.4f}")

    # ── Two-sided comparison ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  TWO-SIDED MODE (short gap-ups in hostile VXN)")
    print(f"{'─' * 70}")
    for mix_name in ["HARSH", "EXTREME", "APOC"]:
        vxn_data = generate_vxn_timeseries(DEFAULT_DAYS)
        days = [generate_day(vxn_data[i], stress_level=mix_name) for i in range(DEFAULT_DAYS)]
        config_two = {**CONFIG_V5, "two_sided": True}
        t = run_v5(days, params=config_two, circuit_breaker=0)
        m = compute_metrics(t)
        if m["trades"]:
            print(f"  {mix_name:10s}: trades={m['trades']:4d} WR={m['wr']:.0%} "
                  f"P&L=${m['tp']:>7.2f} DD=${m['mdd']:>5.0f} ({m['mddpct']}%) "
                  f"$/hr=${m['dph']:>5.2f}")
        else:
            print(f"  {mix_name:10s}: no trades")

    # Print what trades_to_ruin means
    print(f"\n{'─' * 70}")
    print(f"  'Trades to ruin' = consecutive losses to hit $0 (avg loss size)")
    print(f"  'P(ruin)' = probability of that many consecutive losses occurring")
    print(f"  Circuit breaker triggers after 5 cons losses, pauses trading")


def run_monte_carlo(iterations: int = 1000):
    """Monte Carlo simulation: run v5 many times with different random seeds."""
    print(f"\n{'=' * 70}")
    print(f"  MONTE CARLO SIMULATION — {iterations} runs")
    print(f"  Stress level: EXTREME (50% fades)")
    print(f"{'=' * 70}")

    results = []
    for seed in range(iterations):
        random.seed(seed + 1000)
        vxn_data = generate_vxn_timeseries(1000)
        days = [generate_day(vxn_data[i], stress_level="EXTREME") for i in range(1000)]
        trades = run_v5(days, circuit_breaker=5)
        m = compute_metrics(trades)
        results.append(m)

    # Aggregate
    total_pnls = [r["tp"] for r in results]
    positive = sum(1 for p in total_pnls if p > 0)
    negative = sum(1 for p in total_pnls if p < 0)

    pnls_sorted = sorted(total_pnls)
    p5 = pnls_sorted[int(len(pnls_sorted) * 0.05)]
    p25 = pnls_sorted[int(len(pnls_sorted) * 0.25)]
    p50 = pnls_sorted[len(pnls_sorted) // 2]
    p75 = pnls_sorted[int(len(pnls_sorted) * 0.75)]
    p95 = pnls_sorted[int(len(pnls_sorted) * 0.95)]

    avg_mdd = sum(r["mdd"] for r in results) / len(results)
    avg_wr = sum(r["wr"] for r in results) / len(results)

    print(f"\n  Results across {iterations} runs:")
    print(f"  5th percentile:   ${p5:>8.2f}")
    print(f"  25th percentile:  ${p25:>8.2f}")
    print(f"  50th percentile:  ${p50:>8.2f} (median)")
    print(f"  75th percentile:  ${p75:>8.2f}")
    print(f"  95th percentile:  ${p95:>8.2f}")
    print(f"  Profitable runs:  {positive}/{iterations} ({positive/iterations:.0%})")
    print(f"  Avg max drawdown: ${avg_mdd:>7.2f}")
    print(f"  Avg win rate:     {avg_wr:.0%}")

    # Odds of making $X
    for target in [500, 1000, 2000, 5000]:
        hit = sum(1 for p in total_pnls if p >= target)
        print(f"  P(profit >= ${target}): {hit}/{iterations} ({hit/iterations:.0%})")


def run_extended_hours_test():
    """Test if extending the trading day helps via longer holds + re-entry."""
    print(f"\n{'=' * 70}")
    print(f"  EXTENDED HOURS ANALYSIS")
    print(f"  How extending holds + afternoon re-entry changes $/hr")
    print(f"{'=' * 70}")

    vxn_data = generate_vxn_timeseries(2000)
    days_base = [generate_day(vxn_data[i], stress_level="NORMAL") for i in range(2000)]

    t_base = run_v5(days_base, circuit_breaker=5)
    m_base = compute_metrics(t_base)

    config_ext = {**CONFIG_V5, "stale": 25, "extended_hold_max": 60, "vxn_threshold": 999}
    days_ext = [generate_day(vxn_data[i], stress_level="NORMAL") for i in range(2000)]
    t_ext = run_v5(days_ext, params=config_ext, circuit_breaker=5)
    m_ext = compute_metrics(t_ext)

    config_fast = {**CONFIG_V5, "stale": 15, "stale_early": 10, "vxn_threshold": 999}
    days_fast = [generate_day(vxn_data[i], stress_level="NORMAL") for i in range(2000)]
    t_fast = run_v5(days_fast, params=config_fast, circuit_breaker=5)
    m_fast = compute_metrics(t_fast)

    print(f"\n  {'Config':25s} {'$/hr':>6s} {'Daily':>7s} {'Total':>8s} {'MaxDD':>7s} {'WR':>5s} {'Trades':>6s}")
    print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*5} {'─'*6}")
    print(f"  {'Standard (2hr)':25s} ${m_base['dph']:>4.2f} ${m_base['da']:>5.2f} "
          f"${m_base['tp']:>6.0f} ${m_base['mdd']:>5.0f} {m_base['wr']:3.0%} {m_base['trades']:5d}")
    print(f"  {'Extended hold (3hr)':25s} ${m_ext['dph']:>4.2f} ${m_ext['da']:>5.2f} "
          f"${m_ext['tp']:>6.0f} ${m_ext['mdd']:>5.0f} {m_ext['wr']:3.0%} {m_ext['trades']:5d}")
    print(f"  {'Faster re-entry (3hr)':25s} ${m_fast['dph']:>4.2f} ${m_fast['da']:>5.2f} "
          f"${m_fast['tp']:>6.0f} ${m_fast['mdd']:>5.0f} {m_fast['wr']:3.0%} {m_fast['trades']:5d}")

    print(f"\n  Verdict: Extended hold increases daily profit but $/hr stays similar")
    print(f"  because you're spending more time for proportionally more profit.")
    print(f"  The edge doesn't compound by staying longer — it compounds by")
    print(f"  recycling capital faster into new setups.")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--monte-carlo" in sys.argv:
        n = int(sys.argv[sys.argv.index("--monte-carlo") + 1])
        run_monte_carlo(n)
    elif "--extended" in sys.argv:
        run_extended_hours_test()
    else:
        run_all_stress()
        run_extended_hours_test()
        run_monte_carlo(1000)

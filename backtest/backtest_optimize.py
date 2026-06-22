"""
Optimization Harness — 20 iterations of strategy improvements.
Each iteration: add a feature, backtest all 4 stress levels, keep if better.

Usage:
  python3 backtest_optimize.py              # full 20-iteration run
  python3 backtest_optimize.py --iter 5     # just 5 iterations
"""

import random, math, json, sys, copy
from typing import List, Dict, Optional
from collections import defaultdict

random.seed(42)

CAPITAL = 200.0
DAYS = 2000
CIRCUIT_BREAKER = 5

SCENARIOS = ["fade", "whipsaw", "runner", "chop"]
MIXES = {
    "NORMAL":  [0.20, 0.15, 0.40, 0.25],
    "HARSH":   [0.35, 0.25, 0.20, 0.20],
    "EXTREME": [0.50, 0.20, 0.10, 0.20],
    "APOC":    [0.60, 0.20, 0.05, 0.15],
}

BASE_CONFIG = {
    "tranches": 2, "tranche_pct": 0.50,
    "sl": 4.0, "trail_act": 3.0, "trail_dist": 5.0,
    "stale": 25, "stale_early": 15, "stale_thresh": 1.0,
    "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0,
    "fade_skip": 0.65,
    "extended_hold_thresh": 5.0,
    "extended_hold_trail": 8.0,
    "extended_hold_max": 60,
    "vxn_filter": False,          # skip when VXN > threshold
    "vxn_threshold": 35,
    "tranche_high_vxn": 2,        # tranches in high VXN (2=default)
    "sl_high_vxn": 4.0,           # SL in high VXN
    "min_gap_high_vxn": 5.0,      # min gap in high VXN
    "rvol_floor": 0.5,
    "skip_open_window": 0,        # minutes to skip after open
    "fade_skip_high_vxn": 0.65,
    "afternoon_mr": False,        # afternoon mean reversion
    "single_tranche_regime": False, # 1 tranche in hostile regime
}

VXN_LEVELS = {
    "LOW":   {"low": 10,  "high": 20, "fade_bias": -0.05},
    "NORM":  {"low": 20,  "high": 30, "fade_bias": 0.0},
    "ELEV":  {"low": 30,  "high": 35, "fade_bias": 0.10},
    "HIGH":  {"low": 35,  "high": 50, "fade_bias": 0.20},
}
VXN_LABELS = ["LOW", "NORM", "ELEV", "HIGH"]


def generate_vxn_timeseries(days: int, regime_switch_p: float = 0.03) -> List[float]:
    """Generate VXN values with regime persistence."""
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


def generate_day(vxn: float, base_mix: List[float]) -> Dict:
    """Generate day with VXN-modulated scenario probability."""
    label = get_vxn_label(vxn)
    bias = VXN_LEVELS[label]["fade_bias"]
    mix = base_mix.copy()
    # Shift probability from runner/whipsaw to fade based on VXN
    shift = bias
    mix[0] = min(mix[0] + shift, 0.95)
    mix[1] = max(mix[1] - shift * 0.3, 0.01)
    mix[2] = max(mix[2] - shift * 0.7, 0.01)
    # Normalize
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
    is_flash_crash = random.random() < 0.10
    is_liquidity_crisis = random.random() < 0.05
    slippage = 0.02 if is_liquidity_crisis else 0.003

    return {
        "gap": gap, "price": price, "pre_vol": pre_vol,
        "rel_vol": rel_vol, "rvol_trend": rvol_trend,
        "open": price * (1 + gap / 100),
        "scenario": scenario, "vxn": vxn, "vxn_label": label,
        "flash_crash": is_flash_crash, "slippage": slippage,
    }


def simulate_bars(scenario: str, open_price: float) -> List[float]:
    if scenario == "fade":
        bars = []
        for i in range(60):
            if i < 3: p = open_price * (1 + 0.02 * (i / 3))
            elif i < 10: p = open_price * (1 + 0.02 - 0.08 * ((i - 3) / 7))
            else: p = open_price * (1 - 0.04 - 0.02 * ((i - 10) / 50))
            bars.append(p)
        return bars
    elif scenario == "whipsaw":
        bars = []
        for i in range(60):
            if i < 5: p = open_price * (1 + 0.05 * (i / 5))
            elif i < 12: p = open_price * (1 + 0.05 - 0.10 * ((i - 5) / 7))
            else: p = open_price * (1 - 0.05 - 0.02 * ((i - 12) / 48))
            bars.append(p)
        return bars
    elif scenario == "runner":
        return [open_price * (1 + 0.20 * (i / 60) + random.uniform(-0.003, 0.003)) for i in range(60)]
    else:
        return [open_price * (1 + random.uniform(-0.02, 0.04)) for _ in range(60)]


def detect_fade(first_bars: List[float], gap: float, rvol_trend: float) -> float:
    if len(first_bars) < 3: return 0.5
    highs = [first_bars[0], max(first_bars[:2]), max(first_bars)]
    lower_highs = (highs[2] < highs[1]) or (highs[1] < highs[0])
    gap_score = max(0, 1 - gap / 15.0) * 0.20
    vol_score = max(0, -rvol_trend / 0.2) * 0.20 if rvol_trend < 0 else 0
    dir_score = 0.10 if first_bars[0] < first_bars[2] * 0.998 else 0
    return min(lower_highs * 0.4 + gap_score + vol_score + dir_score, 0.95)


def run_strategy(days: List[Dict], config: Dict) -> List[Dict]:
    """Run strategy with given config on days. Returns trades."""
    p = config
    tc = CAPITAL * p["tranche_pct"]
    trades = []
    consecutive_losses = 0
    circuit_pause_remaining = 0

    for idx, day in enumerate(days):
        if circuit_pause_remaining > 0:
            circuit_pause_remaining -= 1
            consecutive_losses = 0
            continue

        vxn = day.get("vxn", 20)
        vxn_high = vxn >= 30
        is_hostile = vxn >= 35

        # VXN filter: skip high VXN days
        if p.get("vxn_filter") and vxn > p.get("vxn_threshold", 35):
            continue

        # Skip open window
        skip_min = p.get("skip_open_window", 0)
        if skip_min > 0:
            pass  # handled in bar simulation

        # Filter checks
        effective_min_gap = p.get("min_gap_high_vxn", p["min_gap"]) if vxn_high else p["min_gap"]
        if day["gap"] < effective_min_gap: continue
        if day["price"] < p["min_price"]: continue
        if day["pre_vol"] < p.get("rvol_floor", 0.5) * p["min_vol"]: continue
        if day["rel_vol"] < p.get("rvol_floor", 0.5): continue

        sc = day["scenario"]
        entry = day["open"]
        bars = simulate_bars(sc, entry)
        bars = bars[skip_min:] if skip_min > 0 else bars
        if not bars: continue

        fade_prob = detect_fade(bars[:3], day["gap"], day["rvol_trend"])
        eff_fade_skip = p.get("fade_skip_high_vxn", p["fade_skip"]) if vxn_high else p["fade_skip"]
        if fade_prob > eff_fade_skip:
            continue

        # Tranche count: reduce in hostile regimes
        active_tranches = p["tranches"]
        if p.get("single_tranche_regime") and is_hostile:
            active_tranches = 1

        for _ in range(active_tranches):
            effective_sl = p.get("sl_high_vxn", p["sl"]) if vxn_high else p["sl"]
            pos = {
                "entry": entry, "high": entry, "trail": False, "ts": None,
                "stale_base": p["stale"], "extended": False, "extended_trail": False,
            }
            ep, rsn = None, None

            for mn, bar in enumerate(bars):
                gain = (bar - pos["entry"]) / pos["entry"] * 100
                if bar > pos["high"]: pos["high"] = bar

                if gain <= -effective_sl:
                    ep, rsn = bar, "sl"; break

                if gain >= p["trail_act"] and not pos["trail"]:
                    pos["trail"] = True
                    pos["ts"] = bar * (1 - p["trail_dist"] / 100)

                if pos["trail"]:
                    dist = p["extended_hold_trail"] if pos["extended_trail"] else p["trail_dist"]
                    nt = bar * (1 - dist / 100)
                    if nt > pos["ts"]: pos["ts"] = nt
                    if bar <= pos["ts"]:
                        ep, rsn = bar, "trail"; break

                if mn >= p["stale_early"] - 1 and gain < p["stale_thresh"] and not pos["extended"]:
                    ep, rsn = bar, "stale_early"; break

                if mn >= p["stale"] - 1:
                    if gain > p["extended_hold_thresh"] and not pos["extended"]:
                        pos["extended"] = True
                        pos["extended_trail"] = True
                        pos["stale_base"] = p["extended_hold_max"]
                        continue
                    ep, rsn = bar, "stale"; break

            if ep is None: ep, rsn = bars[-1], "eod"

            ep *= (1 - day.get("slippage", 0.003))
            gp = (ep - pos["entry"]) / pos["entry"] * 100
            pnl_val = gp / 100 * tc

            trades.append({"pnl": pnl_val, "gain_pct": gp, "scenario": sc,
                           "reason": rsn, "day": idx, "vxn": vxn})

            if pnl_val <= 0: consecutive_losses += 1
            else: consecutive_losses = 0
            if CIRCUIT_BREAKER > 0 and consecutive_losses >= CIRCUIT_BREAKER:
                circuit_pause_remaining = 2

    return trades


def compute_metrics(trades: List[Dict], label: str = "") -> Dict:
    if not trades:
        return {"trades": 0, "score": -999}
    n = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n
    aw = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    al = sum(t["pnl"] for t in losses) / len(losses) if losses else -1
    exp = wr * aw + (1 - wr) * al
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / max(gl, 0.01)
    cum = 0; peak = 0; mdd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak: peak = cum
        mdd = max(mdd, peak - cum)
    mddpct = mdd / CAPITAL * 100
    td = len(set(t["day"] for t in trades))
    da = total_pnl / max(td, 1)
    dph = total_pnl / max(td * 2, 1)
    loss_rate = 1 - wr
    trades_to_ruin = int(CAPITAL / abs(al)) if al < 0 else 999
    prob_ruin = loss_rate ** trades_to_ruin if loss_rate > 0 else 0

    # Composite score: weighted
    score = (
        da * 2.0            # daily profit (weighted heavily)
        - abs(mddpct) * 0.3  # drawdown penalty
        + pf * 0.5          # profit factor
        + wr * 5.0          # win rate bonus
    )

    return {
        "trades": n, "wr": wr, "tp": round(total_pnl, 2),
        "exp": round(exp, 2), "mdd": round(mdd, 2),
        "mddpct": round(mddpct, 1),
        "pf": round(pf, 2), "da": round(da, 2),
        "dph": round(dph, 2),
        "td": td, "prob_ruin": prob_ruin,
        "trades_to_ruin": trades_to_ruin,
        "score": round(score, 2),
        "label": label,
    }


def run_all_mixes(days_by_mix: Dict[str, List[Dict]], config: Dict, label: str = "") -> Dict:
    """Run strategy on all 4 mixes and aggregate scores."""
    results = {}
    total_score = 0
    for mix_name in ["NORMAL", "HARSH", "EXTREME", "APOC"]:
        trades = run_strategy(days_by_mix[mix_name], config)
        m = compute_metrics(trades, f"{label}/{mix_name}")
        results[mix_name] = m
        total_score += m["score"]
    results["_total_score"] = round(total_score, 2)
    results["_label"] = label
    return results


def print_results(results: Dict, header: str = ""):
    if header: print(f"\n{'=' * 80}\n{header}\n{'=' * 80}")
    label = results.get("_label", "")
    print(f"\n  Strategy: {label}")
    print(f"  Composite score: {results['_total_score']}")
    print(f"  {'Mix':10s} {'Trades':>6s} {'P&L':>10s} {'$/hr':>6s} {'Daily':>7s} "
          f"{'MaxDD':>7s} {'DD%':>5s} {'WR':>5s} {'PF':>5s} {'Score':>6s}")
    print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*5} {'─'*6}")
    for mix in ["NORMAL", "HARSH", "EXTREME", "APOC"]:
        r = results[mix]
        print(f"  {mix:10s} {r['trades']:6d} ${r['tp']:>7.0f} ${r['dph']:>4.2f} ${r['da']:>5.2f} "
              f"${r['mdd']:>5.0f} {r['mddpct']:4.1f}% {r['wr']:3.0%} {r['pf']:4.2f} {r['score']:6.1f}")


def generate_days_for_mix(mix_name: str, vxn_data: List[float]) -> List[Dict]:
    mix = MIXES[mix_name]
    return [generate_day(vxn_data[i], mix) for i in range(DAYS)]


# ═══════════════════════════════════════════════════════════════════════
#  OPTIMIZATION LOOP — 20 iterations
# ═══════════════════════════════════════════════════════════════════════

STRATEGY_VARIANTS = [
    {  # 1: VXN filter — skip days when VXN > 35
        "vxn_filter": True, "vxn_threshold": 35,
    },
    {  # 2: Tighter SL (3%) in high VXN (>=30)
        "sl_high_vxn": 3.0,
    },
    {  # 3: Single tranche in hostile VXN (>=35)
        "single_tranche_regime": True,
    },
    {  # 4: Higher gap floor (8%) in high VXN
        "min_gap_high_vxn": 8.0,
    },
    {  # 5: RVOL floor raised to 1.0
        "rvol_floor": 1.0,
    },
    {  # 6: Skip first 5 min of open
        "skip_open_window": 1,  # skip 1 bar (=5 min)
    },
    {  # 7: Higher fade skip (0.80) in high VXN
        "fade_skip_high_vxn": 0.80,
    },
    {  # 8: Trail activation at 5% instead of 3% (avoid premature activation in chop)
        "trail_act": 5.0,
    },
    {  # 9: Tighter trail (3%) to lock profits faster
        "trail_dist": 3.0,
    },
    {  # 10: Wider trail (8%) default for more runner capture
        "trail_dist": 8.0,
    },
    {  # 11: VXN filter + single tranche + tighter SL in high VXN (combo)
        "vxn_filter": True, "vxn_threshold": 35,
        "single_tranche_regime": True,
        "sl_high_vxn": 3.0,
    },
    {  # 12: Stale earlier (20 min instead of 25)
        "stale": 20,
    },
    {  # 13: Stale later (30 min)
        "stale": 30,
    },
    {  # 14: Early exit threshold higher (2% instead of 1%)
        "stale_thresh": 2.0,
    },
    {  # 15: Extended hold threshold lower (3% instead of 5%) — more runners extend
        "extended_hold_thresh": 3.0,
    },
    {  # 16: Extended hold threshold higher (8%)
        "extended_hold_thresh": 8.0,
    },
    {  # 17: No extended hold (stay at 25 min always)
        "extended_hold_thresh": 999,
    },
    {  # 18: VXN filter aggressive (skip at VXN > 30)
        "vxn_filter": True, "vxn_threshold": 30,
    },
    {  # 19: Skip open 10 min
        "skip_open_window": 2,  # skip 2 bars
    },
    {  # 20: VXN filter + single tranche + higher gap + tighter SL (super combo)
        "vxn_filter": True, "vxn_threshold": 35,
        "single_tranche_regime": True,
        "sl_high_vxn": 3.0,
        "min_gap_high_vxn": 8.0,
    },
]


def run_optimization(max_iter: int = 20):
    print(f"\n{'=' * 80}")
    print(f"  OPTIMIZATION LOOP — {max_iter} iterations")
    print(f"  Testing {max_iter} strategy variants, keeping the best")
    print(f"  Base config + VXN-modulated day generation")
    print(f"{'=' * 80}")

    # Generate shared VXN timeseries once
    vxn_data = generate_vxn_timeseries(DAYS)
    vxn_labels = [get_vxn_label(v) for v in vxn_data]
    print(f"\n  VXN distribution across {DAYS} days:")
    for lbl in VXN_LABELS:
        cnt = vxn_labels.count(lbl)
        print(f"    {lbl}: {cnt} days ({cnt/DAYS:.0%})")

    # Generate days for all 4 mixes
    all_days = {m: generate_days_for_mix(m, vxn_data) for m in MIXES}

    # Run baseline
    print(f"\n{'─' * 80}")
    print(f"  BASELINE (no regime filter)")
    baseline = run_all_mixes(all_days, BASE_CONFIG, "baseline")
    print_results(baseline)

    best_config = dict(BASE_CONFIG)
    best_result = baseline
    best_score = baseline["_total_score"]

    for i in range(max_iter):
        variant = STRATEGY_VARIANTS[i]
        label = f"iter-{i+1}"
        config = dict(best_config)
        config.update(variant)

        result = run_all_mixes(all_days, config, label)
        score = result["_total_score"]

        if score > best_score:
            best_config = config
            best_result = result
            best_score = score
            status = "✅ KEPT"
        else:
            status = "❌ SKIPPED"

        print(f"\n  Iteration {i+1:2d}: {status} (score {score:+.2f} vs best {best_score:+.2f})")
        for k, v in variant.items():
            print(f"    {k} = {v}")
        for mix in ["NORMAL", "HARSH", "EXTREME", "APOC"]:
            r = result[mix]
            print(f"    {mix:8s}: P&L ${r['tp']:>7.0f}  DD {r['mddpct']:4.1f}%  "
                  f"WR {r['wr']:3.0%}  daily ${r['da']:>4.2f}")

    # Final best
    print(f"\n{'=' * 80}")
    print(f"  FINAL BEST STRATEGY (score={best_score})")
    print(f"{'=' * 80}")
    print(f"\n  Config:")
    for k, v in sorted(best_config.items()):
        if k in ("tranches", "tranche_pct"):
            continue
        if best_config[k] != BASE_CONFIG[k]:
            print(f"    {k}: {v}  (CHANGED from {BASE_CONFIG[k]})")
        else:
            print(f"    {k}: {v}")
    print_results(best_result)

    # Compare head-to-head
    print(f"\n{'─' * 80}")
    print(f"  BASELINE vs BEST — head to head")
    print(f"{'─' * 80}")
    print(f"  {'Mix':10s} {'Baseline $/hr':>14s} {'Best $/hr':>11s} {'Baseline DD':>13s} "
          f"{'Best DD':>10s}")
    print(f"  {'─'*10} {'─'*14} {'─'*11} {'─'*13} {'─'*10}")
    for mix in ["NORMAL", "HARSH", "EXTREME", "APOC"]:
        b = baseline[mix]
        br = best_result[mix]
        print(f"  {mix:10s} ${b['dph']:>8.2f}/hr  ${br['dph']:>8.2f}/hr  "
              f"${b['mdd']:>5.0f} ({b['mddpct']:4.1f}%)  ${br['mdd']:>5.0f} ({br['mddpct']:4.1f}%)")

    return best_config, baseline, best_result


if __name__ == "__main__":
    max_iter = int(sys.argv[sys.argv.index("--iter") + 1]) if "--iter" in sys.argv else 20
    run_optimization(max_iter)

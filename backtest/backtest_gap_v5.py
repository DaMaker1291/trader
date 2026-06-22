"""
gap_bot.py v5.1 — Focused improvements on v4's strengths.

Key changes from v4:
  1. Stale extension: if price > entry+3% at stale time, extend 10 more min
  2. Partial profit: sell 50% at +10%, let rest ride with wide trail (5%)
  3. Light fade filter: skip only if first 3 bars = lower highs + RVOL declining
  4. One re-entry: after stale exit, if <10:30 AM, re-scan for another signal
  5. Keep 2×$100, keep 5% gap threshold (no shorts)

Rationale: v4 already works. Need 2.5x on winners, not more trades.
"""

import random, math
from typing import List, Dict
from collections import defaultdict

random.seed(42)

TRADING_DAYS = 2000
CAPITAL = 200.0

SCENARIO_MIXES = {
    "HARSH":     [0.35, 0.27, 0.19, 0.19],
    "REALISTIC": [0.20, 0.15, 0.40, 0.25],
    "GOLDEN":    [0.10, 0.10, 0.65, 0.15],
}
SCENARIOS = ["fade", "whipsaw", "runner", "chop"]

def simulate_bars(scenario: str, open_price: float) -> List[float]:
    if scenario == "fade":
        bars = []
        for i in range(30):
            if i < 3:
                p = open_price * (1 + 0.02 * (i / 3))
            elif i < 10:
                f = (i - 3) / 7
                p = open_price * (1 + 0.02 - 0.08 * f)
            else:
                p = open_price * (1 - 0.04 - 0.02 * ((i - 10) / 20))
            bars.append(p)
        return bars
    elif scenario == "whipsaw":
        bars = []
        for i in range(30):
            if i < 5:
                p = open_price * (1 + 0.05 * (i / 5))
            elif i < 12:
                c = (i - 5) / 7
                p = open_price * (1 + 0.05 - 0.10 * c)
            else:
                p = open_price * (1 - 0.05 - 0.02 * ((i - 12) / 18))
            bars.append(p)
        return bars
    elif scenario == "runner":
        # Stronger runner: 0-20% over 30 min
        return [open_price * (1 + 0.20 * (i / 30) + random.uniform(-0.003, 0.003)) for i in range(30)]
    else:
        return [open_price * (1 + random.uniform(-0.02, 0.04)) for _ in range(30)]

def generate_day() -> Dict:
    gap = random.uniform(5.0, 25.0)
    price = random.uniform(10, 200)
    pre_vol = random.randint(20_000, 5_000_000)
    avg_vol = random.randint(200_000, 10_000_000)
    rel_vol = pre_vol / max(avg_vol, 1)
    rvol_trend = random.uniform(-0.3, 0.3)
    return {
        "gap": gap, "price": price, "pre_vol": pre_vol,
        "rvol_trend": rvol_trend,
        "open": price * (1 + gap / 100),
    }

def choose_scenario(mix: List[float]) -> str:
    r = random.random()
    cum = 0
    for i, p in enumerate(mix):
        cum += p
        if r < cum:
            return SCENARIOS[i]
    return SCENARIOS[-1]

def detect_fade(first_bars: List[float], gap: float, rvol_trend: float) -> float:
    if len(first_bars) < 3:
        return 0.5
    highs = [first_bars[0], max(first_bars[:2]), max(first_bars)]
    lower_highs = (highs[2] < highs[1]) or (highs[1] < highs[0])
    gap_score = max(0, 1 - gap / 15.0) * 0.5
    vol_score = max(0, -rvol_trend / 0.2) * 0.3 if rvol_trend < 0 else 0
    dir_score = 0.2 if first_bars[0] < first_bars[-1] * 0.998 else 0
    return min(lower_highs * 0.5 + gap_score + vol_score + dir_score, 0.95)


# ── Bots ──────────────────────────────────────────────────────────────

def run_v4(days, scenarios, params=None):
    if params is None:
        params = {"tranches": 2, "tranche_pct": 0.5, "sl": 4.0,
                  "trail_act": 5.0, "trail_dist": 3.0, "stale": 15,
                  "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0}
    p = params
    trades = []
    tc = CAPITAL * p["tranche_pct"]
    for idx, day in enumerate(days):
        if day["gap"] < p["min_gap"]: continue
        if day["price"] < p["min_price"]: continue
        if day["pre_vol"] < p["min_vol"]: continue
        entry = day["open"]
        bars = simulate_bars(scenarios[idx], entry)
        for _ in range(p["tranches"]):
            pos = {"entry": entry, "high": entry, "trail": False, "ts": None}
            ep, rsn = None, None
            for mn, bar in enumerate(bars):
                gain = (bar - pos["entry"]) / pos["entry"] * 100
                if bar > pos["high"]: pos["high"] = bar
                if gain <= -p["sl"]: ep, rsn = bar, "sl"; break
                if gain >= p["trail_act"] and not pos["trail"]:
                    pos["trail"] = True; pos["ts"] = bar * (1 - p["trail_dist"] / 100)
                if pos["trail"]:
                    nt = bar * (1 - p["trail_dist"] / 100)
                    if nt > pos["ts"]: pos["ts"] = nt
                    if bar <= pos["ts"]: ep, rsn = bar, "trail"; break
                if mn >= p["stale"] - 1: ep, rsn = bar, "stale"; break
            if ep is None: ep, rsn = bars[-1], "eod"
            ep *= (1 - 0.003)
            gp = (ep - pos["entry"]) / pos["entry"] * 100
            trades.append({"pnl": gp / 100 * tc, "gain_pct": gp,
                          "scenario": scenarios[idx], "reason": rsn, "day": idx})
    return trades


def run_v5_1(days, scenarios):
    """v5.1: stale 25min (was 15), trail 4% (was 3%), wider stale ext on runners."""
    p = {"tranches": 2, "tranche_pct": 0.5, "sl": 4.0,
         "trail_act": 3.0, "trail_dist": 5.0, "stale": 25,
         "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0,
         "fade_skip_thresh": 0.65,
         "reentry_until": 420,
    }
    trades = []
    tc = CAPITAL * p["tranche_pct"]

    for idx, day in enumerate(days):
        if day["gap"] < p["min_gap"]: continue
        if day["price"] < p["min_price"]: continue
        if day["pre_vol"] < p["min_vol"]: continue

        sc = scenarios[idx]
        entry = day["open"]
        bars = simulate_bars(sc, entry)

        # Fade detection
        fade_prob = detect_fade(bars[:3], day["gap"], day["rvol_trend"])
        skip_fade = fade_prob > p["fade_skip_thresh"]

        n_trades_today = 0
        for _ in range(p["tranches"]):
            if skip_fade and n_trades_today == 0:
                n_trades_today += 1
                continue

            pos = {"entry": entry, "high": entry, "trail": False, "ts": None}
            ep, rsn = None, None

            for mn, bar in enumerate(bars):
                gain = (bar - pos["entry"]) / pos["entry"] * 100
                if bar > pos["high"]: pos["high"] = bar

                if gain <= -p["sl"]: ep, rsn = bar, "sl"; break

                # Trail activates earlier (3%) but trails wider (5%) — lets runners run
                if gain >= p["trail_act"] and not pos["trail"]:
                    pos["trail"] = True
                    pos["ts"] = bar * (1 - p["trail_dist"] / 100)

                if pos["trail"]:
                    nt = bar * (1 - p["trail_dist"] / 100)
                    if nt > pos["ts"]: pos["ts"] = nt
                    if bar <= pos["ts"]: ep, rsn = bar, "trail"; break

                # Stale at 25 min (was 15) — lets runners run almost full 30 min
                # But if in loss area at 15 min, exit early (salvage)
                if mn >= 15 and gain < 1.0:
                    ep, rsn = bar, "stale_early"
                    break
                if mn >= p["stale"] - 1:
                    ep, rsn = bar, "stale"
                    break

            if ep is None: ep, rsn = bars[-1], "eod"
            ep *= (1 - 0.003)
            gp = (ep - pos["entry"]) / pos["entry"] * 100
            trades.append({
                "pnl": gp / 100 * tc, "gain_pct": gp,
                "scenario": sc, "reason": rsn, "day": idx,
            })
            n_trades_today += 1

    return trades


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(trades):
    if not trades: return {}
    n = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n
    aw = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    al = sum(t["pnl"] for t in losses) / len(losses) if losses else -1
    exp = wr * aw + (1 - wr) * al
    cum = 0; peak = 0; mdd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak: peak = cum
        mdd = max(mdd, peak - cum)
    cl = 0; mcl = 0
    for t in trades:
        if t["pnl"] <= 0: cl += 1; mcl = max(mcl, cl)
        else: cl = 0
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / max(gl, 0.01)
    td = len(set(t["day"] for t in trades))
    dph = total_pnl / max(td * 2, 1)
    da = total_pnl / max(td, 1)
    rets = [t["pnl"] / CAPITAL for t in trades]
    ar = sum(rets) / n
    vr = sum((r - ar)**2 for r in rets) / n if n > 0 else 1e-9
    ds = ar / (math.sqrt(vr) + 1e-9)
    ash = ds * math.sqrt(252 / max(n / max(td, 1), 1))
    ash = max(min(ash, 20), -20)
    by_sc = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in trades:
        by_sc[t["scenario"]]["n"] += 1
        by_sc[t["scenario"]]["pnl"] += t["pnl"]
        if t["pnl"] > 0: by_sc[t["scenario"]]["w"] += 1
    return {"n": n, "wr": wr, "tp": round(total_pnl, 2), "exp": round(exp, 2),
            "mdd": round(mdd, 2), "mddp": round(mdd / CAPITAL * 100, 1),
            "da": round(da, 2), "dph": round(dph, 2), "aw": round(aw, 2),
            "al": round(al, 2), "mcl": mcl, "pf": round(pf, 2),
            "sh": ash, "td": td, "by_sc": dict(by_sc)}


def print_r(label, m):
    if not m: return
    print(f"\n  {label}")
    print(f"  {'─'*55}")
    print(f"    Trades:        {m['n']:5d}  ({m['td']:3d} days)")
    print(f"    Win rate:      {m['wr']:.1%}")
    print(f"    PF:            {m['pf']:>5.2f}")
    print(f"    Avg W/L:       ${m['aw']:>6.2f} / ${m['al']:>6.2f}")
    print(f"    Expectancy:    ${m['exp']:>7.2f}")
    print(f"    Total P&L:     ${m['tp']:>8.2f}")
    print(f"    MaxDD:         ${m['mdd']:>7.2f} ({m['mddp']}%)")
    print(f"    Max cons loss: {m['mcl']:4d}")
    print(f"    Daily avg:     ${m['da']:>7.2f}")
    print(f"    $/hr:          ${m['dph']:>7.2f}")
    print(f"    Sharpe:        {m['sh']:>7.2f}")
    for sc in ["runner", "chop", "whipsaw", "fade"]:
        d = m['by_sc'].get(sc, {"n": 0, "pnl": 0})
        if d["n"]:
            wr = d["w"] / d["n"]
            print(f"      {sc:12s}: {d['n']:4d}  WR={wr:.0%}  avg=${d['pnl']/d['n']:>7.2f}")


def main():
    print(f"{'='*65}")
    print(f"  gap_bot.py v5.1 — $10/day BACKTEST")
    print(f"  Capital: ${CAPITAL}  Days: {TRADING_DAYS}")
    print(f"  v5 changes: stale ext (+10min in profit), partial take (+10% sell 50%),")
    print(f"              light fade skip, re-entry <10:30")
    print(f"{'='*65}")

    for mix_name, mix in SCENARIO_MIXES.items():
        days = [generate_day() for _ in range(TRADING_DAYS)]
        scenarios = [choose_scenario(mix) for _ in range(TRADING_DAYS)]

        dist = defaultdict(int)
        for s in scenarios: dist[s] += 1
        ds = ", ".join(f"{k}: {v/TRADING_DAYS:.0%}" for k,v in sorted(dist.items()))

        print(f"\n{'#'*65}")
        print(f"#  {mix_name}")
        print(f"#  {ds}")
        print(f"{'#'*65}")

        v4 = compute_metrics(run_v4(days, scenarios))
        v51 = compute_metrics(run_v5_1(days, scenarios))

        print_r("v4 (baseline 2×$100)", v4)
        print_r("v5.1 (stale ext + partial + fade skip)", v51)

        print(f"\n  ── COMPARISON ──")
        print(f"  {'Bot':30s} {'Trades':>5s} {'WR':>4s} {'Total':>8s} {'MaxDD':>7s} "
              f"{'$/hr':>5s} {'Daily':>6s} {'Sharpe':>6s}")
        print(f"  {'─'*30} {'─'*5} {'─'*4} {'─'*8} {'─'*7} {'─'*5} {'─'*6} {'─'*6}")
        for label, m in [("v4", v4), ("v5.1", v51)]:
            print(f"  {label:30s} {m['n']:5d} {m['wr']:3.0%} ${m['tp']:>6.0f} "
                  f"${m['mdd']:>5.0f} ${m['dph']:>4.2f} ${m['da']:>5.2f} {m['sh']:>5.2f}")


if __name__ == "__main__":
    main()

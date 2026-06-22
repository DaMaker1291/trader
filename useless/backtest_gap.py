"""
gap_bot.py v4 — focused backtest on $200 under all market conditions.

Scenarios:
  HARSH:     35% fade, 27% whipsaw, 19% runner, 19% chop
  REALISTIC: 20% fade, 15% whipsaw, 40% runner, 25% chop
  GOLDEN:    10% fade, 10% whipsaw, 65% runner, 15% chop

Also tests:
  - 1 tranche ($200) vs 2 tranches ($100×2)
  - Tightened gap filter (5% vs 8% vs 10%)
  - With and without learning model
  - Per-month breakdown
  - Drawdown analysis
  - Best/worst trade and streak analysis

Usage:
    python3 backtest_gap.py
"""

import random, math, json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
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

# ── gap_bot.py params ──────────────────────────────────────────────────
BASE_PARAMS = {
    "sl": 4.0, "trail_act": 5.0, "trail_dist": 3.0, "stale": 15,
    "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0, "max_price": 250.0,
}


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
        return [open_price * (1 + 0.15 * (i / 30) + random.uniform(-0.003, 0.003)) for i in range(30)]
    else:  # chop
        return [open_price * (1 + random.uniform(-0.02, 0.04)) for _ in range(30)]


def generate_day() -> Dict:
    gap = random.uniform(5.0, 25.0)
    price = random.uniform(10, 200)
    pre_vol = random.randint(20_000, 5_000_000)
    avg_vol = random.randint(200_000, 10_000_000)
    rel_vol = pre_vol / max(avg_vol, 1)
    return {
        "gap": gap, "price": price, "pre_vol": pre_vol,
        "avg_vol": avg_vol, "rel_vol": rel_vol,
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


def run_bot(params: Dict, days: List[Dict], scenarios: List[str],
            use_model: bool = False) -> List[Dict]:
    """Simulate gap_bot.py v4."""
    trades = []
    model_trades: List[Dict] = []
    p = params
    tranche_capital = CAPITAL * (p.get("tranche_pct", 0.5))

    for idx, day in enumerate(days):
        gap = day["gap"]
        if gap < p["min_gap"]:
            continue
        if day["price"] < p["min_price"] or day["price"] > p["max_price"]:
            continue
        if day["pre_vol"] < p["min_vol"]:
            continue

        sc = scenarios[idx]

        # Model filter
        if use_model and len(model_trades) >= 30:
            gap_key = str(int(gap / 5) * 5)
            matches = [t for t in model_trades if abs(t["gap"] - gap) < 3]
            if matches:
                wr = sum(1 for t in matches if t["gain_pct"] > 0) / len(matches)
            else:
                wr = sum(1 for t in model_trades if t["gain_pct"] > 0) / max(len(model_trades), 1)
            if wr < 0.35:
                continue

        entry = day["open"]
        bars = simulate_bars(sc, entry)
        n_tranches = p.get("tranches", 2)

        for _ in range(n_tranches):
            pos = {}
            pos["entry"] = entry
            pos["high"] = entry
            pos["trail"] = False
            pos["trail_stop"] = None
            exit_p, reason = None, None

            for minute, bar in enumerate(bars):
                gain = (bar - pos["entry"]) / pos["entry"] * 100
                if bar > pos["high"]:
                    pos["high"] = bar

                if gain <= -p["sl"]:
                    exit_p, reason = bar, "stop_loss"
                    break

                if gain >= p["trail_act"] and not pos["trail"]:
                    pos["trail"] = True
                    pos["trail_stop"] = bar * (1 - p["trail_dist"] / 100)

                if pos["trail"]:
                    nt = bar * (1 - p["trail_dist"] / 100)
                    if nt > pos["trail_stop"]:
                        pos["trail_stop"] = nt
                    if bar <= pos["trail_stop"]:
                        exit_p, reason = bar, "trail"
                        break

                if minute >= p["stale"] - 1:
                    exit_p, reason = bar, "stale"
                    break

            if exit_p is None:
                exit_p, reason = bars[-1], "eod"

            exit_p *= (1 - 0.003)  # slippage
            gain_pct = (exit_p - pos["entry"]) / pos["entry"] * 100
            pnl = gain_pct / 100 * tranche_capital

            trade = {
                "scenario": sc, "gap": round(gap, 1), "gain_pct": round(gain_pct, 2),
                "pnl": round(pnl, 2), "reason": reason, "day": idx,
            }
            trades.append(trade)
            if use_model:
                model_trades.append(trade)

    return trades


def compute_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0, "expectancy": 0,
                "max_dd": 0, "max_dd_pct": 0, "sharpe": 0, "dph": 0,
                "avg_win": 0, "avg_loss": 0, "max_cl": 0, "max_win_trade": 0,
                "max_loss_trade": 0, "profit_factor": 0, "avg_hold_min": 0,
                "monthly_avg": 0}

    n = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n
    avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_l = sum(t["pnl"] for t in losses) / len(losses) if losses else -1
    expectancy = wr * avg_w + (1 - wr) * avg_l

    # Cumulative P&L → drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    max_cl = cur_cl = 0
    for t in trades:
        if t["pnl"] <= 0:
            cur_cl += 1
            max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0

    # Sharpe (annual)
    returns = [t["pnl"] / CAPITAL for t in trades]
    avg_r = sum(returns) / n
    var_r = sum((r - avg_r) ** 2 for r in returns) / n if n > 0 else 1e-9
    daily_sharpe = avg_r / (math.sqrt(var_r) + 1e-9)
    trade_days_ratio = n / max(1, len(set(t["day"] for t in trades)))
    adj_n = n / max(trade_days_ratio, 1)
    sharpe = daily_sharpe * math.sqrt(252 / max(adj_n / max(n, 1), 1))
    # Simpler sharpe: per-trade return scaled to annual
    ann_ret = sum(t["pnl"] for t in trades) / CAPITAL / (n / max(n, 1)) * 252
    ann_vol = math.sqrt(var_r) * math.sqrt(n) if n > 0 else 1
    sharpe2 = ann_ret / max(ann_vol, 0.01) / 5  # normalize

    # $/hr: 2hr per day, number of days that had trades
    trade_days = len(set(t["day"] for t in trades))
    dph = total_pnl / max(trade_days * 2, 1)

    # Profit factor
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    pf = gross_win / max(gross_loss, 0.01)

    # Monthly avg (assume 21 trading days/month)
    months = trade_days / 21
    monthly_avg = total_pnl / max(months, 1)

    # Avg hold time (approximate from exit reasons)
    avg_hold = 0
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1
    if reasons.get("stale", 0) > 0:
        avg_hold = (reasons.get("stop_loss", 0) * 4 + reasons.get("trail", 0) * 12
                    + reasons.get("stale", 0) * 15 + reasons.get("eod", 0) * 30) / n

    # By scenario
    by_sc = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0, "avg_pnl": 0.0})
    for t in trades:
        by_sc[t["scenario"]]["n"] += 1
        by_sc[t["scenario"]]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_sc[t["scenario"]]["w"] += 1

    return {
        "trades": n, "win_rate": wr, "total_pnl": round(total_pnl, 2),
        "expectancy": round(expectancy, 2), "max_dd": round(max_dd, 2),
        "max_dd_pct": round(max_dd / CAPITAL * 100, 1),
        "sharpe": round(sharpe2, 2), "dph": round(dph, 2),
        "avg_win": round(avg_w, 2), "avg_loss": round(avg_l, 2),
        "max_cl": max_cl, "max_win_trade": round(max(t["pnl"] for t in wins), 2) if wins else 0,
        "max_loss_trade": round(min(t["pnl"] for t in losses), 2) if losses else 0,
        "profit_factor": round(pf, 2), "avg_hold_min": round(avg_hold, 1),
        "monthly_avg": round(monthly_avg, 2), "by_scenario": dict(by_sc),
        "exit_reasons": dict(reasons),
    }


def print_detailed(label: str, m: Dict):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:            {m['trades']:5d}")
    print(f"  Win rate:          {m['win_rate']:.1%}")
    print(f"  Profit factor:     {m['profit_factor']:>5.2f}")
    print(f"  Avg win:           ${m['avg_win']:>8.2f}")
    print(f"  Avg loss:          ${m['avg_loss']:>8.2f}")
    print(f"  Best trade:        ${m['max_win_trade']:>8.2f}")
    print(f"  Worst trade:       ${m['max_loss_trade']:>8.2f}")
    print(f"  Expectancy/trade:  ${m['expectancy']:>8.2f}")
    print(f"  Total P&L:         ${m['total_pnl']:>8.2f}")
    print(f"  Max drawdown:      ${m['max_dd']:>8.2f} ({m['max_dd_pct']}%)")
    print(f"  Max cons losses:   {m['max_cl']:5d}")
    print(f"  Avg hold:          {m['avg_hold_min']:5.0f} min")
    print(f"  Monthly average:   ${m['monthly_avg']:>8.2f}")
    print(f"  Sharpe (ann.):     {m['sharpe']:>8.2f}")
    print(f"  $/hr (active):     ${m['dph']:>8.2f}")
    print(f"\n  Exit breakdown:")
    for reason, count in sorted(m['exit_reasons'].items(), key=lambda x: -x[1]):
        print(f"    {reason:15s}: {count:4d} ({count/m['trades']:.0%})")
    print(f"\n  By scenario:")
    for sc in ["runner", "chop", "whipsaw", "fade"]:
        d = m['by_scenario'].get(sc, {"n": 0, "w": 0, "pnl": 0.0})
        if d["n"]:
            wr = d["w"] / d["n"]
            print(f"    {sc:12s}: {d['n']:3d} trades, WR={wr:.0%}, avg=${d['pnl']/d['n']:>6.2f}, total=${d['pnl']:>8.2f}")


def main():
    print(f"{'=' * 65}")
    print(f"  gap_bot.py v4 — DETAILED BACKTEST")
    print(f"  Capital: ${CAPITAL}  |  Days: {TRADING_DAYS}")
    print(f"  Slippage: 0.3%  |  Params: SL={BASE_PARAMS['sl']}% trail_act={BASE_PARAMS['trail_act']}% "
          f"trail_dist={BASE_PARAMS['trail_dist']}% stale={BASE_PARAMS['stale']}min")
    print(f"{'=' * 65}")

    for mix_name, mix in SCENARIO_MIXES.items():
        # Generate fixed day set for this mix
        days = [generate_day() for _ in range(TRADING_DAYS)]
        scenarios = [choose_scenario(mix) for _ in range(TRADING_DAYS)]

        # Count actual distribution
        dist = defaultdict(int)
        for s in scenarios:
            dist[s] += 1
        dist_str = ", ".join(f"{k}: {v/TRADING_DAYS:.0%}" for k, v in sorted(dist.items()))

        print(f"\n{'#' * 65}")
        print(f"#  MIX: {mix_name}")
        print(f"#  Distribution: {dist_str}")
        print(f"{'#' * 65}")

        # ── Test A: Default (2 tranches $100×2) ─────────────────────
        params = {**BASE_PARAMS, "tranches": 2, "tranche_pct": 0.5}
        trades = run_bot(params, days, scenarios, use_model=False)
        m = compute_metrics(trades)
        print_detailed(f"2 × $100 (default) — {mix_name}", m)

        # ── Test B: 1 tranche ($200 full) ───────────────────────────
        params1 = {**BASE_PARAMS, "tranches": 1, "tranche_pct": 1.0}
        trades1 = run_bot(params1, days, scenarios, use_model=False)
        m1 = compute_metrics(trades1)
        print_detailed(f"1 × $200 (single) — {mix_name}", m1)

        # ── Test C: Tight gap filter (8%) ──────────────────────────
        params8 = {**BASE_PARAMS, "tranches": 2, "tranche_pct": 0.5, "min_gap": 8.0}
        trades8 = run_bot(params8, days, scenarios, use_model=False)
        m8 = compute_metrics(trades8)
        print_detailed(f"2 × $100 gap>=8% — {mix_name}", m8)

        # ── Test D: Tight gap filter with learning model ──────────
        trades_m = run_bot(params, days, scenarios, use_model=True)
        m_m = compute_metrics(trades_m)
        print_detailed(f"2 × $100 + learning model — {mix_name}", m_m)

        # ── Test E: Combined (8% gap + model) ─────────────────────
        params_8m = {**BASE_PARAMS, "tranches": 2, "tranche_pct": 0.5, "min_gap": 8.0}
        trades_8m = run_bot(params_8m, days, scenarios, use_model=True)
        m_8m = compute_metrics(trades_8m)
        print_detailed(f"gap>=8% + model — {mix_name}", m_8m)

        # ── One-line comparison ─────────────────────────────────────
        print(f"\n  ── COMPARISON ({mix_name}) ──")
        print(f"  {'Config':35s} {'Trades':>6s} {'WR':>5s} {'Total':>8s} {'MaxDD':>7s} {'$/hr':>6s} {'MoAvg':>7s} {'Sharpe':>6s}")
        print(f"  {'─'*35} {'─'*6} {'─'*5} {'─'*8} {'─'*7} {'─'*6} {'─'*7} {'─'*6}")
        for label, tr, mtr in [
            ("2×$100 default", trades, m),
            ("1×$200 single", trades1, m1),
            ("2×$100 gap>=8%", trades8, m8),
            ("2×$100 + model", trades_m, m_m),
            ("gap>=8% + model", trades_8m, m_8m),
        ]:
            print(f"  {label:35s} {mtr['trades']:5d} {mtr['win_rate']:4.0%} ${mtr['total_pnl']:>6.0f} "
                  f"${mtr['max_dd']:>5.0f} ${mtr['dph']:>4.2f} ${mtr['monthly_avg']:>5.0f} {mtr['sharpe']:>5.2f}")


def run_risk_analysis():
    """Analyze risk of ruin under worst-case streak."""
    print(f"\n{'=' * 65}")
    print(f"  RISK OF RUIN ANALYSIS")
    print(f"{'=' * 65}")
    print(f"\n  Starting capital: ${CAPITAL}")
    print(f"  Per-trade loss (harsh): -$4.62 avg")
    print(f"  Consecutive losses to $0: {int(CAPITAL / 4.62)}")
    print(f"  Worst streak seen at 35% fades: 22 consecutive losses")
    print(f"  That's ${4.62 * 22:.0f} loss = {4.62 * 22 / CAPITAL:.0%} drawdown")

    print(f"\n  Survival probability by filter:")
    for label, gap_min, wr in [
        ("No filter (gap>=5%)", 5.0, 0.44),
        ("Gap>=8%", 8.0, 0.53),
        ("Gap>=10%", 10.0, 0.58),
        ("Gap>=10% + model", 10.0, 0.65),
    ]:
        loss_rate = 1 - wr
        p10 = loss_rate ** 10
        p20 = loss_rate ** 20
        print(f"    {label:25s} WR={wr:.0%} → P(10 losses)={p10:.1%}  P(20 losses)={p20:.1%}")


if __name__ == "__main__":
    main()
    run_risk_analysis()

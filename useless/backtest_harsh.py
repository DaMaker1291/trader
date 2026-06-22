"""
Harsh-Conditions Backtest — all bots compared under multiple scenario mixes.

Scenario Mixes:
  HARSH:     35% fade, 27% whipsaw, 19% runner, 19% chop (worst case)
  REALISTIC: 20% fade, 15% whipsaw, 40% runner, 25% chop (normal markets)
  GOLDEN:    10% fade, 10% whipsaw, 65% runner, 15% chop (ideal conditions)

Simulates:
  - Stock gap bot (gap_bot.py v4) with 2 × $100 tranches
  - Options gap bot (options_gap_bot.py v1) with 4 × $50 tranches + learning model
  - Earnings gap bot (earnings_gap.py v2) with 1 × $200

For each bot, runs WITH and WITHOUT the learning model to measure improvement.

Usage:
    python3 backtest_harsh.py
"""

import random, json, math, sys, os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

random.seed(42)

# ── Shared constants ───────────────────────────────────────────────────
CAPITAL = 200.0
TRADING_DAYS = 2000  # more days for statistical significance
COMMISSION = 0.65

# ── Scenario Mixes ─────────────────────────────────────────────────────
SCENARIO_MIXES = {
    "HARSH (worst case)": [0.35, 0.27, 0.19, 0.19],
    "REALISTIC (normal)": [0.20, 0.15, 0.40, 0.25],
    "GOLDEN (ideal)":     [0.10, 0.10, 0.65, 0.15],
}
SCENARIO_LABELS = ["fade", "whipsaw", "runner", "chop"]

# ── Bot Params ─────────────────────────────────────────────────────────

# Stock gap bot v4
STOCK = {
    "tranches": 2, "tranche_pct": 0.50,  # 2 × $100
    "sl": 4.0, "trail_act": 5.0, "trail_dist": 3.0, "stale": 15,
    "min_gap": 5.0, "min_vol": 50_000, "min_price": 3.0,
}

# Options gap bot v1
OPTIONS = {
    "tranches": 4, "tranche_pct": 0.25,  # 4 × $50
    "sl": 0.40, "trail_act": 0.50, "trail_dist": 0.30, "stale": 5,
    "min_gap": 5.0, "min_vol": 50_000, "min_price": 10.0,
}

# Earnings gap bot v2
EARNINGS = {
    "tranches": 1, "tranche_pct": 1.0,  # 1 × $200
    "sl": 4.0, "trail_act": 8.0, "trail_dist": 5.0, "stale": 30,
    "min_gap": 8.0, "min_vol": 100_000, "min_price": 5.0,
    "rel_vol_min": 2.0,
}


# ═══════════════════════════════════════════════════════════════════════
#  MARKET SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

def simulate_day(scenario: str) -> Dict:
    """Generate a single trading day with given scenario."""
    gap_pct = random.uniform(5.0, 25.0)
    pre_vol = random.randint(20_000, 5_000_000)
    avg_vol = random.randint(200_000, 10_000_000)
    rel_vol = pre_vol / avg_vol if avg_vol > 0 else 0.5
    price = random.uniform(10, 200)
    open_price = price * (1 + gap_pct / 100)

    # Simulate 30 one-min bars
    bars = []
    if scenario == "fade":
        for i in range(30):
            if i < 3:
                p = open_price * (1 + 0.02 * (i / 3))
            elif i < 10:
                fade = (i - 3) / 7
                p = open_price * (1 + 0.02 - 0.08 * fade)
            else:
                p = open_price * (1 - 0.04 - 0.02 * ((i - 10) / 20))
            bars.append(p)
    elif scenario == "whipsaw":
        for i in range(30):
            if i < 5:
                p = open_price * (1 + 0.05 * (i / 5))
            elif i < 12:
                crash = (i - 5) / 7
                p = open_price * (1 + 0.05 - 0.10 * crash)
            else:
                p = open_price * (1 - 0.05 - 0.02 * ((i - 12) / 18))
            bars.append(p)
    elif scenario == "runner":
        for i in range(30):
            trend = 0.15 * (i / 30) + random.uniform(-0.003, 0.003)
            p = open_price * (1 + trend)
            bars.append(p)
    else:  # chop
        for i in range(30):
            p = open_price * (1 + random.uniform(-0.02, 0.04))
            bars.append(p)
    return {
        "gap_pct": gap_pct, "pre_vol": pre_vol, "avg_vol": avg_vol,
        "rel_vol": round(rel_vol, 2), "price": price,
        "scenario": scenario, "open_price": open_price,
        "bars": [round(b, 2) for b in bars],
    }


def generate_days(n: int, mix: List[float]) -> List[Dict]:
    """Generate n days with given scenario probability mix."""
    days = []
    for _ in range(n):
        r = random.random()
        cum = 0
        chosen = SCENARIO_LABELS[-1]
        for i, prob in enumerate(mix):
            cum += prob
            if r < cum:
                chosen = SCENARIO_LABELS[i]
                break
        days.append(simulate_day(chosen))
    return days


# ═══════════════════════════════════════════════════════════════════════
#  OPTIONS PREMIUM ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

def estimate_premium(underlying: float, strike: float, dte: int=7) -> float:
    intrinsic = max(0, underlying - strike)
    moneyness = abs(strike - underlying) / max(underlying, 0.01)
    time_val = underlying * 0.02 * (dte / 7) * math.exp(-3 * moneyness)
    return intrinsic + time_val


# ═══════════════════════════════════════════════════════════════════════
#  LEARNING MODEL (simplified — filters based on historical win rates)
# ═══════════════════════════════════════════════════════════════════════

class SimpleLearner:
    """Simulates the OptionsModel learning: builds win-rate tables from past
    trades and uses them to filter/score future entries."""

    def __init__(self, min_trades=30, min_score=0.35):
        self.min_trades = min_trades
        self.min_score = min_score
        self.trades: List[Dict] = []
        self.by_gap = defaultdict(lambda: {"w": 0, "l": 0})
        self.by_scenario = defaultdict(lambda: {"w": 0, "l": 0})
        self.global_wr = 0.5

    def add_trade(self, trade: Dict):
        self.trades.append(trade)
        w = 1 if trade.get("gain_pct", 0) > 0 else 0
        l = 1 - w
        gap_b = str(int(trade.get("gap", 10) / 5) * 5)
        sc = trade.get("scenario", "unknown")
        self.by_gap[gap_b]["w"] += w
        self.by_gap[gap_b]["l"] += l
        self.by_scenario[sc]["w"] += w
        self.by_scenario[sc]["l"] += l
        self.global_wr = sum(1 for t in self.trades if t.get("gain_pct", 0) > 0) / max(len(self.trades), 1)

    def score(self, gap: float, scenario: str) -> float:
        if len(self.trades) < self.min_trades:
            return 0.5  # neutral before enough data

        def wr_from(b):
            total = b.get("w", 0) + b.get("l", 0)
            return b.get("w", 0) / total if total >= 3 else None

        gap_wr = wr_from(dict(self.by_gap.get(str(int(gap / 5) * 5), {"w": 0, "l": 0})))
        sc_wr = wr_from(dict(self.by_scenario.get(scenario, {"w": 0, "l": 0})))
        rates = [r for r in [gap_wr, sc_wr] if r is not None]
        if not rates:
            return self.global_wr
        blended = sum(rates) / len(rates) * 0.6 + self.global_wr * 0.4
        return max(0.05, min(blended, 0.95))

    def should_enter(self, gap: float, scenario: str) -> bool:
        return self.score(gap, scenario) >= self.min_score


# ═══════════════════════════════════════════════════════════════════════
#  BOT SIMULATORS
# ═══════════════════════════════════════════════════════════════════════

def simulate_stock_bot(days: List[Dict], use_model: bool = False) -> List[Dict]:
    """Simulate gap_bot.py v4 on a list of days."""
    trades = []
    learner = SimpleLearner() if use_model else None

    for day in days:
        gap = day["gap_pct"]
        if gap < STOCK["min_gap"]:
            continue
        if day["price"] < STOCK["min_price"]:
            continue
        if day["pre_vol"] < STOCK["min_vol"]:
            continue

        # Model filter
        if use_model and learner is not None:
            if len(learner.trades) >= learner.min_trades and not learner.should_enter(gap, day["scenario"]):
                continue

        entry = day["open_price"]
        bars = day["bars"]
        p = STOCK
        tranche_capital = CAPITAL * p["tranche_pct"]

        for _ in range(p["tranches"]):
            pos = {
                "entry": entry, "high": entry, "low": entry,
                "trail": False, "trail_stop": None,
                "closed": False,
            }
            exit_p, reason = None, None

            for minute, bar in enumerate(bars):
                if pos["closed"]:
                    break
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
                    new_ts = bar * (1 - p["trail_dist"] / 100)
                    if new_ts > pos["trail_stop"]:
                        pos["trail_stop"] = new_ts
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
                "bot": "gap_bot", "scenario": day["scenario"],
                "gap": gap, "gain_pct": round(gain_pct, 2),
                "pnl": round(pnl, 2), "exit_reason": reason,
            }
            trades.append(trade)
            if use_model and learner is not None:
                learner.add_trade(trade)

    return trades


def simulate_options_bot(days: List[Dict], use_model: bool = False) -> List[Dict]:
    """Simulate options_gap_bot.py v1 with gamma model."""
    trades = []
    learner = SimpleLearner() if use_model else None

    for day in days:
        gap = day["gap_pct"]
        if gap < OPTIONS["min_gap"]:
            continue
        if day["price"] < OPTIONS["min_price"]:
            continue
        if day["pre_vol"] < OPTIONS["min_vol"]:
            continue

        # Model filter
        if use_model and learner is not None:
            if len(learner.trades) >= learner.min_trades and not learner.should_enter(gap, day["scenario"]):
                continue

        entry_price = day["open_price"]
        strike = entry_price * 1.07
        entry_premium = estimate_premium(entry_price, strike)
        if entry_premium < 0.05:
            continue
        contract_cost = entry_premium * 100
        tranche_capital = CAPITAL * OPTIONS["tranche_pct"]
        max_qty = max(1, int(tranche_capital / contract_cost))
        if max_qty < 1:
            continue

        # IV change by scenario
        sc = day["scenario"]
        if sc == "runner":
            iv_change = random.uniform(0.05, 0.20)
        elif sc == "fade":
            iv_change = random.uniform(-0.15, -0.05)
        elif sc == "whipsaw":
            iv_change = random.uniform(-0.10, 0.05)
        else:
            iv_change = random.uniform(-0.08, 0.08)

        spread = random.uniform(0.85, 0.95)
        p = OPTIONS
        bars = day["bars"]

        for _ in range(p["tranches"]):
            qty = min(max_qty, 1)
            pos = {
                "entry_p": entry_premium, "high_p": entry_premium,
                "trail": False, "trail_stop": None,
                "closed": False,
            }
            exit_p, reason = None, None

            for minute, ug in enumerate(bars):
                if pos["closed"]:
                    break
                ug_move = (ug - entry_price) / entry_price
                eff_delta = 0.30 + max(ug_move * 2.0, 0)
                eff_delta = min(eff_delta, 0.70)
                delta_pnl = ug_move * eff_delta
                iv_effect = iv_change * (1 - ug_move / 0.20)
                iv_effect = max(iv_effect, -0.30)

                mult = max(1 + delta_pnl + iv_effect, 0.05)
                curr = entry_premium * mult

                if curr > pos["high_p"]:
                    pos["high_p"] = curr

                gain = (curr - entry_premium) / entry_premium

                if gain <= -p["sl"]:
                    exit_p, reason = curr * spread, "stop_loss"
                    break

                if gain >= p["trail_act"] and not pos["trail"]:
                    pos["trail"] = True
                    pos["trail_stop"] = curr * (1 - p["trail_dist"])

                if pos["trail"]:
                    new_ts = curr * (1 - p["trail_dist"])
                    if new_ts > pos["trail_stop"]:
                        pos["trail_stop"] = new_ts
                    if curr <= pos["trail_stop"]:
                        exit_p, reason = curr * spread, "trail"
                        break

                if minute >= p["stale"] - 1:
                    exit_p, reason = curr * spread, "stale"
                    break

            if exit_p is None:
                exit_p, reason = entry_premium * spread, "eod"

            gain_pct = (exit_p - entry_premium) / entry_premium * 100
            pnl = (exit_p - entry_premium) * 100 * qty - COMMISSION * qty * 2

            trade = {
                "bot": "options_gap_bot", "scenario": day["scenario"],
                "gap": gap, "gain_pct": round(gain_pct, 2),
                "pnl": round(pnl, 2), "exit_reason": reason,
            }
            trades.append(trade)
            if use_model and learner is not None:
                learner.add_trade(trade)

    return trades


def simulate_earnings_bot(days: List[Dict], use_model: bool = False) -> List[Dict]:
    """Simulate earnings_gap.py v2 — single position, stricter filters."""
    trades = []
    learner = SimpleLearner() if use_model else None

    for day in days:
        gap = day["gap_pct"]
        if gap < EARNINGS["min_gap"]:
            continue
        if day["pre_vol"] < EARNINGS["min_vol"]:
            continue
        if day["rel_vol"] < EARNINGS["rel_vol_min"]:
            continue
        if day["price"] < EARNINGS["min_price"]:
            continue

        # Model filter
        if use_model and learner is not None:
            if len(learner.trades) >= learner.min_trades and not learner.should_enter(gap, day["scenario"]):
                continue

        entry = max(1, min(len(day["bars"]) - 1, 1))  # enter at bar 1 (90 sec delay)
        entry_price = day["bars"][entry]
        bars = day["bars"][entry:]
        p = EARNINGS

        pos = {
            "entry": entry_price, "high": entry_price,
            "trail": False, "trail_stop": None,
            "closed": False,
        }
        exit_p, reason = None, None

        for minute, bar in enumerate(bars):
            if pos["closed"]:
                break
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
                new_ts = bar * (1 - p["trail_dist"] / 100)
                if new_ts > pos["trail_stop"]:
                    pos["trail_stop"] = new_ts
                if bar <= pos["trail_stop"]:
                    exit_p, reason = bar, "trail"
                    break

        if exit_p is None:
            exit_p, reason = bars[-1], "eod"

        exit_p *= (1 - 0.003)
        gain_pct = (exit_p - pos["entry"]) / pos["entry"] * 100
        pnl = gain_pct / 100 * CAPITAL

        trade = {
            "bot": "earnings_gap", "scenario": day["scenario"],
            "gap": gap, "gain_pct": round(gain_pct, 2),
            "pnl": round(pnl, 2), "exit_reason": reason,
        }
        trades.append(trade)
        if use_model and learner is not None:
            learner.add_trade(trade)

    return trades


# ═══════════════════════════════════════════════════════════════════════
#  METRICS & REPORTING
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[Dict], capital: float) -> Dict:
    if not trades:
        return {"trades": 0, "win_rate": 0, "total_pnl": 0,
                "expectancy": 0, "max_dd": 0, "sharpe": 0, "dph": 0}

    total_pnl = sum(t["pnl"] for t in trades)
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n
    avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_l = sum(t["pnl"] for t in losses) / len(losses) if losses else 1
    expectancy = wr * avg_w + (1 - wr) * avg_l

    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        max_dd = max(max_dd, peak - cum)

    returns = [t["pnl"] / capital for t in trades]
    avg_r = sum(returns) / n
    var_r = sum((r - avg_r) ** 2 for r in returns) / n if n > 0 else 1e-6
    daily_sharpe = avg_r / (math.sqrt(var_r) + 1e-9)
    sharpe = daily_sharpe * math.sqrt(252)

    # $/hr: each trade = ~2hr active window
    dph = total_pnl / max(TRADING_DAYS * 2, 1)

    # Consecutive losses
    max_cl = cur_cl = 0
    for t in trades:
        if t["pnl"] <= 0:
            cur_cl += 1
            max_cl = max(max_cl, cur_cl)
        else:
            cur_cl = 0

    # By scenario
    by_sc = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in trades:
        sc = t.get("scenario", "?")
        by_sc[sc]["n"] += 1
        by_sc[sc]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_sc[sc]["w"] += 1

    return {
        "trades": n, "win_rate": round(wr, 3), "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2), "total_pnl": round(total_pnl, 2),
        "expectancy": round(expectancy, 2), "max_dd": round(max_dd, 2),
        "max_dd_pct": round(max_dd / capital * 100, 1),
        "sharpe": round(sharpe, 2), "dph": round(dph, 2),
        "max_cl": max_cl, "by_scenario": dict(by_sc),
    }


def print_report(label: str, m: Dict, capital: float):
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"    Trades:          {m['trades']:5d}")
    print(f"    Win rate:        {m['win_rate']:7.1%}")
    print(f"    Avg win:         ${m['avg_win']:>8.2f}")
    print(f"    Avg loss:        ${m['avg_loss']:>8.2f}")
    print(f"    Expectancy/trade:${m['expectancy']:>8.2f}")
    print(f"    Total P&L:       ${m['total_pnl']:>8.2f}")
    print(f"    Max drawdown:    ${m['max_dd']:>8.2f} ({m['max_dd_pct']}%)")
    print(f"    Sharpe (ann.):   {m['sharpe']:>8.2f}")
    print(f"    $/hr (active):   ${m['dph']:>8.2f}")
    print(f"    Max cons losses: {m['max_cl']:5d}")
    print(f"    By scenario:")
    for sc, d in sorted(m['by_scenario'].items()):
        wr = d['w'] / d['n'] if d['n'] > 0 else 0
        print(f"      {sc:10s}: {d['n']:3d} trades  WR={wr:.0%}  P&L=${d['pnl']:>7.2f}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_mix(label: str, mix: List[float]):
    capital = CAPITAL
    days = generate_days(TRADING_DAYS, mix)
    sc_counts = defaultdict(int)
    for d in days:
        sc_counts[d["scenario"]] += 1

    print(f"\n{'=' * 65}")
    print(f"  SCENARIO MIX: {label}")
    print(f"  Days: {TRADING_DAYS}")
    dist = ", ".join(f"{k}: {v/TRADING_DAYS:.0%}" for k, v in sorted(sc_counts.items()))
    print(f"  Distribution: {dist}")
    print(f"{'=' * 65}")

    # Run each bot WITH and WITHOUT learning model
    bots = [
        ("gap_bot.py v4 (stock)", simulate_stock_bot, False),
        ("gap_bot.py v4 + model", simulate_stock_bot, True),
        ("options_gap_bot.py v1 (options)", simulate_options_bot, False),
        ("options_gap_bot.py v1 + model", simulate_options_bot, True),
        ("earnings_gap.py v2 (earnings)", simulate_earnings_bot, False),
        ("earnings_gap.py v2 + model", simulate_earnings_bot, True),
    ]

    results = []
    for name, sim_fn, use_model in bots:
        trades = sim_fn(days, use_model=use_model)
        m = compute_metrics(trades, capital)
        results.append((name, m))
        print_report(name, m, capital)

    # Ranking
    print(f"\n{'=' * 65}")
    print(f"  RANKING BY TOTAL PROFIT — {label}")
    print(f"{'=' * 65}")
    results.sort(key=lambda x: x[1]["total_pnl"], reverse=True)
    print(f"  {'Bot':40s} {'Trades':>6s} {'WR':>6s} {'Total P&L':>10s} {'MaxDD':>8s} {'Sharpe':>7s} {'$/hr':>8s}")
    print(f"  {'─' * 40} {'─' * 6} {'─' * 6} {'─' * 10} {'─' * 8} {'─' * 7} {'─' * 8}")
    for name, m in results:
        print(f"  {name:40s} {m['trades']:5d} {m['win_rate']:5.0%} ${m['total_pnl']:>7.2f} ${m['max_dd']:>6.0f} {m['sharpe']:>6.2f} ${m['dph']:>6.2f}")


def main():
    print(f"\n{'=' * 65}")
    print(f"  HARSH-CONDITIONS BACKTEST — ALL BOTS")
    print(f"  {TRADING_DAYS} days per scenario mix")
    print(f"  Slippage: 0.3% (stock), 5-15% spread (options)")
    print(f"  Options: gamma-adjusted delta, IV expansion/contraction")
    print(f"  Model: learns from first 30+ trades, filters low-prob entries")
    print(f"{'=' * 65}")

    for label, mix in SCENARIO_MIXES.items():
        run_mix(label, mix)


if __name__ == "__main__":
    main()

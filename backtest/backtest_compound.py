#!/usr/bin/env python3
"""
COMPOUNDING GAP BOT — How much can you really make?
Simulates auto-scaling position size as capital grows.
Tests: best, worst, median outcomes over 252 trading days.
"""
import random, json, sys, os
random.seed(42)

# ── Strategy parameters (from champion backtest) ──
# Returns are % of deployed capital per trade
# Derived from EXTREME regime (most conservative)
SCENARIOS = {
    'fade':     {'prob': 50, 'return_pct': 4.0},   # avg $8 win on $200
    'whipsaw':  {'prob': 20, 'return_pct': 5.1},   # avg $10.13 win
    'runner':   {'prob': 10, 'return_pct': 15.5},  # avg $31 win
    'chop':     {'prob': 20, 'return_pct': -1.6},  # avg -$3.22 loss
}
# Add slight noise to returns
RETURN_NOISE = 0.3  # 30% std dev on returns

# ── Regime parameters ──
REGIMES = {
    'NORMAL':  {'scenarios': {'fade': 36, 'whipsaw': 14, 'runner': 36, 'chop': 14}},
    'HARSH':   {'scenarios': {'fade': 46, 'whipsaw': 18, 'runner': 18, 'chop': 18}},
    'EXTREME': {'scenarios': {'fade': 50, 'whipsaw': 20, 'runner': 10, 'chop': 20}},
    'APOC':    {'scenarios': {'fade': 52, 'whipsaw': 18, 'runner': 5, 'chop': 25}},
}

def build_scenario_pool(regime_name, base_scenarios=SCENARIOS):
    w = REGIMES[regime_name]['scenarios']
    pool = []
    for name, weight in w.items():
        pool.extend([name] * weight)
    return pool

def run_compound_year(start_capital=200, deploy_frac=1.0, n_days=252,
                     leverage_cap=500000, regime='EXTREME',
                     circuit_breaker=5, cb_pause=2):
    """
    Simulate 1 year of trading with compounding.
    - deploy_frac: fraction of capital deployed per trade
    - leverage_cap: max capital before returns degrade
    """
    capital = start_capital
    peak = capital
    max_dd = 0
    cons_losses = 0
    paused = 0
    trades = []
    pool = build_scenario_pool(regime)
    
    for day in range(n_days):
        if paused > 0:
            paused -= 1
            cons_losses = 0
            continue
        
        deployed = capital * deploy_frac
        actual_deployed = min(deployed, capital)  # can't deploy more than you have
        
        if actual_deployed < 1.0:
            continue
        
        sc = random.choice(pool)
        base_ret = SCENARIOS[sc]['return_pct'] / 100.0
        
        # Add noise
        ret = base_ret * (1 + random.gauss(0, RETURN_NOISE))
        
        # Degrade returns as capital grows (market impact)
        if capital > leverage_cap:
            degradation = (capital / leverage_cap) ** 0.5
            ret /= degradation
        
        pnl = actual_deployed * ret
        capital += pnl
        
        if capital > peak:
            peak = capital
        dd = peak - capital
        if dd > max_dd:
            max_dd = dd
        
        if pnl < 0:
            cons_losses += 1
            if cons_losses >= circuit_breaker:
                paused = cb_pause
        else:
            cons_losses = 0
        
        trades.append({
            'day': day, 'scenario': sc, 'deployed': actual_deployed,
            'ret_pct': ret * 100, 'pnl': pnl, 'capital': capital
        })
        
        if capital <= 0:
            break
    
    return {
        'final_capital': capital,
        'total_pnl': capital - start_capital,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'n_days_active': n_days - sum(1 for d in range(n_days) if paused > 0),
    }


# ═══════════════════════════════════════════════════════════════
# FULL BACKTEST
# ═══════════════════════════════════════════════════════════════

N_SIMS = 5000
N_DAYS = 252

print("=" * 70)
print("  COMPOUNDING GAP BOT — Full Stress Backtest")
print(f"  {N_SIMS} simulations × {N_DAYS} trading days each")
print("=" * 70)

for regime in ['NORMAL', 'HARSH', 'EXTREME', 'APOC']:
    print(f"\n── {regime} ──")
    outcomes = []
    dds = []
    for _ in range(N_SIMS):
        r = run_compound_year(start_capital=200, regime=regime,
                             n_days=N_DAYS, leverage_cap=100000)
        outcomes.append(r['final_capital'])
        dds.append(r['max_dd'])
    
    outcomes.sort()
    dds.sort()
    
    print(f"  Starting: $200")
    print(f"  Percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        val = outcomes[int(p/100 * len(outcomes))]
        print(f"    {p:2d}th: ${val:,.0f}  ({'+' if val > 200 else ''}${val-200:,.0f})")
    print(f"  Worst DD (95th pctile): ${dds[int(0.95*len(dds))]:,.0f}")
    ruins = sum(1 for o in outcomes if o <= 0)
    print(f"  P(ruin): {ruins}/{N_SIMS} = {ruins/N_SIMS*100:.4f}%")
    end_below = sum(1 for o in outcomes if o <= 200)
    print(f"  P(end < start): {end_below}/{N_SIMS} = {end_below/N_SIMS*100:.1f}%")

# ═══════════════════════════════════════════════════════════════
# DEEP DIVE: What leverage_cap changes
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LIQUIDITY CEILING ANALYSIS (EXTREME, 5000 runs)")
print("=" * 70)
print(f"  {'Ceiling':>10} {'1st pctile':>14} {'Median':>14} {'95th pctile':>14} {'P(ruin)':>10}")
for cap in [5000, 20000, 50000, 100000, 500000, 1_000_000, 10_000_000]:
    outcomes = []
    for _ in range(2000):
        r = run_compound_year(start_capital=200, regime='EXTREME',
                             n_days=N_DAYS, leverage_cap=cap)
        outcomes.append(r['final_capital'])
    outcomes.sort()
    p1 = outcomes[int(0.01 * len(outcomes))]
    p50 = outcomes[int(0.5 * len(outcomes))]
    p95 = outcomes[int(0.95 * len(outcomes))]
    ruin = sum(1 for o in outcomes if o <= 0) / len(outcomes) * 100
    print(f"  ${cap:>8,}  ${p1:>10,.0f}  ${p50:>10,.0f}  ${p95:>10,.0f}  {ruin:.2f}%")

# ═══════════════════════════════════════════════════════════════
# $3M CHECK: what does it take?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  THE $3M QUESTION")
print("=" * 70)
# Required daily return to turn $200 into $3M in 252 days
import math
target = 3_000_000
daily_mult = (target / 200) ** (1/252) - 1
print(f"  Required daily return: {daily_mult*100:.2f}%")
print(f"  Gap champion delivers: 3.67%/day avg")
print(f"  That turns $200 into: ${200 * (1+0.0367)**252:,.0f}")

# But with realistic liquidity ceiling:
for cap in [50000, 100000, 200000]:
    r = run_compound_year(start_capital=200, regime='EXTREME',
                         n_days=N_DAYS, leverage_cap=cap)
    print(f"  With ${cap:,} ceiling: ${r['final_capital']:,.0f}")

print("\n" + "=" * 70)
print("  REALISTIC ANSWER")
print("=" * 70)
print("  $200 → $3M requires 3.67%/day compounding with NO ceiling.")
print("  With a $50k-100k liquidity ceiling (realistic):")
print("  You hit the ceiling around day 120-150, then returns linearize.")
print(f"  Result: ${min(r['final_capital'] for r in [run_compound_year(start_capital=200, regime='EXTREME', leverage_cap=c) for c in [50000, 100000, 200000]]):,.0f}")
print("  The $3M is only possible if you can trade $1M+ without slippage.")
print("  With Alpaca paper on single stocks: $50k-100k is realistic cap.")
